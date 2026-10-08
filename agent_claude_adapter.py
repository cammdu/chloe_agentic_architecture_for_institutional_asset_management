# =============================================================================
# SELF-DRIVING PORTFOLIO: CLAUDE (ANTHROPIC) LLM ADAPTER
# =============================================================================
# Lets the pipeline run on Claude instead of OpenAI without rewriting the
# agent loops. The notebook builds conversations in the OpenAI chat format
# (developer/user/assistant/tool messages, "function" tool schemas); this
# module translates them to the Anthropic Messages API, calls Claude, and
# translates the reply back into OpenAI-shaped objects.
#
# It serves both LLM paths in the notebook:
#   1. The direct ReAct loops (Tasks 17, 18, 22), through
#      ClaudeChatAdapter, which agent_llm_infrastructure.invoke_and_extract_
#      agent_response accepts in place of an OpenAI client.
#   2. The AutoGen agents (Tasks 19, 27-31), through ClaudeAutogenClient, a
#      custom AG2 model client selected by build_claude_autogen_llm_config.
#
# Claude-specific behaviour handled here:
#   - Thinking is always on for current Claude models, and the thinking blocks
#     of each reply must be sent back unchanged on the next turn. The agent
#     loops only keep text and tool calls, so the raw reply blocks are cached
#     and substituted back in when the same assistant turn is replayed.
#   - Reasoning effort maps to output_config.effort; thinking tokens count
#     towards max_tokens, so small OpenAI token budgets are raised to a floor.
#   - Forced tool_choice is not supported, so it is downgraded to "auto".
#   - Every call updates a running spend tally, and an optional spend/call cap
#     stops the run before it can exceed a budget.
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from openai.types.chat import ChatCompletion

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)

# Name the AutoGen config_list refers to; must equal ClaudeAutogenClient.__name__
CLAUDE_AUTOGEN_CLIENT_NAME: str = "ClaudeAutogenClient"

# USD per million tokens (input, output). Cache reads bill at 0.1x input and
# cache writes at 1.25x input. Used only for the spend tally and the cap.
CLAUDE_PRICES_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# OpenAI effort names that Claude does not have, mapped to the nearest level
_EFFORT_MAP: Dict[str, str] = {"none": "low", "minimal": "low"}

# Map Claude stop reasons onto OpenAI finish_reason values
_FINISH_REASON_MAP: Dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}


class ClaudeBudgetExceeded(RuntimeError):
    """Raised before a call that would go past the configured spend or call cap."""


# =============================================================================
# Spend tracking
# =============================================================================

@dataclass
class ClaudeSpendTracker:
    """Running totals of Claude calls, tokens and estimated USD cost."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model: str, usage: Any) -> float:
        # Price unknown models at the most expensive listed rate so the cap stays safe
        price_in, price_out = CLAUDE_PRICES_PER_MTOK.get(
            _base_model_name(model), max(CLAUDE_PRICES_PER_MTOK.values())
        )
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cost = (
            input_tokens * price_in
            + cache_read * price_in * 0.1
            + cache_write * price_in * 1.25
            + output_tokens * price_out
        ) / 1_000_000
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cache_read_tokens += cache_read
            self.cache_write_tokens += cache_write
            self.cost_usd += cost
        return cost

    def summary(self) -> str:
        return (
            f"{self.calls} Claude calls, "
            f"{self.input_tokens + self.cache_read_tokens + self.cache_write_tokens:,} input tokens "
            f"({self.cache_read_tokens:,} from cache), {self.output_tokens:,} output tokens, "
            f"estimated cost ${self.cost_usd:.4f}"
        )


# One tracker for the whole process, shared by both LLM paths
SPEND: ClaudeSpendTracker = ClaudeSpendTracker()

# Optional caps shared by both LLM paths (None = no cap)
_LIMITS: Dict[str, Optional[float]] = {"max_spend_usd": None, "max_calls": None}


def set_claude_limits(max_spend_usd: Optional[float] = None, max_calls: Optional[int] = None) -> None:
    """Stop the run (ClaudeBudgetExceeded) once spend or call count reaches a cap."""
    _LIMITS["max_spend_usd"] = max_spend_usd
    _LIMITS["max_calls"] = max_calls


def reset_claude_spend() -> None:
    """Zero the running spend tally (caps are measured against it)."""
    global SPEND
    SPEND = ClaudeSpendTracker()


def _check_limits() -> None:
    max_spend = _LIMITS["max_spend_usd"]
    max_calls = _LIMITS["max_calls"]
    if max_spend is not None and SPEND.cost_usd >= max_spend:
        raise ClaudeBudgetExceeded(
            f"Claude spend cap reached (${SPEND.cost_usd:.4f} of ${max_spend:.2f}); stopping before the next call."
        )
    if max_calls is not None and SPEND.calls >= max_calls:
        raise ClaudeBudgetExceeded(
            f"Claude call cap reached ({SPEND.calls} of {int(max_calls)}); stopping before the next call."
        )


def _base_model_name(model: str) -> str:
    # Strip dated suffixes such as claude-haiku-4-5-20251001
    for name in CLAUDE_PRICES_PER_MTOK:
        if model.startswith(name):
            return name
    return model


# =============================================================================
# Raw reply cache (preserves thinking blocks across turns)
# =============================================================================

# Maps a tool_use id, or the text of a text-only reply, to the exact content
# blocks Claude returned for that assistant turn.
_RAW_ASSISTANT_BLOCKS: Dict[str, List[Dict[str, Any]]] = {}
_RAW_LOCK = threading.Lock()


def _remember_raw_blocks(blocks: List[Dict[str, Any]]) -> None:
    keys: List[str] = [b["id"] for b in blocks if b.get("type") == "tool_use"]
    if not keys:
        text = _join_text(blocks)
        if text:
            keys = ["text:" + text]
    with _RAW_LOCK:
        for key in keys:
            _RAW_ASSISTANT_BLOCKS[key] = blocks


def _lookup_raw_blocks(text: Optional[str], tool_call_ids: List[str]) -> Optional[List[Dict[str, Any]]]:
    with _RAW_LOCK:
        if tool_call_ids:
            return _RAW_ASSISTANT_BLOCKS.get(tool_call_ids[0])
        if text:
            return _RAW_ASSISTANT_BLOCKS.get("text:" + text)
    return None


def _join_text(blocks: List[Dict[str, Any]]) -> str:
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


# =============================================================================
# OpenAI format -> Anthropic format
# =============================================================================

def _get(obj: Any, key: str, default: Any = None) -> Any:
    # Messages and tool calls arrive as dicts (AutoGen) or as openai objects (direct loops)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def openai_tools_to_claude(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI {"type": "function", "function": {...}} schemas to Claude tool definitions."""
    if not tools:
        return None
    converted = []
    for tool in tools:
        fn = tool.get("function", tool)
        converted.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return converted


def openai_tool_choice_to_claude(tool_choice: Any) -> Optional[Dict[str, Any]]:
    """Map OpenAI tool_choice to Claude. Forced choices become "auto" (not supported on current Claude models)."""
    if tool_choice is None:
        return None
    if tool_choice == "none":
        return {"type": "none"}
    if tool_choice != "auto":
        logger.warning("Forced tool_choice %r is not supported by Claude; using 'auto' instead.", tool_choice)
    return {"type": "auto"}


def _assistant_blocks(message: Any) -> List[Dict[str, Any]]:
    text = _content_to_text(_get(message, "content"))
    tool_calls = _get(message, "tool_calls") or []
    ids = [_get(tc, "id") for tc in tool_calls]

    cached = _lookup_raw_blocks(text, ids)
    if cached is not None:
        return cached

    # Not produced by this process (or edited): rebuild from text + tool calls
    blocks: List[Dict[str, Any]] = []
    if text.strip():
        blocks.append({"type": "text", "text": text})
    for tc in tool_calls:
        fn = _get(tc, "function")
        raw_args = _get(fn, "arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            args = {"_raw_arguments": raw_args}
        blocks.append({"type": "tool_use", "id": _get(tc, "id"), "name": _get(fn, "name"), "input": args})
    return blocks


def openai_messages_to_claude(messages: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert an OpenAI chat message list into (system, messages) for Claude.

    developer/system messages become the system prompt; tool results become
    tool_result blocks in a user turn; consecutive same-role turns are merged.
    """
    system_parts: List[str] = []
    turns: List[Dict[str, Any]] = []

    def _append(role: str, blocks: List[Dict[str, Any]]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": list(blocks)})

    for message in messages:
        role = _get(message, "role")
        if role in ("system", "developer"):
            text = _content_to_text(_get(message, "content")).strip()
            if text:
                system_parts.append(text)
        elif role == "assistant":
            _append("assistant", _assistant_blocks(message))
        elif role == "tool":
            # AutoGen may bundle several results into one message under tool_responses
            responses = _get(message, "tool_responses") or [message]
            _append("user", [
                {
                    "type": "tool_result",
                    "tool_use_id": _get(r, "tool_call_id"),
                    "content": _content_to_text(_get(r, "content")) or "(no output)",
                }
                for r in responses
            ])
        else:
            # user, function (legacy) or anything else is treated as user text
            text = _content_to_text(_get(message, "content"))
            _append("user", [{"type": "text", "text": text if text.strip() else "(empty message)"}])

    # Claude conversations must start with a user turn and must not end on an assistant turn
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": [{"type": "text", "text": "Begin."}]})
    if turns[-1]["role"] == "assistant":
        turns.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})

    return "\n\n".join(system_parts), turns


# =============================================================================
# Anthropic reply -> OpenAI ChatCompletion
# =============================================================================

def _to_chat_completion(reply: Any, model: str) -> ChatCompletion:
    blocks = [b.model_dump(mode="json", exclude_none=True) for b in reply.content]
    _remember_raw_blocks(blocks)

    text = _join_text(blocks)
    tool_calls = [
        {
            "id": b["id"],
            "type": "function",
            "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
        }
        for b in blocks if b.get("type") == "tool_use"
    ]

    usage = reply.usage
    prompt_tokens = (
        (usage.input_tokens or 0)
        + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    )
    return ChatCompletion.model_validate({
        "id": reply.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "finish_reason": _FINISH_REASON_MAP.get(reply.stop_reason or "", "stop"),
            "message": {
                "role": "assistant",
                "content": text or None,
                "tool_calls": tool_calls or None,
            },
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": usage.output_tokens or 0,
            "total_tokens": prompt_tokens + (usage.output_tokens or 0),
        },
    })


# =============================================================================
# CALLABLE: ClaudeChatAdapter
# =============================================================================

class ClaudeChatAdapter:
    """
    Calls Claude with OpenAI-format messages and returns an OpenAI ChatCompletion.

    Parameters
    ----------
    model : str
        Claude model id, e.g. 'claude-opus-5-5'.
    api_key_env_var : str
        Environment variable holding the Anthropic API key.
    min_max_tokens : int
        Floor applied to max_tokens. Claude's thinking counts towards
        max_tokens, so the small budgets written for GPT-5.2 are raised.
        Only tokens actually generated are billed.
    client : anthropic.Anthropic, optional
        Pre-built client (used by tests); built from the env var otherwise.
    """

    def __init__(
        self,
        model: str,
        api_key_env_var: str = "ANTHROPIC_API_KEY",
        min_max_tokens: int = 32000,
        client: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.min_max_tokens = min_max_tokens
        self.api_key_env_var = api_key_env_var
        self._client = client

    @property
    def client(self) -> Any:
        # Built on first use, so agents can be constructed before a key is loaded
        if self._client is None:
            api_key = os.environ.get(self.api_key_env_var)
            if not api_key:
                raise RuntimeError(
                    f"The environment variable '{self.api_key_env_var}' is not set or is empty. "
                    f"Put your Claude API key in a .env file (never in the notebook or on GitHub)."
                )
            self._client = anthropic.Anthropic(api_key=api_key, max_retries=4)
        return self._client

    def complete(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        effort: str = "high",
        max_tokens: Optional[int] = None,
        tool_choice: Any = None,
    ) -> ChatCompletion:
        _check_limits()

        system, claude_messages = openai_messages_to_claude(messages)
        request: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(max_tokens or 0, self.min_max_tokens),
            "messages": claude_messages,
            "output_config": {"effort": _EFFORT_MAP.get(effort, effort)},
            # Automatic prompt caching: each call re-reads the growing conversation from cache
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            request["system"] = system
        claude_tools = openai_tools_to_claude(tools)
        if claude_tools:
            request["tools"] = claude_tools
            claude_tool_choice = openai_tool_choice_to_claude(tool_choice)
            if claude_tool_choice:
                request["tool_choice"] = claude_tool_choice

        # Stream so long, high-effort replies don't hit request timeouts
        with self.client.messages.stream(**request) as stream:
            reply = stream.get_final_message()

        cost = SPEND.record(self.model, reply.usage)
        logger.info(
            "Claude call %d (%s, effort=%s): stop=%s, cost $%.4f, running total $%.4f",
            SPEND.calls, self.model, request["output_config"]["effort"], reply.stop_reason, cost, SPEND.cost_usd,
        )
        if reply.stop_reason == "max_tokens":
            logger.warning("Claude reply hit max_tokens (%d); output may be cut off.", request["max_tokens"])
        if reply.stop_reason == "refusal":
            logger.warning("Claude declined to answer this request (stop_reason=refusal).")

        return _to_chat_completion(reply, self.model)


def build_claude_adapter(config: Any, client: Optional[Any] = None) -> ClaudeChatAdapter:
    """Build a ClaudeChatAdapter from STUDY_CONFIG['INFRASTRUCTURE']['LLM_PROVIDER']."""
    provider_cfg = config["INFRASTRUCTURE"]["LLM_PROVIDER"]
    return ClaudeChatAdapter(
        model=provider_cfg["anthropic_model"],
        api_key_env_var=provider_cfg.get("anthropic_api_key_env_var", "ANTHROPIC_API_KEY"),
        min_max_tokens=int(provider_cfg.get("anthropic_min_max_tokens", 32000)),
        client=client,
    )


# =============================================================================
# AutoGen (AG2) integration
# =============================================================================

# Optional pre-built Anthropic client shared by every AutoGen agent (used by tests)
_AUTOGEN_SHARED_CLIENT: Dict[str, Any] = {"client": None}


class ClaudeAutogenClient:
    """AG2 custom model client that routes an agent's LLM calls to Claude."""

    RESPONSE_USAGE_KEYS: List[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        self._effort = config.get("effort", "high")
        self._adapter = ClaudeChatAdapter(
            model=config["model"],
            api_key_env_var=config.get("api_key_env_var", "ANTHROPIC_API_KEY"),
            min_max_tokens=int(config.get("min_max_tokens", 32000)),
            client=_AUTOGEN_SHARED_CLIENT["client"],
        )

    def create(self, params: Dict[str, Any]) -> ChatCompletion:
        response = self._adapter.complete(
            messages=params["messages"],
            tools=params.get("tools"),
            effort=params.get("effort", self._effort),
            max_tokens=params.get("max_tokens"),
            tool_choice=params.get("tool_choice"),
        )
        # AG2 reads response.cost when it totals usage
        response.cost = self.cost(response)  # type: ignore[attr-defined]
        return response

    def message_retrieval(self, response: ChatCompletion) -> List[Any]:
        return [choice.message for choice in response.choices]

    def cost(self, response: ChatCompletion) -> float:
        price_in, price_out = CLAUDE_PRICES_PER_MTOK.get(
            _base_model_name(response.model), max(CLAUDE_PRICES_PER_MTOK.values())
        )
        usage = response.usage
        if usage is None:
            return 0.0
        return (usage.prompt_tokens * price_in + usage.completion_tokens * price_out) / 1_000_000

    @staticmethod
    def get_usage(response: ChatCompletion) -> Dict[str, Any]:
        usage = response.usage
        return {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "cost": getattr(response, "cost", 0.0),
            "model": response.model,
        }


_AUTOGEN_HOOK_INSTALLED: bool = False


def enable_claude_for_autogen() -> None:
    """
    Make every AG2 agent whose config_list names ClaudeAutogenClient use it.

    AG2 normally requires agent.register_model_client(...) after each agent is
    built, and again whenever tools are registered (which rebuilds the
    client). Patching OpenAIWrapper once activates the Claude client
    everywhere, so the notebook's agent factories need no changes.
    """
    global _AUTOGEN_HOOK_INSTALLED
    if _AUTOGEN_HOOK_INSTALLED:
        return
    from autogen.oai import client as ag2_client

    original_init = ag2_client.OpenAIWrapper.__init__

    def _init_with_claude(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        for i, model_client in enumerate(self._clients):
            if (
                isinstance(model_client, ag2_client.PlaceHolderClient)
                and model_client.config.get("model_client_cls") == CLAUDE_AUTOGEN_CLIENT_NAME
            ):
                self._clients[i] = ClaudeAutogenClient(model_client.config)

    ag2_client.OpenAIWrapper.__init__ = _init_with_claude
    _AUTOGEN_HOOK_INSTALLED = True


def build_claude_autogen_llm_config(
    provider_cfg: Any,
    effort: str,
    max_output_tokens: int,
    force_tool_choice: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an AG2 llm_config that sends this agent's calls to Claude."""
    enable_claude_for_autogen()
    if force_tool_choice is not None:
        logger.info("Forced tool_choice is not supported by Claude; the agent's prompt asks for the tool call instead.")
    return {
        "config_list": [{
            "model": provider_cfg["anthropic_model"],
            "model_client_cls": CLAUDE_AUTOGEN_CLIENT_NAME,
            "api_key_env_var": provider_cfg.get("anthropic_api_key_env_var", "ANTHROPIC_API_KEY"),
            "min_max_tokens": int(provider_cfg.get("anthropic_min_max_tokens", 32000)),
            "effort": _EFFORT_MAP.get(effort, effort),
            "max_tokens": int(max_output_tokens),
        }],
        "cache_seed": None,
    }
