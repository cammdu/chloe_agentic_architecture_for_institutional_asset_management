# =============================================================================
# SELF-DRIVING PORTFOLIO: LLM INTERACTION INFRASTRUCTURE
# =============================================================================
# Implements the foundational utility functions for secure, deterministic, and
# mathematically rigorous interactions with advanced reasoning models (GPT-5.2)
# as required by the agentic Strategic Asset Allocation (SAA) pipeline described in:
#   Ang, Azimbayev, and Kim (2026) — "The Self-Driving Portfolio"
#
# This module enforces the strict separation of concerns: LLMs handle judgment
# and orchestration, while Python scripts handle arithmetic. It explicitly
# discards legacy decoding parameters (temperature, top_p) in favor of
# reasoning effort allocations, and mandates the 'developer' role for system
# instructions.
#
# All functions are purely deterministic Python callables.
# =============================================================================

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import anthropic
from openai import APIError, Client, RateLimitError

from agent_claude_adapter import ClaudeBudgetExceeded, ClaudeChatAdapter, build_claude_adapter

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
# Initialise a named logger so callers can configure log levels independently
logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# CALLABLE 1: AgentLLMConfig (Dataclass)
# =============================================================================

@dataclass(frozen=True)
class AgentLLMConfig:
    """
    A strictly typed, immutable configuration object for GPT-5.2 reasoning agents.

    This dataclass encapsulates the specific parameters required to invoke advanced
    reasoning models via the OpenAI SDK. It explicitly discards legacy decoding
    parameters (temperature, top_p) in favor of reasoning effort allocations.

    Attributes:
        model_name (str): The exact model identifier (e.g., 'gpt-5.2').
        reasoning_effort (Literal['low', 'medium', 'high', 'xhigh']): The compute
            allocation for the reasoning trace.
        max_completion_tokens (int): The absolute ceiling for generated tokens.
    """
    # The string identifier for the OpenAI model
    model_name: str
    # The compute allocation parameter specific to reasoning models
    reasoning_effort: Literal['low', 'medium', 'high', 'xhigh']
    # The maximum number of tokens the model is permitted to generate
    max_completion_tokens: int

    # The post-initialization hook to enforce runtime validation
    def __post_init__(self) -> None:
        # Define the mathematically and operationally valid set of effort levels
        valid_efforts = {'low', 'medium', 'high', 'xhigh'}
        # Check if the provided reasoning_effort is within the valid set
        if self.reasoning_effort not in valid_efforts:
            # Raise a ValueError immediately if an invalid effort level is detected
            raise ValueError(
                f"Invalid reasoning_effort: '{self.reasoning_effort}'. "
                f"Must be one of {valid_efforts}."
            )
        # Check if the max_completion_tokens is a strictly positive integer
        if self.max_completion_tokens <= 0:
            # Raise a ValueError if the token limit is mathematically invalid
            raise ValueError(
                f"max_completion_tokens must be strictly positive, "
                f"got {self.max_completion_tokens}."
            )


# =============================================================================
# CALLABLE 2: initialize_openai_client
# =============================================================================

def initialize_openai_client(api_key_env_var: str = "OPENAI_API_KEY") -> Client:
    """
    Securely initializes and returns an OpenAI Client instance.

    This function enforces strict Information Security protocols by requiring
    the API key to be injected via environment variables. Hardcoded credentials
    will result in immediate pipeline failure.

    Parameters
    ----------
    api_key_env_var : str, optional
        The exact name of the environment variable containing the OpenAI API key.
        Defaults to 'OPENAI_API_KEY'.

    Returns
    -------
    Client
        A fully instantiated and authenticated OpenAI synchronous client.

    Raises
    ------
    RuntimeError
        If the specified environment variable is missing or empty.
    """
    # Attempt to retrieve the API key from the operating system environment
    api_key: str | None = os.environ.get(api_key_env_var)

    # Evaluate if the retrieved API key is None or an empty string
    if not api_key:
        # Raise a critical RuntimeError to halt the pipeline immediately
        logger.error("Failed to retrieve API key from environment variable: '%s'", api_key_env_var)
        raise RuntimeError(
            f"Critical Security/Configuration Error: The environment variable "
            f"'{api_key_env_var}' is not set or is empty. Fiduciary pipelines "
            f"require secure credential injection."
        )

    # Instantiate the OpenAI Client using the securely retrieved API key
    client = Client(api_key=api_key)

    logger.info("Successfully initialized OpenAI Client using env var: '%s'", api_key_env_var)

    # Return the authenticated client to the caller for dependency injection
    return client


# =============================================================================
# CALLABLE 2b: initialize_llm_client
# =============================================================================

def initialize_llm_client(config: Any) -> Any:
    """
    Returns the LLM client named by STUDY_CONFIG['INFRASTRUCTURE']['LLM_PROVIDER'].

    provider 'anthropic' returns a ClaudeChatAdapter (reads ANTHROPIC_API_KEY);
    provider 'openai' (or no LLM_PROVIDER block) returns an OpenAI Client
    (reads the env var named in LLM_INVOCATION_POLICY). Both are accepted by
    invoke_and_extract_agent_response.
    """
    infrastructure = config["INFRASTRUCTURE"]
    provider_cfg = infrastructure.get("LLM_PROVIDER", {})
    if provider_cfg.get("provider", "openai") == "anthropic":
        adapter = build_claude_adapter(config)
        # Touch the client now so a missing key fails before the pipeline starts
        _ = adapter.client
        logger.info("Successfully initialized Claude client for model '%s'", adapter.model)
        return adapter
    return initialize_openai_client(
        infrastructure["LLM_INVOCATION_POLICY"].get("api_key_env_var", "OPENAI_API_KEY")
    )


# =============================================================================
# CALLABLE 3: format_reasoning_messages
# =============================================================================

def format_reasoning_messages(
    developer_instruction: str,
    user_query: str
) -> List[Dict[str, str]]:
    """
    Constructs the strict message array required for GPT-5.2 reasoning models.

    Legacy models utilized the 'system' role for overarching instructions.
    Advanced reasoning models mandate the 'developer' role for the operational
    boundary (e.g., the IPS constraints and Skill definitions).

    Parameters
    ----------
    developer_instruction : str
        The overarching mandate, constraints, and skill definitions (e.g., the
        Prompt Template).
    user_query : str
        The specific task or data payload for the current iteration.

    Returns
    -------
    List[Dict[str, str]]
        The formatted message array ready for API serialization.

    Raises
    ------
    ValueError
        If either the developer_instruction or user_query is empty.
    """
    # Strip whitespace to ensure the developer instruction is not functionally empty
    clean_developer: str = developer_instruction.strip()
    # Strip whitespace to ensure the user query is not functionally empty
    clean_user: str = user_query.strip()

    # Validate that the developer instruction contains actual content
    if not clean_developer:
        # Raise a ValueError if the operational boundary is missing
        logger.error("format_reasoning_messages: developer_instruction is empty.")
        raise ValueError("The developer_instruction cannot be empty.")

    # Validate that the user query contains actual content
    if not clean_user:
        # Raise a ValueError if the task payload is missing
        logger.error("format_reasoning_messages: user_query is empty.")
        raise ValueError("The user_query cannot be empty.")

    # Initialize the message array with the developer role dictionary
    messages: List[Dict[str, str]] = [
        {"role": "developer", "content": clean_developer},
        # Append the user role dictionary containing the specific task
        {"role": "user", "content": clean_user}
    ]

    # Return the strictly formatted message array
    return messages


# =============================================================================
# CALLABLE 4: ParsedLLMResponse (Dataclass)
# =============================================================================

@dataclass(frozen=True)
class ParsedLLMResponse:
    """
    A strictly typed, immutable container for the extracted LLM response.

    This structure isolates the raw text content from deterministic tool calls,
    allowing the orchestrator to route mathematical operations to Python scripts
    while logging the reasoning trace.

    Attributes:
        content (Optional[str]): The natural language reasoning trace or narrative.
        tool_calls (Optional[List[Any]]): The array of requested tool invocations.
    """
    # The natural language output from the model, which may be None if only tools are called
    content: Optional[str]
    # The list of tool call objects, which may be None if no tools are invoked
    tool_calls: Optional[List[Any]]


# =============================================================================
# CALLABLE 5: invoke_and_extract_agent_response
# =============================================================================

def invoke_and_extract_agent_response(
    client: Client,
    config: AgentLLMConfig,
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None
) -> ParsedLLMResponse:
    """
    Executes the API call to the reasoning model and extracts the structured output.

    This function acts as the secure gateway between the local Python environment
    and the OpenAI API. It enforces the use of reasoning-specific parameters and
    handles network-level exceptions with clinical precision.

    Parameters
    ----------
    client : Client
        The authenticated OpenAI client instance.
    config : AgentLLMConfig
        The strictly typed configuration for the agent.
    messages : List[Dict[str, str]]
        The formatted message array.
    tools : Optional[List[Dict[str, Any]]], optional
        The JSON schema array of available Python functions the model is permitted
        to invoke. Defaults to None.

    Returns
    -------
    ParsedLLMResponse
        The immutable container holding the text and tool calls.

    Raises
    ------
    RuntimeError
        If the API call fails due to rate limits or server errors.
    """
    # Route to Claude when the pipeline was given a ClaudeChatAdapter
    if isinstance(client, ClaudeChatAdapter):
        return _invoke_claude(client, config, messages, tools)

    # Begin the try block to catch and handle specific network and API exceptions
    try:
        # Execute the synchronous API call to the chat completions endpoint
        response = client.chat.completions.create(
            # Inject the exact model identifier from the configuration
            model=config.model_name,
            # Inject the formatted message array containing developer and user roles
            messages=messages,
            # Inject the compute allocation specific to reasoning models
            reasoning_effort=config.reasoning_effort,
            # Inject the absolute ceiling for token generation
            max_completion_tokens=config.max_completion_tokens,
            # Inject the tool schemas, defaulting to None if no tools are provided
            tools=tools
        )

        # Extract the primary message object from the first choice in the response array
        message_obj = response.choices[0].message

        # Extract the natural language content, which may be None
        extracted_content: Optional[str] = message_obj.content

        # Extract the array of tool calls, which may be None
        extracted_tool_calls: Optional[List[Any]] = message_obj.tool_calls

        logger.debug(
            "Successfully invoked %s. Content length: %s, Tool calls: %s",
            config.model_name,
            len(extracted_content) if extracted_content else 0,
            len(extracted_tool_calls) if extracted_tool_calls else 0
        )

        # Instantiate and return the immutable ParsedLLMResponse dataclass
        return ParsedLLMResponse(
            content=extracted_content,
            tool_calls=extracted_tool_calls
        )

    # Catch rate limit exceptions specifically to allow upstream orchestrators to backoff
    except RateLimitError as rle:
        # Raise a descriptive RuntimeError encapsulating the rate limit failure
        logger.error("RateLimitError encountered during invocation of %s", config.model_name)
        raise RuntimeError(
            f"API Rate Limit Exceeded during invocation of {config.model_name}. "
            f"Details: {str(rle)}"
        ) from rle

    # Catch general API errors (e.g., 500 Internal Server Error, 503 Bad Gateway)
    except APIError as apie:
        # Raise a descriptive RuntimeError encapsulating the server-side failure
        logger.error("APIError encountered during invocation of %s", config.model_name)
        raise RuntimeError(
            f"OpenAI API Error encountered during invocation of {config.model_name}. "
            f"Details: {str(apie)}"
        ) from apie

    # Catch any other unexpected exceptions to prevent silent pipeline corruption
    except Exception as e:
        # Raise a descriptive RuntimeError for unhandled edge cases
        logger.error("Unexpected error during invocation of %s", config.model_name)
        raise RuntimeError(
            f"Unexpected error during LLM invocation and extraction. "
            f"Details: {str(e)}"
        ) from e


def _invoke_claude(
    adapter: ClaudeChatAdapter,
    config: AgentLLMConfig,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> ParsedLLMResponse:
    """Claude branch of invoke_and_extract_agent_response (same inputs and outputs)."""
    try:
        response = adapter.complete(
            messages=messages,
            tools=tools,
            effort=config.reasoning_effort,
            max_tokens=config.max_completion_tokens,
        )
    except ClaudeBudgetExceeded:
        # Let the spend cap stop the pipeline with its own message
        raise
    except anthropic.RateLimitError as rle:
        logger.error("RateLimitError encountered during invocation of %s", adapter.model)
        raise RuntimeError(
            f"API Rate Limit Exceeded during invocation of {adapter.model}. Details: {str(rle)}"
        ) from rle
    except anthropic.APIError as apie:
        logger.error("APIError encountered during invocation of %s", adapter.model)
        raise RuntimeError(
            f"Anthropic API Error encountered during invocation of {adapter.model}. Details: {str(apie)}"
        ) from apie
    except Exception as e:
        logger.error("Unexpected error during invocation of %s", adapter.model)
        raise RuntimeError(
            f"Unexpected error during LLM invocation and extraction. Details: {str(e)}"
        ) from e

    message_obj = response.choices[0].message
    return ParsedLLMResponse(content=message_obj.content, tool_calls=message_obj.tool_calls)
