"""
Backend abstraction layer for the agent system.

This module defines the :class:`AgentBackend` interface and concrete
implementations for different LLM coding-agent backends. Each backend wraps
a specific tool (OpenCode, Claude Agent SDK) behind a unified synchronous
API so that the rest of the agent system can switch backends transparently.

Use the :func:`get_backend` factory function to instantiate a backend by name.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional


#: Type alias for the ``on_event`` callback that receives intermediate agent
#: steps. Events are dicts with at least a ``"type"`` key. Common event types:
#:
#: - ``{"type": "text", "content": "..."}`` — agent narration
#: - ``{"type": "tool_call", "name": "Bash", "input": "..."}`` — tool invocation
#: - ``{"type": "tool_result", "name": "Bash", "output": "..."}`` — tool result
EventCallback = Callable[[Dict[str, Any]], None]


class AgentBackend:
    """
    Abstract base class for LLM agent backends.

    Concrete subclasses must implement :meth:`send_message` to dispatch a
    prompt to a specific LLM backend and return the final text response.
    Each subclass declares a :attr:`PROMPT_VARIANT` that determines which
    set of prompt templates to use (see :mod:`~chem_mat_data.agent.prompts`).
    """

    #: Identifies which prompt variant this backend uses.
    #: Must be set by subclasses (e.g. ``"opencode"``, ``"claude"``).
    PROMPT_VARIANT: str = ""

    def send_message(
        self,
        prompt: str,
        timeout: int = 120,
        on_event: Optional[EventCallback] = None,
    ) -> str:
        """
        Send a prompt to the LLM backend and return the final text response.

        :param prompt: The full prompt text to send.
        :param timeout: Maximum time in seconds to wait for a response.
        :param on_event: Optional callback invoked for intermediate agent
            steps. Not all backends support this — backends that do not
            stream intermediate steps simply ignore the callback.

        :returns: The final text response from the LLM.
        :raises RuntimeError: If the backend fails to produce a response.
        """
        raise NotImplementedError()


class OpenCodeBackend(AgentBackend):
    """
    Backend that delegates to a locally running OpenCode server.

    Wraps the existing :mod:`~chem_mat_data.agent.opencode_client` HTTP
    client. The ``on_event`` callback is accepted but never called because
    the OpenCode HTTP API does not stream intermediate steps.
    """

    PROMPT_VARIANT = "opencode"

    def __init__(self, agent: str | None = None):
        """
        :param agent: Optional OpenCode agent name that selects a permission
            profile on the server (e.g. ``"dataset-links"``).
        """
        self.agent = agent

    def send_message(
        self,
        prompt: str,
        timeout: int = 120,
        on_event: Optional[EventCallback] = None,
    ) -> str:
        """
        Send a prompt to the OpenCode server and return the response text.

        :param prompt: The prompt text.
        :param timeout: Timeout in seconds.
        :param on_event: Ignored — OpenCode does not support streaming.

        :returns: The text response.
        :raises RuntimeError: If OpenCode returns an empty response.
        """
        from chem_mat_data.agent.opencode_client import send_message

        result = send_message(prompt, timeout=timeout, agent=self.agent)
        if not result:
            raise RuntimeError("OpenCode returned an empty response")
        return result


class ClaudeBackend(AgentBackend):
    """
    Backend that uses the Claude Agent SDK.

    Wraps the async :func:`claude_agent_sdk.query` function with
    :func:`asyncio.run` to provide a synchronous interface. Intermediate
    assistant messages and tool calls are forwarded to the ``on_event``
    callback when provided.

    Authentication is resolved in the following order of precedence:

    1. ``api_key`` parameter (sets ``ANTHROPIC_API_KEY`` env var)
    2. ``oauth_token`` parameter (sets ``CLAUDE_CODE_OAUTH_TOKEN`` env var)
    3. Existing environment variables (``ANTHROPIC_API_KEY`` or
       ``CLAUDE_CODE_OAUTH_TOKEN``) — used as-is if already set.

    The OAuth token allows using a Claude Pro/Max subscription instead of
    an API account. Obtain it by running ``claude login`` followed by
    ``claude setup-token`` in your terminal.
    """

    PROMPT_VARIANT = "claude"

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        allowed_tools: List[str] | None = None,
        mcp_servers: Dict[str, Any] | None = None,
        api_key: str | None = None,
        oauth_token: str | None = None,
    ):
        """
        :param model: The Claude model identifier to use.
        :param allowed_tools: List of tools the agent is allowed to use.
            Defaults to a broad set of built-in tools when ``None``.
        :param mcp_servers: MCP server configuration dict passed to
            :class:`~claude_agent_sdk.ClaudeAgentOptions`.
        :param api_key: Anthropic API key. If provided, sets
            ``ANTHROPIC_API_KEY`` in the environment before calling the SDK.
        :param oauth_token: Claude Pro/Max OAuth token. If provided, sets
            ``CLAUDE_CODE_OAUTH_TOKEN`` in the environment.
        """
        self.model = model
        self.allowed_tools = allowed_tools
        self.mcp_servers = mcp_servers or {}
        self.api_key = api_key
        self.oauth_token = oauth_token

    def send_message(
        self,
        prompt: str,
        timeout: int = 120,
        on_event: Optional[EventCallback] = None,
    ) -> str:
        """
        Send a prompt to the Claude Agent SDK and return the final result.

        Internally runs an async query loop, forwarding intermediate
        text narrations and tool invocations to ``on_event``.

        :param prompt: The prompt text.
        :param timeout: Informational; the SDK manages its own timeouts.
        :param on_event: Callback for intermediate text and tool-call events.

        :returns: The final result text from the agent.
        :raises RuntimeError: If the agent produces no result.
        """
        self._configure_auth()
        return asyncio.run(self._query(prompt, on_event))

    def _configure_auth(self) -> None:
        """
        Set authentication environment variables from config values.

        Only sets variables that are not already present in the environment,
        so explicit env vars always take precedence over config file values.
        """
        if self.api_key and "ANTHROPIC_API_KEY" not in os.environ:
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
        if self.oauth_token and "CLAUDE_CODE_OAUTH_TOKEN" not in os.environ:
            os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = self.oauth_token

    async def _query(
        self,
        prompt: str,
        on_event: Optional[EventCallback],
    ) -> str:
        """
        Internal async implementation that iterates over the Claude Agent SDK
        message stream.
        """
        from claude_agent_sdk import (
            query,
            ClaudeAgentOptions,
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
        )

        options_kwargs: Dict[str, Any] = {
            "model": self.model,
            "permission_mode": "bypassPermissions",
        }
        if self.allowed_tools:
            options_kwargs["allowed_tools"] = self.allowed_tools
        if self.mcp_servers:
            options_kwargs["mcp_servers"] = self.mcp_servers

        options = ClaudeAgentOptions(**options_kwargs)

        final_text = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage) and on_event:
                for block in message.content:
                    if isinstance(block, TextBlock):
                        on_event({"type": "text", "content": block.text})
                    elif isinstance(block, ToolUseBlock):
                        on_event({
                            "type": "tool_call",
                            "name": block.name,
                            "input": str(block.input) if hasattr(block, "input") else "",
                        })
            elif isinstance(message, ResultMessage):
                final_text = message.result

        if not final_text:
            raise RuntimeError("Claude Agent SDK returned no result")
        return final_text


# =============================================================================
# Backend registry and factory
# =============================================================================

_BACKEND_REGISTRY: Dict[str, type] = {
    "opencode": OpenCodeBackend,
    "claude": ClaudeBackend,
}


def get_backend(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    agent: str | None = None,
) -> AgentBackend:
    """
    Instantiate an :class:`AgentBackend` by name.

    The ``config`` dict is expected to contain backend-specific settings from
    the ``[agent.<name>]`` section of the ChemMatData config file. For the
    ``"opencode"`` backend, the ``agent`` parameter selects a permission
    profile on the server. For ``"claude"``, settings like ``model`` and
    ``mcp_servers`` are read from ``config``.

    :param name: Backend identifier (``"opencode"`` or ``"claude"``).
    :param config: Backend-specific settings dict.
    :param agent: OpenCode agent name; only used by the ``"opencode"`` backend.

    :returns: A configured :class:`AgentBackend` instance.
    :raises ValueError: If ``name`` is not a recognised backend.
    """
    if name not in _BACKEND_REGISTRY:
        valid = ", ".join(sorted(_BACKEND_REGISTRY))
        raise ValueError(f"Unknown backend {name!r}. Valid backends: {valid}")

    config = config or {}

    if name == "opencode":
        return OpenCodeBackend(agent=agent)
    elif name == "claude":
        return ClaudeBackend(
            model=config.get("model", "claude-sonnet-4-20250514"),
            allowed_tools=config.get("allowed_tools"),
            mcp_servers=config.get("mcp_servers"),
            api_key=config.get("api_key"),
            oauth_token=config.get("oauth_token"),
        )

    # Fallback for future backends registered in _BACKEND_REGISTRY
    return _BACKEND_REGISTRY[name]()
