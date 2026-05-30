"""Cat 9a orchestrator drivers — real-model tool-use loops for the
invocation-rate sub-test (issue #194).

A *driver* is a callable ``(question_text) -> Cat9aQueryOutcome`` that
runs one question through a real model wired with the ``mempalace_search``
tool, and reports how many times the model invoked it. The scorer in
``sme.categories.harness_integration.run_cat9a`` is model-agnostic; all
the runtime weight (boto3 / the ollama server / an API key) lives here
and imports lazily, so the SME core stays dependency-light.

Two drivers, one shared backend, one shared tool definition — so the
invocation rate is measured *identically* across the Tau2 ladder:

  - ``BedrockOrchestrator``  — Anthropic tool-use loop over Bedrock
    (claude-opus-4-8 etc., via the ``us.anthropic.`` inference profile).
    The frontier high-Tau2 arm.
  - ``OllamaOrchestrator``   — OpenAI-compatible tool-calling loop over
    a local ollama server (qwen3.5:4b, gemma4:e4b). The low-cost
    contrast arms — and the exact two models from the documented
    reference_tau2_predicts_cat9a pair.

Memory backend: palace-daemon ``GET /search`` — READ-ONLY. A Cat 9a
probe must never ingest; these drivers only issue search queries.

The tool definition mirrors ``RlmAdapter``'s ``mempalace_search``
custom_tool so the orchestrator sees the same surface the 2026-04-30
baselines used.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional
from urllib import error as _urlerror
from urllib import parse as _urlparse
from urllib import request as _urlrequest

from sme.categories.harness_integration import Cat9aQueryOutcome

log = logging.getLogger(__name__)

_DEFAULT_LIMIT = 5
_DEFAULT_DAEMON_TIMEOUT = 15.0
_MAX_TOOL_ITERATIONS = 6  # safety cap on the agentic loop per question

# Canonical tool surface — kept byte-for-byte aligned with RlmAdapter's
# mempalace_search description so the orchestrator sees the same tool the
# original baselines exposed.
_TOOL_NAME = "mempalace_search"
_TOOL_DESCRIPTION = (
    "Search JP's palace for drawers semantically related to a query. "
    "Returns a list of dicts with text, wing, room, source_file, similarity. "
    "Default limit is 5. Use this to ground factual claims about JP, "
    "his projects, his realm, and any past events."
)


# --- Shared read-only daemon search backend ---------------------------


class DaemonSearch:
    """Read-only palace-daemon ``/search`` client shared by every driver.

    Tracks every query for audit (so a run can prove zero writes). The
    only HTTP verb issued is GET ``/search`` — there is no ingest path
    here by construction.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str = "",
        *,
        kind: str = "content",
        timeout_s: float = _DEFAULT_DAEMON_TIMEOUT,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.kind = kind
        self.timeout_s = timeout_s
        self.queries: list[str] = []

    def search(self, query: str, limit: int = _DEFAULT_LIMIT) -> list[dict]:
        self.queries.append(query)
        params = {"q": query, "limit": str(limit), "kind": self.kind}
        url = f"{self.api_url}/search?" + _urlparse.urlencode(params)
        req = _urlrequest.Request(url, method="GET")
        if self.api_key:
            req.add_header("x-api-key", self.api_key)
        try:
            with _urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (_urlerror.URLError, _urlerror.HTTPError, OSError) as e:
            return [{"error": str(e), "results": []}]
        results = payload.get("results", []) or []
        trimmed: list[dict] = []
        for r in results[:limit]:
            trimmed.append(
                {
                    "drawer_id": r.get("drawer_id") or r.get("id"),
                    "text": (r.get("text") or "")[:500],
                    "wing": r.get("wing"),
                    "room": r.get("room"),
                    "source_file": r.get("source_file"),
                    "similarity": r.get("similarity"),
                }
            )
        return trimmed


def _context_from_results(all_results: list[dict]) -> str:
    """Build the context_string the substring scorer reads, identical in
    shape to RlmAdapter's retrieval context (drawer tags + text)."""
    lines = [f"── Cat9a-orchestrated retrieval ({len(all_results)} drawers) ──"]
    for r in all_results:
        tags = []
        if r.get("drawer_id"):
            tags.append(f"drawer_id={r['drawer_id']}")
        if r.get("source_file"):
            tags.append(f"source_file={r['source_file']}")
        if r.get("wing"):
            tags.append(f"wing={r['wing']}")
        if r.get("room"):
            tags.append(f"room={r['room']}")
        sim = r.get("similarity")
        if isinstance(sim, (int, float)):
            tags.append(f"similarity={sim:.3f}")
        lines.append("[" + " · ".join(tags) + "]")
        lines.append(r.get("text", ""))
        lines.append("")
    return "\n".join(lines)


# --- Bedrock (Anthropic tool-use) driver ------------------------------

# Friendly model id -> Bedrock inference-profile id. On-demand throughput
# for opus-4.x needs the cross-region ``us.`` profile, not the bare model
# id (same mapping as sme/eval/answer_generator.py).
_CLAUDE_BEDROCK_IDS = {
    "claude-opus-4-8": "us.anthropic.claude-opus-4-8",
    "claude-opus-4-7": "us.anthropic.claude-opus-4-7",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5",
}


@dataclass
class BedrockOrchestrator:
    """Drive a Claude model over Bedrock through an Anthropic tool-use
    loop, counting mempalace_search invocations.

    The model is given the question and the search tool; it decides
    whether and how often to call it. We run the standard Anthropic
    agentic loop (assistant emits tool_use → we run the tool → feed back
    tool_result) until the model stops requesting tools or the iteration
    cap is hit. ``tool_calls`` is the count of tool_use blocks the model
    emitted across the whole loop — the real 9a signal.
    """

    model: str
    backend: DaemonSearch
    max_tokens: int = 1024
    region: Optional[str] = None
    _client: Any = None

    def __post_init__(self) -> None:
        from anthropic import AnthropicBedrock  # lazy — optional dep

        region = (
            self.region
            or os.environ.get("BEDROCK_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-west-1"
        )
        self._client = AnthropicBedrock(aws_region=region)

    @property
    def _tool_schema(self) -> dict:
        return {
            "name": _TOOL_NAME,
            "description": _TOOL_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {"type": "integer", "description": "Max drawers (default 5)."},
                },
                "required": ["query"],
            },
        }

    def __call__(self, question: str) -> Cat9aQueryOutcome:
        bid = _CLAUDE_BEDROCK_IDS.get(self.model, self.model)
        messages: list[dict] = [{"role": "user", "content": question}]
        tool_calls = 0
        captured: list[dict] = []
        answer_parts: list[str] = []

        for _ in range(_MAX_TOOL_ITERATIONS):
            resp = self._client.messages.create(
                model=bid,
                max_tokens=self.max_tokens,
                tools=[self._tool_schema],
                messages=messages,
            )
            # Collect text + tool_use blocks from this assistant turn.
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            for b in resp.content:
                if getattr(b, "type", None) == "text":
                    answer_parts.append(b.text)

            if not tool_uses or resp.stop_reason != "tool_use":
                break

            # Append assistant turn, then run each tool and feed results back.
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for tu in tool_uses:
                tool_calls += 1
                args = tu.input or {}
                query = args.get("query", question)
                limit = int(args.get("limit", _DEFAULT_LIMIT) or _DEFAULT_LIMIT)
                results = self.backend.search(query, limit=limit)
                captured.extend(results)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": json.dumps(results)[:6000],
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        context = _context_from_results(captured)
        if answer_parts:
            context = context + "\n── orchestrator answer ──\n" + "\n".join(answer_parts)
        return Cat9aQueryOutcome(
            question_id="",
            tool_calls=tool_calls,
            context=context,
            answer="\n".join(answer_parts),
        )


# --- Ollama (OpenAI-compatible tool-calling) driver -------------------


@dataclass
class OllamaOrchestrator:
    """Drive a local ollama model through the OpenAI-compatible
    tool-calling loop, counting mempalace_search invocations.

    ollama serves ``/v1/chat/completions`` with ``tools=[...]`` and
    returns ``message.tool_calls`` when the model elects to invoke. We
    run the same agentic loop as the Bedrock driver so the invocation
    metric is measured identically. Models that don't support tool
    calling simply never emit tool_calls → invocation rate 0 (a real,
    interpretable reading, not an error).
    """

    model: str
    backend: DaemonSearch
    base_url: Optional[str] = None
    max_tokens: int = 1024
    _client: Any = None

    def __post_init__(self) -> None:
        from openai import OpenAI  # lazy — optional dep

        base_url = self.base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        )
        self._client = OpenAI(base_url=base_url, api_key="ollama")

    @property
    def _tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": _TOOL_NAME,
                "description": _TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }

    def __call__(self, question: str) -> Cat9aQueryOutcome:
        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You answer questions about JP, his projects, and his realm. "
                    "You have a mempalace_search tool that retrieves relevant "
                    "drawers. Use it to ground factual claims before answering."
                ),
            },
            {"role": "user", "content": question},
        ]
        tool_calls = 0
        captured: list[dict] = []
        answer_parts: list[str] = []

        for _ in range(_MAX_TOOL_ITERATIONS):
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[self._tool_schema],
                max_tokens=self.max_tokens,
            )
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None) or []
            if msg.content:
                answer_parts.append(msg.content)

            if not calls:
                break

            # Echo the assistant tool-call turn, then return results.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in calls
                    ],
                }
            )
            for c in calls:
                tool_calls += 1
                try:
                    args = json.loads(c.function.arguments or "{}")
                except (ValueError, TypeError):
                    args = {}
                query = args.get("query", question)
                limit = int(args.get("limit", _DEFAULT_LIMIT) or _DEFAULT_LIMIT)
                results = self.backend.search(query, limit=limit)
                captured.extend(results)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": c.id,
                        "content": json.dumps(results)[:6000],
                    }
                )

        context = _context_from_results(captured)
        if answer_parts:
            context = context + "\n── orchestrator answer ──\n" + "\n".join(answer_parts)
        return Cat9aQueryOutcome(
            question_id="",
            tool_calls=tool_calls,
            context=context,
            answer="\n".join(answer_parts),
        )


def make_orchestrator(model: str, backend: DaemonSearch, **kwargs) -> Any:
    """Pick a driver by model id. Claude/Bedrock ids -> BedrockOrchestrator;
    everything else (ollama tags like ``qwen3.5:4b``) -> OllamaOrchestrator.
    Mirrors the lane split in sme/eval/answer_generator.py."""
    if model.startswith(("claude", "anthropic.", "us.anthropic.")):
        return BedrockOrchestrator(model=model, backend=backend, **kwargs)
    return OllamaOrchestrator(model=model, backend=backend, **kwargs)
