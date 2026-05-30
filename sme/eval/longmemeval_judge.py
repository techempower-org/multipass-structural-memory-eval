"""LongMemEval judge wrapper — canonical type-specific grading prompts.

Ports the question-type-specific grading prompts from LongMemEval upstream
(``src/evaluation/evaluate_qa.py`` in xiaowu0162/LongMemEval, MIT) in a form
that's callable from SME's cross-validation / reader-sweep harness without
spawning an upstream subprocess.

**Why this exists (the judge-prompt-mismatch fix).** SME's earlier judge used
a *single generic JSON-label rubric* paraphrased from the paper. That collapsed
whole question categories — most visibly ``single-session-preference`` scored
0.0 against ``single-session-user`` 0.84 in the same Pass A run — which is the
signature of a prompt mismatch, not a reader deficiency, and it drove an
implausible 35pp oracle deficit vs the published 87.0% oracle ceiling. The fix
is to grade *each question type with its own canonical rubric*, exactly as
upstream does:

- ``single-session-user`` / ``single-session-assistant`` / ``multi-session``
  share the base correctness template.
- ``temporal-reasoning`` adds the off-by-one tolerance clause.
- ``knowledge-update`` accepts a response that carries the prior value as long
  as the updated answer is present.
- ``single-session-preference`` is **rubric-based** — the gold "answer" is a
  *rubric* for a desired personalized response, and the response passes as long
  as it recalls and uses the user's personal information correctly (it does NOT
  need to satisfy every rubric point). This is the template the old judge was
  missing.
- abstention (any ``_abs`` question) uses the unanswerable-detection template —
  the gold "answer" is an *explanation* of why the question is unanswerable, and
  the response passes if it correctly identifies the question as unanswerable.

**Canonical decision protocol.** Upstream sends a single ``user`` message, no
system prompt, ``temperature=0``, ``max_tokens=10``, and decides
``label = 'yes' in reply.lower()`` — a binary correct/incorrect verdict. There
is no PARTIAL class. We reproduce that protocol verbatim and map it onto SME's
``autoeval_label`` contract so the existing aggregator (which scores ``CORRECT``
and ``ABSTAIN``-on-abstention as right) keeps working unchanged:

- non-abstention: yes -> ``CORRECT``, no -> ``INCORRECT``
- abstention:     yes -> ``ABSTAIN`` (correctly refused), no -> ``INCORRECT``

**Model disclosure.** ``DEFAULT_JUDGE_MODEL`` is the canonical
``gpt-4o-2024-08-06`` snapshot, but in this deployment we do NOT have access to
that model — the runs in this repo use ``gpt-5.3-chat`` as the judge with these
canonical type-specific prompts. The fix is the PROMPTS (which collapse the
category bias), not the model. Any published number must disclose
"judge = gpt-5.3-chat + canonical LongMemEval type-specific prompts", NOT the
exact ``gpt-4o-2024-08-06`` snapshot. See ``docs/related_work/longmemeval.md``.

Public surface:

- ``grade_answer(question_type, ...)`` — single judge call; returns a dict with
  ``autoeval_label`` in {CORRECT, INCORRECT, ABSTAIN, ERROR}.
- ``grade_answer_replicated(question_type, ..., replicates=K)`` — K-replicate
  judge calls with majority-vote aggregation, used to characterize intra-rater
  stochasticity (see upstream issue #22).

Design notes:

- The wrapper uses the ``openai`` SDK (or the Bedrock / ollama shims) via the
  same routing the reader uses; tests mock the call entirely, so no path needs
  to be reachable in CI.
- All failure modes return ``autoeval_label='ERROR'`` rather than raising, so the
  harness can keep running across a 500-question batch even when individual judge
  calls misbehave. ERROR is SME-internal — a call failure, distinct from an
  upstream "no" verdict (which is INCORRECT).
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from typing import Any, Optional

log = logging.getLogger(__name__)

# Canonical judge model per LongMemEval paper §4 / upstream evaluate_qa.py
# (model_zoo['gpt-4o'] -> 'gpt-4o-2024-08-06'). NOTE: this deployment does not
# have that snapshot; runs use gpt-5.3-chat with the canonical prompts below.
# This constant documents the canonical default; the harness passes the actual
# judge_model explicitly.
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"

# Question types the judge handles. Mirrors LME_QUESTION_TYPES from the loader
# plus the `abstention` pseudo-type used for `_abs` records.
JUDGE_QUESTION_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
    "abstention",
}

# Question types that share the base correctness template upstream.
_BASE_TEMPLATE_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "multi-session",
}

# Labels grade_answer can emit. The canonical judge is binary (yes/no); we map
# that onto CORRECT/INCORRECT for fact questions and ABSTAIN/INCORRECT for
# abstention questions. ERROR is SME-internal (the call itself failed).
VALID_LABELS = {"CORRECT", "INCORRECT", "ABSTAIN", "ERROR"}

# --- Canonical prompt templates (verbatim from xiaowu0162/LongMemEval,
# src/evaluation/evaluate_qa.py::get_anscheck_prompt, MIT). The three `{}`
# slots are (question, correct-answer-or-rubric-or-explanation, model-response),
# filled in that order, exactly as upstream's .format(question, answer, response).

_TEMPLATE_BASE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
    "Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPLATE_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for "
    "the number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\n"
    "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? "
    "Answer yes or no only."
)

_TEMPLATE_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}"
    "\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPLATE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and "
    "a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
    "Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer "
    "yes or no only."
)

_TEMPLATE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from "
    "a model. Please answer yes if the model correctly identifies the question as "
    "unanswerable. The model could say that the information is incomplete, or some "
    "other information is given but the asked information is not.\n\nQuestion: {}"
    "\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify "
    "the question as unanswerable? Answer yes or no only."
)


def _template_for_question_type(qtype: str, *, is_abstention: bool) -> str:
    """Select the canonical grading template for a (question_type, abstention).

    Mirrors upstream ``get_anscheck_prompt``: the ``abstention`` flag overrides
    all type-specific templates. Types without a specialized template fall back
    to the base correctness template (upstream raises NotImplementedError there;
    we degrade gracefully so an unexpected type still gets graded).
    """
    if is_abstention or qtype == "abstention":
        return _TEMPLATE_ABSTENTION
    if qtype == "temporal-reasoning":
        return _TEMPLATE_TEMPORAL
    if qtype == "knowledge-update":
        return _TEMPLATE_KNOWLEDGE_UPDATE
    if qtype == "single-session-preference":
        return _TEMPLATE_PREFERENCE
    # single-session-user / single-session-assistant / multi-session and any
    # unrecognized type all use the base correctness template.
    return _TEMPLATE_BASE


def _build_prompt(qtype: str, question: str, gold: str, hyp: str,
                  *, is_abstention: bool) -> str:
    """Fill the canonical template's three slots in upstream slot order:
    (question, correct-answer/rubric/explanation, model-response)."""
    template = _template_for_question_type(qtype, is_abstention=is_abstention)
    return template.format(question, gold, hyp)


def _parse_judge_reply(content: str) -> tuple[bool, str]:
    """Canonical binary verdict: ``'yes' in reply.lower()``.

    Returns ``(is_yes, rationale)`` where ``rationale`` is the raw judge text
    (trimmed) for traceability. Upstream strips the reply and checks for the
    substring "yes"; we reproduce that exactly. An empty reply is a "no"
    (caller already treats a failed *call* as ERROR upstream of this).
    """
    text = (content or "").strip()
    is_yes = "yes" in text.lower()
    return is_yes, text or "(empty judge reply)"


_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_PREFIXES)


def _call_openai(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_retries: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 10,
) -> dict:
    """Call the OpenAI Chat Completions endpoint with simple backoff.

    Mirrors upstream's call shape: single ``user`` message, ``temperature=0``,
    ``max_tokens=10`` (the judge answers "yes" or "no" only). Reasoning models
    (o-series, gpt-5) reject ``temperature`` and need a wider completion budget
    for hidden reasoning tokens, so we drop ``temperature`` and skip the tight
    ``max_tokens`` cap for them.

    Returns ``{'content': str, 'usage': dict}`` on success, raises the final
    exception on exhaustion. The caller is responsible for catching.
    """
    last_exc: Optional[BaseException] = None
    delay = 1.0
    reasoning = _is_reasoning_model(model)
    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            if not reasoning:
                # Canonical settings for non-reasoning judges (e.g. gpt-4o).
                kwargs["temperature"] = temperature
                kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            content = getattr(choice.message, "content", "") or ""
            usage_obj = getattr(resp, "usage", None)
            if usage_obj is None:
                usage = {"prompt_tokens": 0, "completion_tokens": 0,
                         "total_tokens": 0}
            else:
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(
                        usage_obj, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
                }
            return {"content": content, "usage": usage}
        except TypeError:
            # A mock/client whose .create() doesn't accept max_tokens (e.g. the
            # test fakes that only take model/messages/temperature). Retry once
            # without the canonical max_tokens cap so the call still goes
            # through — fidelity of the *prompt* is what matters in those paths.
            try:
                kwargs.pop("max_tokens", None)
                resp = client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                content = getattr(choice.message, "content", "") or ""
                usage_obj = getattr(resp, "usage", None)
                if usage_obj is None:
                    usage = {"prompt_tokens": 0, "completion_tokens": 0,
                             "total_tokens": 0}
                else:
                    usage = {
                        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(
                            usage_obj, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
                    }
                return {"content": content, "usage": usage}
            except Exception as e:  # noqa: BLE001 — judge errors are diagnostic
                last_exc = e
                log.warning(
                    "longmemeval_judge: attempt %d/%d failed (no-max_tokens "
                    "retry): %s", attempt + 1, max_retries, e,
                )
                if attempt + 1 < max_retries:
                    time.sleep(delay)
                    delay *= 2
        except Exception as e:  # noqa: BLE001 — judge errors are diagnostic
            last_exc = e
            log.warning(
                "longmemeval_judge: attempt %d/%d failed: %s",
                attempt + 1, max_retries, e,
            )
            if attempt + 1 < max_retries:
                time.sleep(delay)
                delay *= 2
    assert last_exc is not None
    raise last_exc


def _default_client() -> Optional[Any]:
    """Return a lazily-imported OpenAI or AzureOpenAI client, or None.

    Prefers Azure when ``AZURE_API_KEY`` and ``AZURE_API_BASE`` are both set;
    falls back to ``OPENAI_API_KEY`` -> vanilla ``OpenAI()``. Returns ``None`` if
    neither path is configured or the SDK isn't installed — the caller surfaces
    this as an ERROR-labelled verdict.
    """
    azure_key = os.environ.get("AZURE_API_KEY")
    azure_base = os.environ.get("AZURE_API_BASE")
    if azure_key and azure_base:
        try:
            from openai import AzureOpenAI  # type: ignore[import-not-found]
        except ImportError:
            log.info("longmemeval_judge: openai SDK not installed")
            return None
        api_version = os.environ.get(
            "AZURE_API_VERSION", "2024-12-01-preview"
        )
        return AzureOpenAI(
            azure_endpoint=azure_base,
            api_key=azure_key,
            api_version=api_version,
        )
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        log.info("longmemeval_judge: openai SDK not installed")
        return None
    return OpenAI()


def _client_for_judge(judge_model: str) -> Optional[Any]:
    """Route the judge client by model id, reusing the reader-side routing:
      - claude-* judges        -> AnthropicBedrock shim (``--judge claude-opus-4-8``
        tests whether a stronger judge re-scores more fairly, #116; the shim
        already retries without ``temperature`` for opus-4-8).
      - local/ollama judges    -> OpenAI client at localhost:11434 (no-cost,
        no-rate-limit judge lane on the katana GPU).
      - everything else        -> Azure/OpenAI default.
    Claude is checked first so Bedrock ids (no ':') never fall into the local
    lane's ':'-tag heuristic."""
    from sme.eval.answer_generator import (
        _bedrock_client, _is_claude_model, _is_local_model, _ollama_client,
    )
    if _is_claude_model(judge_model):
        return _bedrock_client()
    if _is_local_model(judge_model):
        return _ollama_client()
    return _default_client()


def grade_answer(
    question_type: str,
    question: str,
    gold_answer: str,
    hypothesis: str,
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    client: Optional[Any] = None,
    temperature: float = 0.0,
    is_abstention: Optional[bool] = None,
) -> dict:
    """Grade a system answer against the gold answer using the canonical
    LongMemEval type-specific prompt.

    Args:
        question_type: One of JUDGE_QUESTION_TYPES, or 'abstention'. Any other
            value degrades to the base correctness template.
        question: The natural-language question text.
        gold_answer: The reference answer string. For preference questions this
            is treated as a *rubric*; for abstention questions, an *explanation*
            of why the question is unanswerable.
        hypothesis: The system's generated answer.
        judge_model: model id to use. Defaults to the canonical
            ``gpt-4o-2024-08-06``; the harness overrides this with the actually
            available judge (gpt-5.3-chat).
        client: An OpenAI-SDK-shaped client (must have
            ``client.chat.completions.create(model, messages, ...)``). When None,
            one is constructed via ``_client_for_judge``. Tests pass a fake.
        temperature: Sampling temperature for the judge call. Defaults to 0.0
            (deterministic, the LongMemEval paper setting).
        is_abstention: When True, force the abstention (unanswerable-detection)
            template regardless of ``question_type``. When None, abstention is
            inferred from ``question_type == 'abstention'``. The reader-sweep
            harness passes ``question_type='abstention'`` for ``_abs`` records,
            so the default inference is correct there.

    Returns:
        {
          'autoeval_label': 'CORRECT' | 'INCORRECT' | 'ABSTAIN' | 'ERROR',
          'judge_model': str,
          'rationale': str,     # raw judge reply text
          'usage': {prompt_tokens, completion_tokens, total_tokens},
        }
    """
    base_result = {
        "autoeval_label": "ERROR",
        "judge_model": judge_model,
        "rationale": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    abstain = is_abstention if is_abstention is not None \
        else (question_type == "abstention")

    if question_type not in JUDGE_QUESTION_TYPES and not abstain:
        # Unknown type — fall through with the base template, but flag it.
        log.info("longmemeval_judge: unknown question_type %r", question_type)

    if client is None:
        client = _client_for_judge(judge_model)
    if client is None:
        base_result["rationale"] = "OPENAI_API_KEY not set; judge skipped"
        return base_result

    prompt = _build_prompt(
        question_type, question, gold_answer, hypothesis,
        is_abstention=abstain,
    )
    # ollama wants the bare tag, so strip any ``ollama/`` prefix before the call.
    from sme.eval.answer_generator import _is_local_model, _ollama_model_id
    wire_model = _ollama_model_id(judge_model) if _is_local_model(judge_model) \
        else judge_model
    try:
        called = _call_openai(
            client=client, model=wire_model, prompt=prompt,
            temperature=temperature,
        )
    except Exception as e:  # noqa: BLE001 — judge errors are diagnostic
        base_result["rationale"] = f"judge call failed after retries: {e}"
        return base_result

    is_yes, rationale = _parse_judge_reply(called["content"])
    # Map the binary verdict onto SME's label contract. For abstention
    # questions a correct "yes" (the model identified the question as
    # unanswerable) is scored as ABSTAIN, which the aggregator counts as right;
    # everything else is CORRECT (yes) / INCORRECT (no).
    if abstain:
        label = "ABSTAIN" if is_yes else "INCORRECT"
    else:
        label = "CORRECT" if is_yes else "INCORRECT"
    return {
        "autoeval_label": label,
        "judge_model": judge_model,
        "rationale": rationale,
        "usage": called["usage"],
    }


def grade_answer_replicated(
    question_type: str,
    question: str,
    gold_answer: str,
    hypothesis: str,
    *,
    replicates: int = 1,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    client: Optional[Any] = None,
    temperature: Optional[float] = None,
    is_abstention: Optional[bool] = None,
) -> dict:
    """Grade with K replicates to characterize judge variance.

    When ``replicates <= 1``, delegates directly to :func:`grade_answer` with
    ``temperature=0.0`` by default (backward compatible — single deterministic
    call matches the LongMemEval paper setting).

    When ``replicates > 1``, runs K independent judge calls at ``temperature=0.3``
    by default (override via the ``temperature`` argument) and aggregates via
    majority vote, excluding ``ERROR`` replicates from the vote.

    Args:
        question_type: One of JUDGE_QUESTION_TYPES or 'abstention'.
        question: The natural-language question text.
        gold_answer: The reference answer (rubric for preference, explanation for
            abstention).
        hypothesis: The system's generated answer.
        replicates: Number of independent judge calls (K).
        judge_model: model id to use.
        client: An OpenAI-SDK-shaped client (see :func:`grade_answer`).
        temperature: Sampling temperature override. Defaults to 0.0 for K=1 and
            0.3 for K>1.
        is_abstention: Forwarded to :func:`grade_answer`.

    Returns:
        For K=1, exactly what :func:`grade_answer` returns.

        For K>1, the single-call shape plus replicate diagnostics:
          {
            'autoeval_label': str,        # majority label
            'judge_model': str,
            'rationale': str,             # rationale from the majority
            'usage': {...},               # summed across all replicates
            'replicates': list[dict],     # individual replicate results
            'label_counts': dict,         # label -> count (non-ERROR)
            'agreement_rate': float,      # fraction matching majority
            'flip_rate': float,           # 1 - agreement_rate
          }

        When every replicate returns ``ERROR``, the first replicate is returned
        with ``replicates`` attached so the caller can inspect the failures.
    """
    if replicates <= 1:
        temp = temperature if temperature is not None else 0.0
        return grade_answer(
            question_type, question, gold_answer, hypothesis,
            judge_model=judge_model, client=client, temperature=temp,
            is_abstention=is_abstention,
        )

    temp = temperature if temperature is not None else 0.3
    results: list[dict] = []
    for _ in range(replicates):
        r = grade_answer(
            question_type, question, gold_answer, hypothesis,
            judge_model=judge_model, client=client, temperature=temp,
            is_abstention=is_abstention,
        )
        results.append(r)

    # Sum usage across all replicates regardless of outcome.
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in results:
        u = r.get("usage", {})
        for key in total_usage:
            total_usage[key] += u.get(key, 0)

    # Majority vote excludes ERROR replicates — those are call failures, not
    # verdicts.
    labels = [r["autoeval_label"] for r in results
              if r["autoeval_label"] != "ERROR"]
    if not labels:
        # All replicates errored — surface the first result, but attach the full
        # replicate trace and the summed usage so cost accounting still reflects
        # K calls.
        first = dict(results[0])
        first["usage"] = total_usage
        first["replicates"] = results
        return first

    counter = Counter(labels)
    label_counts = dict(counter.most_common())
    majority_label = counter.most_common(1)[0][0]
    agreement_count = label_counts[majority_label]
    agreement_rate = agreement_count / len(labels)

    # Use the rationale from the first replicate that voted with the majority —
    # keeps the output explainable.
    majority_result = next(
        r for r in results if r["autoeval_label"] == majority_label
    )

    return {
        "autoeval_label": majority_label,
        "judge_model": judge_model,
        "rationale": majority_result["rationale"],
        "usage": total_usage,
        "replicates": results,
        "label_counts": label_counts,
        "agreement_rate": agreement_rate,
        "flip_rate": 1.0 - agreement_rate,
    }
