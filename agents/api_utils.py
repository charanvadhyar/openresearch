"""
Unified LLM client — Anthropic Claude or OpenAI, same interface.

Supports a fallback chain: try providers in order, falling back to the next
one when a rate limit is hit.

Single provider:
    llm = LLMClient.single("anthropic", api_key="sk-ant-...", model="claude-sonnet-4-20250514")
    llm = LLMClient.single("openai",    api_key="sk-...",     model="gpt-4o")

Fallback chain (Anthropic → OpenAI):
    llm = LLMClient([
        ("anthropic", "sk-ant-...", "claude-sonnet-4-20250514"),
        ("openai",    "sk-...",     "gpt-4o"),
    ])

    text = llm.create(system="...", messages=[...], max_tokens=500)

Rate limits:
    On 429 within a provider: wait Retry-After (or 60 s) and retry up to
    MAX_RETRIES times, then fall through to the next provider in the chain.
    If all providers are exhausted, re-raises the last error.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RETRIES   = 4   # retries per provider before falling back to next


# ── Token budget ───────────────────────────────────────────────────────────────

# Conservative context window limits per model family (input tokens).
# We use ~80% of the real limit to leave headroom for system prompt overhead.
_CONTEXT_LIMITS: dict[str, int] = {
    # OpenAI
    "gpt-5":        100_000,
    "gpt-4.1":      800_000,   # 1M context, use 80%
    "gpt-4o":       100_000,
    "gpt-4":         24_000,
    "o1":           160_000,
    "o3":           160_000,
    "o4":           160_000,
    # Anthropic
    "claude-opus":  160_000,
    "claude-sonnet":160_000,
    "claude-haiku":  80_000,
    # Default fallback
    "_default":      24_000,
}

def _context_limit_for(model: str) -> int:
    """Return the safe input token budget for a given model name."""
    lower = model.lower()
    for prefix, limit in _CONTEXT_LIMITS.items():
        if lower.startswith(prefix):
            return limit
    return _CONTEXT_LIMITS["_default"]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 3.5 characters (conservative)."""
    return max(1, int(len(text) / 3.5))


class PromptBudget:
    """
    Manages token allocation for a single LLM call.

    Usage:
        budget = PromptBudget(model="gpt-5", reserved_output=4000)
        budget.reserve("system", system_prompt)
        sections = budget.fit([
            ("error",    error_text,    True),   # (name, text, required)
            ("code",     nb_code,       True),
            ("memory",   memory_hint,   False),  # optional — dropped first
        ])
        # sections is a dict name→fitted_text; dropped optionals are empty string

    The budget is consumed in priority order (required first, then optional).
    Each section is truncated if needed; optional sections are dropped entirely
    before any required section is truncated.
    """

    def __init__(self, model: str, reserved_output: int = 4000):
        self._limit   = _context_limit_for(model)
        self._used    = reserved_output   # reserve space for output
        self._model   = model

    def reserve(self, name: str, text: str) -> None:
        """Pre-consume tokens for a fixed section (system prompt, instructions)."""
        self._used += _estimate_tokens(text)

    def fit(self, sections: list[tuple[str, str, bool]]) -> dict[str, str]:
        """
        Fit sections into remaining budget.

        sections: list of (name, text, required)
        Returns dict of name → fitted text (truncated or empty).
        """
        remaining = max(0, self._limit - self._used)
        result: dict[str, str] = {name: "" for name, _, _ in sections}

        # Pass 1: required sections — truncate to fit
        for name, text, required in sections:
            if not required:
                continue
            tokens = _estimate_tokens(text)
            if tokens <= remaining:
                result[name] = text
                remaining -= tokens
            else:
                # Truncate to remaining budget (chars ≈ tokens * 3.5)
                chars = int(remaining * 3.5)
                result[name] = text[:chars] + "\n... [truncated to fit token limit]"
                remaining = 0

        # Pass 2: optional sections — include only if budget allows
        for name, text, required in sections:
            if required:
                continue
            tokens = _estimate_tokens(text)
            if tokens <= remaining:
                result[name] = text
                remaining -= tokens
            # else: leave as empty string — dropped

        self._used += (_limit_before := self._limit - remaining)  # noqa: F841
        return result


# ── Model discovery ────────────────────────────────────────────────────────────

def list_provider_models(provider: str, api_key: str) -> list[str]:
    """
    Fetch all available model IDs from the given provider.

    Returns model IDs sorted by the provider (Anthropic: newest first;
    OpenAI: chat/reasoning models only, newest first).

    Args:
        provider: "anthropic" or "openai"
        api_key:  API key for that provider

    Returns:
        List of model ID strings, e.g. ["claude-opus-4-20250514", ...]
    """
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        page = client.models.list(limit=100)
        return [m.id for m in page.data]

    elif provider == "openai":
        import openai
        client = openai.OpenAI(api_key=api_key)
        all_models = client.models.list()
        # Exclude known non-chat model families; keep everything else
        # (new model families like gpt-5, o5, etc. are included automatically)
        excluded_prefixes = (
            "dall-e", "whisper", "tts", "text-embedding",
            "text-moderation", "babbage", "davinci", "ada",
            "curie", "code-", "audio-",
        )
        ids = sorted(
            [
                m.id for m in all_models.data
                if not m.id.startswith(excluded_prefixes)
            ],
            reverse=True,
        )
        return ids

    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'openai'.")


def resolve_models(
    provider: str,
    api_key: str,
    requested: str,
    verbose: bool = True,
) -> list[str]:
    """
    Resolve a model specification for one provider.

    - If ``requested`` is ``"auto"``, call the provider API and return all
      available models (ordered best-first).
    - Otherwise return ``[requested]`` unchanged.

    Args:
        provider:  "anthropic" or "openai"
        api_key:   API key
        requested: model name or "auto"
        verbose:   print discovered models

    Returns:
        List of model ID strings to use as a fallback sub-chain.
    """
    if requested.lower() != "auto":
        return [requested]

    logger.info(f"[{provider}] llm_model=auto — fetching available models...")
    models = list_provider_models(provider, api_key)

    if not models:
        raise RuntimeError(
            f"Provider '{provider}' returned no models. "
            "Check your API key and network connection."
        )

    if verbose:
        print(f"\n  🔍 [{provider}] Discovered {len(models)} model(s):")
        for m in models[:10]:          # show first 10 to avoid spam
            print(f"       • {m}")
        if len(models) > 10:
            print(f"       … and {len(models) - 10} more")

    return models
BASE_WAIT_SEC = 60  # default wait on rate limit (API may return a shorter value)


# ── Single-provider backend ────────────────────────────────────────────────────

class _Backend:
    """Wraps one (provider, model) pair and makes API calls."""

    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider
        self.model    = model

        if provider == "anthropic":
            import anthropic
            self._client         = anthropic.Anthropic(api_key=api_key)
            self._rate_limit_exc = anthropic.RateLimitError
        elif provider == "openai":
            import openai
            self._client         = openai.OpenAI(api_key=api_key)
            self._rate_limit_exc = openai.RateLimitError
        else:
            raise ValueError(f"Unknown LLM provider '{provider}'. Use 'anthropic' or 'openai'.")

    def call(self, system: str, messages: list[dict], max_tokens: int,
             verbose: bool = False) -> str:
        """
        Make a single API call with exponential back-off on rate limit.
        Raises the RateLimitError after MAX_RETRIES so the caller can try
        the next backend.
        """
        wait = BASE_WAIT_SEC
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self.provider == "anthropic":
                    return self._call_anthropic(system, messages, max_tokens)
                else:
                    return self._call_openai(system, messages, max_tokens)

            except Exception as e:
                if not isinstance(e, self._rate_limit_exc):
                    raise           # non-rate-limit errors propagate immediately
                if attempt == MAX_RETRIES:
                    raise           # exhausted retries — caller handles fallback

                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("retry-after")
                sleep_sec = int(retry_after) if retry_after else wait

                if verbose:
                    print(f"\n  ⏳ [{self.provider}/{self.model}] Rate limit — "
                          f"waiting {sleep_sec}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(sleep_sec)
                wait = min(wait * 2, 300)

    def _call_anthropic(self, system: str, messages: list[dict], max_tokens: int) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text

    # Models that require 'developer' role instead of 'system'
    _DEVELOPER_ROLE_MODELS = ("o1", "o3", "o4", "gpt-5")

    def _uses_developer_role(self) -> bool:
        return any(self.model.startswith(p) for p in self._DEVELOPER_ROLE_MODELS)

    def _call_openai(self, system: str, messages: list[dict], max_tokens: int) -> str:
        system_role = "developer" if self._uses_developer_role() else "system"
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": system_role, "content": system}] + messages,
        )
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            err = str(e)
            # Some models require max_completion_tokens instead of max_tokens
            if "max_tokens" in err and "max_completion_tokens" in err:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                resp = self._client.chat.completions.create(**kwargs)
            # Some models reject 'system' role — retry with 'developer'
            elif "system" in err and system_role == "system":
                kwargs["messages"][0]["role"] = "developer"
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        content = resp.choices[0].message.content
        # Some models (o-series, gpt-5) may return None content when the response
        # is carried entirely in reasoning tokens or was filtered; surface a clear error.
        if not content:
            finish = resp.choices[0].finish_reason
            raise ValueError(
                f"Model '{self.model}' returned an empty response (finish_reason={finish!r}). "
                "This can happen when the model's output was filtered or the prompt was too long."
            )
        return content


# ── Public LLMClient ───────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified LLM client that supports a prioritised list of backends.

    Tries backends in order; falls back to the next one if the current one
    hits a rate limit and exhausts its retries.

    Usage:
        # Single provider
        llm = LLMClient.single("anthropic", "sk-ant-...", "claude-sonnet-4-20250514")

        # Fallback chain — Anthropic first, OpenAI as backup
        llm = LLMClient([
            ("anthropic", "sk-ant-...", "claude-sonnet-4-20250514"),
            ("openai",    "sk-...",     "gpt-4o"),
        ])
    """

    def __init__(self, backends: list[tuple[str, str, str]]):
        """
        Args:
            backends: List of (provider, api_key, model) tuples tried in order.
        """
        if not backends:
            raise ValueError("At least one backend must be provided.")
        self._backends = [_Backend(p, k, m) for p, k, m in backends]
        # Expose primary model name for logging / display
        self.provider = self._backends[0].provider
        self.model    = self._backends[0].model

    @classmethod
    def single(cls, provider: str, api_key: str, model: str) -> "LLMClient":
        """Convenience constructor for a single provider."""
        return cls([(provider, api_key, model)])

    def create(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
        verbose: bool = False,
    ) -> str:
        """
        Call the first available backend. Falls back to the next backend if
        a rate limit is hit after all retries.

        Returns the response text as a plain string.
        """
        last_error: Exception | None = None

        for backend in self._backends:
            try:
                return backend.call(system, messages, max_tokens, verbose)
            except Exception as e:
                # Check if it's a rate-limit error — if so, try next backend
                if isinstance(e, backend._rate_limit_exc):
                    last_error = e
                    if verbose:
                        print(f"\n  ⚠️  [{backend.provider}/{backend.model}] rate limit "
                              f"exhausted — trying next provider...")
                    continue
                raise   # any other error propagates immediately

        # All backends exhausted
        raise last_error


# ── Legacy shim ────────────────────────────────────────────────────────────────

def claude_create(client, verbose: bool = False, **kwargs):
    """Deprecated: direct Anthropic client wrapper. Use LLMClient instead."""
    import anthropic
    wait = BASE_WAIT_SEC
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after = e.response.headers.get("retry-after")
            sleep_sec = int(retry_after) if retry_after else wait
            if verbose:
                print(f"\n  ⏳ Rate limit hit — waiting {sleep_sec}s before retry "
                      f"(attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(sleep_sec)
            wait = min(wait * 2, 300)
