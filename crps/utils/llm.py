"""LLM inference wrapper supporting vLLM, HuggingFace Transformers, and Anthropic API."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMInference:
    """Unified LLM inference interface for the CRPS framework.

    Supports three backends:
    * **vllm** – high-throughput inference via ``vllm.LLM``.
    * **hf** – HuggingFace ``transformers`` (AutoModelForCausalLM / AutoTokenizer).
    * **anthropic** – Anthropic-compatible API via the ``anthropic`` SDK.
      Supports custom base URLs for OpenAI-compatible endpoints.

    Args:
        model_name_or_path: HuggingFace model id / local path, or API model
            name (e.g. ``"gpt-5-mini"``).
        backend: ``"vllm"``, ``"hf"``, or ``"anthropic"``.
        temperature: Sampling temperature.
        max_tokens: Maximum number of new tokens to generate.
        tensor_parallel_size: GPU tensor-parallel degree (vLLM only).
        api_base_url: Base URL for the Anthropic-compatible API
            (defaults to ``$ANTHROPIC_BASE_URL``).
        api_key: API key (defaults to ``$ANTHROPIC_API_KEY`` /
            ``$ANTHROPIC_AUTH_TOKEN``).
    """

    _VALID_BACKENDS = {"vllm", "hf", "anthropic"}

    def __init__(
        self,
        model_name_or_path: str,
        backend: str = "vllm",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tensor_parallel_size: int = 1,
        api_base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"Unsupported backend {backend!r}. "
                f"Choose from {self._VALID_BACKENDS}."
            )
        self.model_name_or_path = model_name_or_path
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.api_base_url = api_base_url
        self.api_key = api_key

        # Lazy-initialised handles
        self._model: Any = None
        self._tokenizer: Any = None
        self._client: Any = None  # Anthropic client
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model & tokenizer on first use."""
        if self._loaded:
            return
        if self.backend == "vllm":
            self._load_vllm()
        elif self.backend == "hf":
            self._load_hf()
        elif self.backend == "anthropic":
            self._load_anthropic()
        self._loaded = True

    def _load_vllm(self) -> None:
        from vllm import LLM  # type: ignore[import-untyped]

        logger.info("Loading model with vLLM: %s", self.model_name_or_path)
        self._model = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=4096,
        )
        self._tokenizer = self._model.get_tokenizer()

    def _load_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

        logger.info(
            "Loading model with HuggingFace Transformers: %s",
            self.model_name_or_path,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def _load_anthropic(self) -> None:
        import os

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for the 'anthropic' backend. "
                "Install with: pip install anthropic"
            )

        base_url = (
            self.api_base_url
            or os.environ.get("ANTHROPIC_BASE_URL")
        )
        if not base_url:
            raise ValueError(
                "api_base_url must be provided or set via ANTHROPIC_BASE_URL "
                "environment variable for the 'anthropic' backend."
            )
        api_key = (
            self.api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or "no-key"
        )

        logger.info(
            "Connecting to Anthropic API: model=%s, base_url=%s",
            self.model_name_or_path,
            base_url,
        )
        self._client = anthropic.Anthropic(
            base_url=base_url,
            api_key=api_key,
        )

    # ------------------------------------------------------------------
    # Chat template helpers
    # ------------------------------------------------------------------

    def _apply_chat_template(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Apply the tokenizer's chat template, falling back to a simple concat."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback: plain concatenation
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant:")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, n: int = 1, max_tokens: int | None = None) -> list[str]:
        """Generate *n* completions for a single *prompt*."""
        self._ensure_loaded()

        if self.backend == "vllm":
            return self._generate_vllm([prompt], n=n, max_tokens=max_tokens)[0]
        elif self.backend == "anthropic":
            return self._generate_anthropic_plain(prompt, n=n)
        return self._generate_hf(prompt, n=n, max_tokens=max_tokens)

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int = 1,
    ) -> list[str]:
        """Generate completions using separate system and user messages.

        For the Anthropic backend, this uses native system/user message
        separation. For local backends, applies the chat template.
        """
        self._ensure_loaded()

        if self.backend == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt, n=n)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted = self._apply_chat_template(messages)
        return self.generate(formatted, n=n)

    def batch_generate(
        self,
        prompts: list[str],
        n: int = 1,
        max_tokens: int | None = None,
    ) -> list[list[str]]:
        """Batch generation across multiple prompts."""
        self._ensure_loaded()

        if self.backend == "vllm":
            return self._generate_vllm(prompts, n=n, max_tokens=max_tokens)

        return [self.generate(p, n=n, max_tokens=max_tokens) for p in prompts]

    # ------------------------------------------------------------------
    # Backend-specific generation
    # ------------------------------------------------------------------

    def _generate_vllm(
        self,
        prompts: list[str],
        n: int = 1,
        max_tokens: int | None = None,
    ) -> list[list[str]]:
        from vllm import SamplingParams  # type: ignore[import-untyped]

        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            n=n,
        )
        request_outputs = self._model.generate(prompts, params)

        results: list[list[str]] = []
        for output in request_outputs:
            results.append([o.text for o in output.outputs])
        return results

    def _generate_hf(self, prompt: str, n: int = 1, max_tokens: int | None = None) -> list[str]:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(
            self._model.device
        )
        results: list[str] = []
        for _ in range(n):
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                )
            # Decode only the newly generated tokens
            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            text = self._tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            results.append(text)
        return results

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int = 1,
    ) -> list[str]:
        """Generate using the Anthropic Messages API with system/user separation."""
        results: list[str] = []
        for _ in range(n):
            try:
                response = self._client.messages.create(
                    model=self.model_name_or_path,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = response.content[0].text
                results.append(text)
            except Exception as e:
                logger.error("Anthropic API error: %s", e)
                results.append("")
        return results

    def _generate_anthropic_plain(
        self,
        prompt: str,
        n: int = 1,
    ) -> list[str]:
        """Generate using the Anthropic Messages API with a plain prompt."""
        results: list[str] = []
        for _ in range(n):
            try:
                response = self._client.messages.create(
                    model=self.model_name_or_path,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
                results.append(text)
            except Exception as e:
                logger.error("Anthropic API error: %s", e)
                results.append("")
        return results
