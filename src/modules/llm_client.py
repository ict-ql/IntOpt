"""Batch LLM caller — sends prompt files to an OpenAI-compatible API in parallel."""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from openai import OpenAI

from modules.utils import log


class LLMClient:
    """Reads *.prompt.ll files, calls an LLM, writes *.model.predict.ll responses."""

    PROMPT_SUFFIX = ".prompt.ll"
    PRED_SUFFIX = ".model.predict.ll"

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self._tls = threading.local()

    # ------------------------------------------------------------------
    # OpenAI client (thread-local)
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        c = getattr(self._tls, "client", None)
        if c is None:
            self._tls.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            c = self._tls.client
        return c

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_prefix(prompt_path: Path) -> str:
        name = prompt_path.name
        if name.endswith(LLMClient.PROMPT_SUFFIX):
            return name[: -len(LLMClient.PROMPT_SUFFIX)]
        return prompt_path.stem

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _is_completed(path: Path, min_bytes: int) -> bool:
        try:
            return path.exists() and path.is_file() and path.stat().st_size >= min_bytes
        except OSError:
            return False

    @staticmethod
    def _is_not_found_404(exc: Exception) -> bool:
        tname = type(exc).__name__.lower()
        msg = str(exc).lower()
        if "notfound" in tname or "not found" in msg:
            return True
        if "404" in msg and ("page not found" in msg or "not found" in msg):
            return True
        return False

    @staticmethod
    def _extract_chat_text(resp) -> str:
        try:
            choices = getattr(resp, "choices", None)
            if not choices:
                return str(resp)
            return choices[0].message.content or ""
        except Exception:
            return str(resp)

    # ------------------------------------------------------------------
    # Core API call with retry
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        prompt_text: str,
        model: str,
        max_output_tokens: int,
        temperature: float = None,
        truncation: str = "auto",
        store: bool = False,
        api_mode: str = "auto",
        max_retries: int = 2,
        base_backoff: float = 1.0,
    ) -> str:
        client = self._get_client()

        def _via_responses() -> str:
            kwargs = dict(
                model=model,
                input=prompt_text,
                max_output_tokens=max_output_tokens,
                truncation=truncation,
                store=store,
            )
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.responses.create(**kwargs)
            if resp is None:
                raise RuntimeError("responses API returned None")
            out = getattr(resp, "output_text", None)
            result = str(resp) if out is None else out
            if not result or not result.strip():
                log(f"  DEBUG: responses API returned empty (output_text={out!r})")
                raise RuntimeError("responses API returned empty output")
            return result

        def _via_chat() -> str:
            try:
                kwargs = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt_text}],
                    store=store,
                )
                if temperature is not None:
                    kwargs["temperature"] = temperature
                resp = client.chat.completions.create(**kwargs)
                result = self._extract_chat_text(resp)
                return result
            except Exception as e:
                raise

        for attempt in range(max_retries + 1):
            try:
                if api_mode == "responses":
                    return _via_responses()
                if api_mode == "chat":
                    return _via_chat()
                # auto: try responses first, fall back to chat on 404 or empty
                try:
                    return _via_responses()
                except Exception as e:
                    if self._is_not_found_404(e) or "empty output" in str(e):
                        return _via_chat()
                    raise
            except Exception as e:
                msg = str(e).lower()
                retryable = any(
                    k in msg
                    for k in (
                        "rate limit", "429", "timeout", "timed out",
                        "temporarily", "temporary", "connection",
                        "reset", "econnreset",
                        "502", "503", "504", "server error", "internal error",
                    )
                )
                if (not retryable) or (attempt >= max_retries):
                    raise
                sleep_s = base_backoff * (2 ** attempt) + random.random() * 0.25
                time.sleep(sleep_s)

        raise RuntimeError("unreachable retry loop")

    # ------------------------------------------------------------------
    # Process a single prompt file
    # ------------------------------------------------------------------

    def _process_one(
        self,
        prompt_path: Path,
        out_dir: Path,
        model: str,
        max_output_tokens: int,
        temperature: float,
        truncation: str,
        store: bool,
        api_mode: str,
        min_output_bytes: int,
        overwrite: bool,
        max_retries: int,
        base_backoff: float,
    ) -> Tuple[Path, Optional[Path], Optional[str]]:
        prefix = self._split_prefix(prompt_path)
        out_path = out_dir / f"{prefix}{self.PRED_SUFFIX}"

        if (not overwrite) and self._is_completed(out_path, min_bytes=min_output_bytes):
            return prompt_path, out_path, None

        prompt_text = prompt_path.read_text(encoding="utf-8", errors="ignore")
        try:
            answer = self._call_with_retry(
                prompt_text=prompt_text,
                model=model,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                truncation=truncation,
                store=store,
                api_mode=api_mode,
                max_retries=max_retries,
                base_backoff=base_backoff,
            )
            if not answer or not answer.strip():
                log(f"  WARN: empty response for {prompt_path.name} "
                    f"(prompt={len(prompt_text)} chars, model={model})")
                return prompt_path, None, "LLM returned empty response"
            self._atomic_write(out_path, answer if answer.endswith("\n") else (answer + "\n"))
            return prompt_path, out_path, None
        except Exception as e:
            return prompt_path, None, str(e)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def batch_query(
        self,
        in_dir: str,
        out_dir: str = "",
        model: str = "gpt-5",
        api_mode: str = "auto",
        workers: int = 50,
        max_output_tokens: int = 8192,
        temperature: float = 1.0,
        truncation: str = "auto",
        store: bool = False,
        overwrite: bool = False,
        min_output_bytes: int = 16,
        max_retries: int = 2,
        base_backoff: float = 0.8,
    ) -> str:
        """Send every *.prompt.ll under *in_dir* to the LLM; write responses
        as *.model.predict.ll under *out_dir*.  Returns the output directory."""

        in_dir = Path(in_dir)
        if not in_dir.is_dir():
            raise SystemExit(f"in_dir is not a directory: {in_dir}")

        out_dir = Path(out_dir) if out_dir else in_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        prompt_files = sorted(in_dir.glob("*.prompt.ll"))
        if not prompt_files:
            raise SystemExit(f"No *.prompt.ll files found under {in_dir}")

        log(f"Found prompts: {len(prompt_files)}")
        log(f"Input : {in_dir}")
        log(f"Output: {out_dir}")
        log(f"Model : {model}  |  API: {api_mode}  |  Workers: {workers}")

        ok = skipped = failed = 0

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = []
            for p in prompt_files:
                prefix = self._split_prefix(p)
                out_path = out_dir / f"{prefix}{self.PRED_SUFFIX}"
                if (not overwrite) and self._is_completed(out_path, min_bytes=min_output_bytes):
                    skipped += 1
                    continue
                futs.append(
                    ex.submit(
                        self._process_one, p, out_dir, model,
                        max_output_tokens, temperature, truncation,
                        store, api_mode, min_output_bytes, overwrite,
                        max_retries, base_backoff,
                    )
                )

            total_run = len(futs)
            log(f"To run: {total_run}  |  Skipped existing: {skipped}")

            for idx, fut in enumerate(as_completed(futs), start=1):
                prompt_path, out_path, err = fut.result()
                if err is None and out_path is not None:
                    ok += 1
                    if idx % 20 == 0 or idx == total_run:
                        log(f"[{idx}/{total_run}] OK: {out_path.name}")
                else:
                    failed += 1
                    log(f"[{idx}/{total_run}] ERROR: {prompt_path.name} -> {err}")

        log(f"Done. ok={ok}  skipped={skipped}  failed={failed}")
        return str(out_dir)
