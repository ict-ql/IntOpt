import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from openai import OpenAI

class LLM_Optimizer:
    def __init__(self, base_url: str, api_key: str):
        self.BASE_URL = base_url
        self.API_KEY = api_key
        self._tls = threading.local()
        self.PROMPT_SUFFIX = ".prompt.ll"
        self.PRED_SUFFIX = ".model.predict.ll"
    
    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)
    
    def get_client(self) -> OpenAI:
        c = getattr(self._tls, "client", None)
        if c is None:
            self._tls.client = OpenAI(base_url=self.BASE_URL, api_key=self.API_KEY)
            c = self._tls.client
        return c
    
    def split_prefix(self, prompt_path: Path) -> str:
        name = prompt_path.name
        if name.endswith(self.PROMPT_SUFFIX):
            return name[: -len(self.PROMPT_SUFFIX)]
        return prompt_path.stem
    
    def atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    
    def is_completed_output(self, path: Path, min_bytes: int) -> bool:
        try:
            return path.exists() and path.is_file() and path.stat().st_size >= min_bytes
        except OSError:
            return False
    
    def _is_not_found_404(self, exc: Exception) -> bool:
        tname = type(exc).__name__.lower()
        msg = str(exc).lower()
        if "notfound" in tname or "not found" in msg:
            return True
        if "404" in msg and ("page not found" in msg or "not found" in msg):
            return True
        return False
    
    def _extract_chat_text(self, resp) -> str:
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return str(resp)
    
    def call_openai_with_retry(
        self,
        prompt_text: str,
        model: str,
        max_output_tokens: int,
        temperature: float,
        truncation: str,
        store: bool,
        api_mode: str,
        max_retries: int,
        base_backoff: float,
    ) -> str:
        client = self.get_client()

        def call_responses() -> str:
            resp = client.responses.create(
                model=model,
                input=prompt_text,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                truncation=truncation,
                store=store,
            )
            out = getattr(resp, "output_text", None)
            return str(resp) if out is None else out

        def call_chat() -> str:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=temperature,
                    max_completion_tokens=max_output_tokens,
                    store=store,
                )
                return self._extract_chat_text(resp)
            except Exception as e:
                msg = str(e).lower()
                if "max_completion_tokens" in msg:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=temperature,
                        max_tokens=max_output_tokens,
                        store=store,
                    )
                    return self._extract_chat_text(resp)
                raise

        for attempt in range(max_retries + 1):
            try:
                if api_mode == "responses":
                    return call_responses()
                if api_mode == "chat":
                    return call_chat()

                try:
                    return call_responses()
                except Exception as e:
                    if self._is_not_found_404(e):
                        return call_chat()
                    raise

            except Exception as e:
                msg = str(e).lower()
                retryable = any(
                    k in msg
                    for k in (
                        "rate limit", "429",
                        "timeout", "timed out",
                        "temporarily", "temporary",
                        "connection", "reset", "econnreset",
                        "502", "503", "504", "server error", "internal error",
                    )
                )

                if (not retryable) or (attempt >= max_retries):
                    raise

                sleep_s = base_backoff * (2 ** attempt) + random.random() * 0.25
                time.sleep(sleep_s)

        raise RuntimeError("unreachable retry loop")
    
    def process_one(
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
        prefix = self.split_prefix(prompt_path)
        out_path = out_dir / f"{prefix}{self.PRED_SUFFIX}"

        if (not overwrite) and self.is_completed_output(out_path, min_bytes=min_output_bytes):
            return prompt_path, out_path, None

        prompt_text = prompt_path.read_text(encoding="utf-8", errors="ignore")

        try:
            answer = self.call_openai_with_retry(
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
            self.atomic_write_text(out_path, answer if answer.endswith("\n") else (answer + "\n"))
            return prompt_path, out_path, None

        except Exception as e:
            return prompt_path, None, str(e)
    
    def optimize_with_llm(self, in_dir: str, out_dir: str = "", model: str = "gpt-5", api_mode: str = "auto", workers: int = 50, max_output_tokens: int = 8192, temperature: float = 1.0, truncation: str = "auto", store: bool = False, overwrite: bool = False, min_output_bytes: int = 16, max_retries: int = 0, base_backoff: float = 0.8, one: str = ""):
        in_dir = Path(in_dir)
        if not in_dir.is_dir():
            raise SystemExit(f"--in_dir is not a directory: {in_dir}")

        out_dir = Path(out_dir) if out_dir else in_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        prompt_files: list[Path] = []

        if one:
            one_path = Path(one)
            if one_path.exists() and one_path.is_file():
                prompt_files = [one_path]
            else:
                cand = in_dir / one
                if cand.exists() and cand.is_file():
                    prompt_files = [cand]
                else:
                    prefix = one
                    if prefix.endswith(self.PROMPT_SUFFIX):
                        prefix = prefix[: -len(self.PROMPT_SUFFIX)]
                    cand2 = in_dir / f"{prefix}{self.PROMPT_SUFFIX}"
                    if cand2.exists() and cand2.is_file():
                        prompt_files = [cand2]

            if not prompt_files:
                raise SystemExit(f"--one not found: {one}")
        else:
            prompt_files = sorted(in_dir.glob("*.prompt.ll"))
            if not prompt_files:
                raise SystemExit(f"No files matched: {in_dir}/*.prompt.ll")

        self.log(f"Found prompts: {len(prompt_files)}" + (f" (single: {prompt_files[0].name})" if one else ""))
        self.log(f"Input : {in_dir}")
        self.log(f"Output: {out_dir}")
        self.log(f"Model : {model}")
        self.log(f"API   : {api_mode} (auto will fall back to chat on 404)")
        self.log(f"Workers: {1 if one else workers}")

        ok = 0
        skipped = 0
        failed = 0

        if one:
            p = prompt_files[0]
            prompt_path, out_path, err = self.process_one(
                p,
                out_dir,
                model,
                max_output_tokens,
                temperature,
                truncation,
                store,
                api_mode,
                min_output_bytes,
                overwrite,
                max_retries,
                base_backoff,
            )
            if err is None and out_path is not None:
                self.log(f"OK: {out_path}")
                self.log("Tip: open the output file to inspect the raw model text.")
            else:
                raise SystemExit(f"ERROR: {prompt_path.name} -> {err}")
            return

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = []
            for p in prompt_files:
                prefix = self.split_prefix(p)
                out_path = out_dir / f"{prefix}{self.PRED_SUFFIX}"
                if (not overwrite) and self.is_completed_output(out_path, min_bytes=min_output_bytes):
                    skipped += 1
                    continue
                futs.append(
                    ex.submit(
                        self.process_one,
                        p,
                        out_dir,
                        model,
                        max_output_tokens,
                        temperature,
                        truncation,
                        store,
                        api_mode,
                        min_output_bytes,
                        overwrite,
                        max_retries,
                        base_backoff,
                    )
                )

            total_run = len(futs)
            self.log(f"To run now: {total_run} (skipped existing: {skipped})")

            for idx, fut in enumerate(as_completed(futs), start=1):
                prompt_path, out_path, err = fut.result()
                if err is None and out_path is not None:
                    ok += 1
                    if idx % 20 == 0 or idx == total_run:
                        self.log(f"[{idx}/{total_run}] OK: {out_path.name}")
                else:
                    failed += 1
                    self.log(f"[{idx}/{total_run}] ERROR: {prompt_path.name} -> {err}")

        self.log(f"Done. ok={ok} skipped={skipped} failed={failed}")
        return out_dir
