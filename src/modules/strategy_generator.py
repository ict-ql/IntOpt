"""Generate initial optimization strategies for LLVM IR files using a
fine-tuned local LLM (loaded once, distributed across GPUs)."""

import math
import re
import time
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from modules.utils import ensure_dir, log


class StrategyGenerator:
    """Load a causal-LM + LoRA adapter once, then batch-generate
    <step>-based optimization strategies for *.ll files."""

    def __init__(self, model_path: str, adapter_path: str):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.tokenizer = None
        self.model = None

    # ------------------------------------------------------------------
    # Prompt / response helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(ir_content: str) -> str:
        return (
            "[INST]Given the following LLVM IR, propose key optimization "
            "transformation steps to outperform LLVM -O3.\n"
            "Write your answer inside a single <code>...</code> block.\n"
            "Inside <code>, write ONLY <step></step> blocks.\n"
            "Each step MUST follow this format:\n"
            "<step>\n"
            "**Transformation**: [Brief name of the optimization]\n"
            "**Change**: [A short description of the change applied to the code]\n"
            "</step>\n"
            "Do NOT output optimized IR.\n\n"
            f"<ir>{ir_content}</ir>[/INST]\n"
        )

    @staticmethod
    def _strip_inst(full_data: str) -> str:
        """Return only the [INST]…[/INST] portion (the prompt)."""
        if "[/INST]" in full_data:
            return full_data.split("[/INST]")[0] + "[/INST]"
        return full_data

    # ------------------------------------------------------------------
    # Checkpoint resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_latest_checkpoint(adapters_root: Path) -> str:
        if not adapters_root.exists() or adapters_root.name.startswith("checkpoint-"):
            return str(adapters_root)
        cands = [
            p for p in adapters_root.iterdir()
            if p.is_dir() and p.name.startswith("checkpoint-")
        ]
        if not cands:
            return str(adapters_root)

        def _score(p: Path) -> int:
            m = re.match(r"checkpoint-(\d+)", p.name)
            return int(m.group(1)) if m else -1

        return str(max(cands, key=_score))

    # ------------------------------------------------------------------
    # Model loading (once)
    # ------------------------------------------------------------------

    def _load_model(self, gpus: str) -> None:
        if self.model is not None:
            return

        gpu_list = [int(x) for x in gpus.split(",")]
        log(f"Loading model on GPUs: {gpu_list}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        max_memory = {gid: "80GiB" for gid in gpu_list}
        max_memory["cpu"] = "0GiB"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map="auto",
            max_memory=max_memory,
        )

        ckpt = self._pick_latest_checkpoint(Path(self.adapter_path))
        log(f"Loading adapter from: {ckpt}")

        self.model = PeftModel.from_pretrained(self.model, ckpt)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        log("Model loaded and ready.")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        data_path: str,
        out_dir: str,
        batch_size: int = 4,
        gpus: str = "0,1,2",
    ) -> Path:
        """Read *.ll from *data_path*, generate optimisation strategies,
        write *.model.predict.ll and *.prompt.ll into *out_dir*.
        Returns the output directory as a Path."""

        out_dir = ensure_dir(out_dir)
        data_path_obj = Path(data_path)

        if not data_path_obj.is_dir():
            raise SystemExit(f"data_path is not a directory: {data_path_obj}")

        ir_files = sorted(data_path_obj.glob("*.ll"))
        if not ir_files:
            raise SystemExit(f"No *.ll files found in {data_path_obj}")

        log(f"Output dir : {out_dir}")
        log(f"Found {len(ir_files)} IR files in {data_path_obj}")

        # Build dataset, skipping files that already have results
        dataset: List[Dict[str, str]] = []
        skipped = 0
        for ir_file in ir_files:
            stem = ir_file.stem
            pred = out_dir / f"{stem}.model.predict.ll"
            prompt = out_dir / f"{stem}.prompt.ll"
            if pred.exists() and prompt.exists():
                skipped += 1
                continue
            content = ir_file.read_text(encoding="utf-8")
            dataset.append({
                "prompt": self._build_prompt(content),
                "filename": stem,
            })
        total = len(dataset)

        if skipped:
            log(f"Skipped {skipped} files with existing results")
        if total == 0:
            log("All files already processed, nothing to generate")
            return out_dir

        # Load model (once across calls)
        self._load_model(gpus)
        tokenizer = self.tokenizer
        model = self.model

        eos_ids = [tokenizer.eos_token_id]
        stop_ids = tokenizer.encode("</code>", add_special_tokens=False)
        if len(stop_ids) == 1:
            eos_ids.append(stop_ids[0])

        prompts = [self._strip_inst(item["prompt"]) for item in dataset]
        num_batches = math.ceil(total / batch_size)

        t0 = time.time()
        with torch.inference_mode():
            for i in tqdm(range(num_batches), desc="Generating"):
                start = i * batch_size
                end = min(start + batch_size, total)
                batch_prompts = prompts[start:end]

                inputs = tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt",
                ).to(model.device)

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=8192,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )

                gen_start = inputs["input_ids"].shape[1]
                responses = tokenizer.batch_decode(
                    output_ids[:, gen_start:], skip_special_tokens=False,
                )

                for j, response in enumerate(responses):
                    idx = start + j
                    # Strip trailing EOS
                    if tokenizer.eos_token in response:
                        response = response.split(tokenizer.eos_token)[0]

                    base = out_dir / dataset[idx]["filename"]
                    (base.parent / f"{base.name}.model.predict.ll").write_text(
                        response, encoding="utf-8",
                    )
                    (base.parent / f"{base.name}.prompt.ll").write_text(
                        batch_prompts[j], encoding="utf-8",
                    )

        log(f"Finished in {time.time() - t0:.2f}s")
        return out_dir
