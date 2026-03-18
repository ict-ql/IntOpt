import os
import re
import math
import time
from pathlib import Path
from typing import List, Union, Tuple, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class InitialOptimizationGenerator:
    def __init__(self, model_path: str, adapter_path: str):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.tokenizer = None
        self.model = None

    def get_prompt(self, full_data: str) -> str:
        if "[/INST]" in full_data:
            return full_data.split("[/INST]")[0] + "[/INST]"
        return full_data

    def get_res(self, model_res: str) -> str:
        if "</s>" in model_res:
            model_res = model_res.split("</s>")[0]

        pattern = r"<code>(.*?)</code>"
        if "[/INST]" in model_res:
            res_part = model_res.split("[/INST]")[-1]
        else:
            res_part = model_res

        m = re.findall(pattern, res_part, re.DOTALL)
        return m[0] if m else ""

    def ensure_dir(self, p: Union[str, Path]) -> Path:
        path = Path(p)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def pick_latest_checkpoint(self, adapters_root: Path) -> str:
        if not adapters_root.exists():
            if adapters_root.name.startswith("checkpoint-"):
                return str(adapters_root)
            return str(adapters_root)
        if adapters_root.name.startswith("checkpoint-"):
            return str(adapters_root)
        cands = [p for p in adapters_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
        if not cands:
            return str(adapters_root)

        def score(p: Path) -> int:
            m = re.match(r"checkpoint-(\d+)", p.name)
            return int(m.group(1)) if m else -1

        return str(max(cands, key=score))

    def _load_model(self, gpus: str):
        """加载模型和tokenizer，只加载一次，跨多卡放置"""
        if self.model is not None:
            return

        gpu_list = [int(x) for x in gpus.split(",")]
        print(f"[INFO] Loading model on GPUs: {gpu_list}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 构建 device_map，让模型自动分布到指定的多张卡上
        max_memory = {gpu_id: "80GiB" for gpu_id in gpu_list}
        max_memory["cpu"] = "0GiB"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map="auto",
            max_memory=max_memory,
        )

        adapter_path = Path(self.adapter_path)
        checkpoint_dir = self.pick_latest_checkpoint(adapter_path)
        print(f"[INFO] Loading adapter from: {checkpoint_dir}")

        self.model = PeftModel.from_pretrained(self.model, checkpoint_dir)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        print("[INFO] Model loaded and ready.")

    def generate_optimization_strategies(self, data_path: str, out_dir: str, batch_size: int = 4, gpus: str = "0,1,2", save_all: bool = False):
        out_dir = self.ensure_dir(out_dir)

        print(f"[INFO] Output Dir : {out_dir}")
        print(f"[INFO] Strategy   : Strict 8192 length + EOS Cleaning")

        # 检查data_path是否是目录且包含IR文件
        data_path_obj = Path(data_path)
        dataset_data = []

        if data_path_obj.is_dir():
            ir_files = list(data_path_obj.glob("*.ll"))
            if ir_files:
                print(f"[INFO] Found {len(ir_files)} IR files in directory {data_path_obj}")
                for ir_file in ir_files:
                    with open(ir_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    prompt = (
                        f"[INST]Given the following LLVM IR, propose key optimization transformation steps to outperform LLVM -O3.\n"
                        f"Write your answer inside a single <code>...</code> block.\n"
                        f"Inside <code>, write ONLY <step></step> blocks.\n"
                        f"Each step MUST follow this format:\n"
                        f"<step>\n"
                        f"**Transformation**: [Brief name of the optimization]\n"
                        f"**Change**: [A short description of the change applied to the code]\n"
                        f"</step>\n"
                        f"Do NOT output optimized IR.\n\n"
                        f"<ir>{content}</ir>[/INST]\n"
                    )
                    dataset_data.append({"prompt": prompt, "filename": ir_file.stem})

                total_samples = len(dataset_data)
                print(f"[INFO] Total samples: {total_samples}")
            else:
                msg = f"[ERROR] There is no IR files in `{data_path_obj}`"
                print(msg)
                raise SystemExit(msg)
        else:
            msg = f"[ERROR] `{data_path_obj}` is not a directory."
            print(msg)
            raise SystemExit(msg)

        # 加载模型（只加载一次）
        self._load_model(gpus)

        tokenizer = self.tokenizer
        model = self.model

        eos_token_ids = [tokenizer.eos_token_id]
        stop_str = "</code>"
        stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)
        if len(stop_ids) == 1:
            eos_token_ids.append(stop_ids[0])

        prompts = [self.get_prompt(item["prompt"]) for item in dataset_data]
        raw_full_texts = [item["prompt"] for item in dataset_data]
        raw_truths_code = [self.get_res(item["prompt"]) for item in dataset_data]

        num_batches = math.ceil(total_samples / batch_size)

        start_time = time.time()
        iterator = tqdm(range(num_batches), desc="Generating")

        with torch.inference_mode():
            for i in iterator:
                start = i * batch_size
                end = min((i + 1) * batch_size, total_samples)
                batch_prompts = prompts[start:end]

                inputs = tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt"
                ).to(model.device)

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=8192,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids
                )

                gen_input_len = inputs["input_ids"].shape[1]
                generated_tokens = output_ids[:, gen_input_len:]
                batch_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

                for j, response in enumerate(batch_responses):
                    global_idx = start + j
                    if tokenizer.eos_token in response:
                        clean_response = response.split(tokenizer.eos_token)[0]
                    else:
                        clean_response = response

                    base_name = out_dir / dataset_data[global_idx]["filename"]

                    with open(f"{base_name}.model.predict.ll", "w", encoding="utf-8") as f:
                        f.write(clean_response)
                    with open(f"{base_name}.truth.ll", "w", encoding="utf-8") as f:
                        f.write(raw_full_texts[global_idx])
                    with open(f"{base_name}.prompt.ll", "w", encoding="utf-8") as f:
                        f.write(batch_prompts[j])

        print(f"[INFO] Finished in {time.time() - start_time:.2f}s")

        return out_dir
