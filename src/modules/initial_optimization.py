import os
import re
import math
import time
from pathlib import Path
from typing import List, Union, Tuple, Dict, Optional

import torch
import torch.multiprocessing as mp
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from tqdm import tqdm
import pandas as pd

# 定义Args类在模块级别，使其可以被pickle序列化
class Args:
    def __init__(self, label, data_path, out_dir, batch_size, gpus, save_all):
        self.label = label
        self.data_path = data_path
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.gpus = gpus
        self.save_all = save_all

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
    
    def gpu_worker(self, rank: int, gpu_id: int, args, data: List[Dict], subset_indices: List[int], return_dict):
        try:
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
            
            out_dir = Path(args.out_dir)
            save_subdir_name = "all" if args.save_all else "wrong"
            save_file_dir = out_dir / save_subdir_name
            save_file_dir.mkdir(parents=True, exist_ok=True)

            tsv_part_file = out_dir / f"temp_part_{rank}_{args.label}.tsv"
            if not tsv_part_file.exists():
                with open(tsv_part_file, "w", encoding="utf-8") as f:
                    f.write("idx\tok\ttruth\tpred\n")

            print(f"[Worker-{rank}] Launching on GPU {gpu_id} | Data samples: {len(subset_indices)}")

            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            eos_token_ids = [tokenizer.eos_token_id]
            stop_str = "</code>"
            stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)
            if len(stop_ids) == 1:
                eos_token_ids.append(stop_ids[0])

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                device_map=None 
            ).to(device)

            adapter_path = Path(self.adapter_path)
            checkpoint_dir = self.pick_latest_checkpoint(adapter_path)
            print(f"[Worker-{rank}] Loading adapter from: {checkpoint_dir}")

            model = PeftModel.from_pretrained(model, checkpoint_dir)
            model = model.merge_and_unload()
            model.eval()

            # 使用传递的数据，不再重复加载数据集
            my_dataset = [data[i] for i in subset_indices]
            
            prompts = [self.get_prompt(item["prompt"]) for item in my_dataset]
            raw_full_texts = [item["prompt"] for item in my_dataset] 
            raw_truths_code = [self.get_res(item["prompt"]) for item in my_dataset]
            raw_indices = [subset_indices[i] for i in range(len(subset_indices))]

            batch_size = args.batch_size
            total_samples = len(prompts)
            num_batches = math.ceil(total_samples / batch_size)
            
            acc_count = 0
            processed_count = 0

            iterator = tqdm(range(num_batches), desc=f"GPU-{gpu_id}", position=rank)

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
                    ).to(device)

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

                    tsv_lines = []
                    for j, response in enumerate(batch_responses):
                        global_idx = start + j
                        raw_idx = raw_indices[global_idx]

                        if tokenizer.eos_token in response:
                            clean_response = response.split(tokenizer.eos_token)[0]
                        else:
                            clean_response = response
                        
                        pred_code = self.get_res(clean_response)
                        truth_code = raw_truths_code[global_idx]
                        
                        is_ok = (pred_code.strip() == truth_code.strip())
                        if is_ok:
                            acc_count += 1

                        if (not is_ok) or args.save_all:
                            base_name = save_file_dir / f"{raw_idx}"
                            
                            with open(f"{base_name}.model.predict.ll", "w", encoding="utf-8") as f:
                                f.write(clean_response)
                            with open(f"{base_name}.truth.ll", "w", encoding="utf-8") as f:
                                f.write(raw_full_texts[global_idx])
                            with open(f"{base_name}.prompt.ll", "w", encoding="utf-8") as f:
                                f.write(batch_prompts[j])

                        clean_truth = raw_full_texts[global_idx].replace('\n', '\\n').replace('\t', '    ')
                        clean_pred = clean_response.replace('\n', '\\n').replace('\t', '    ')
                        ok_flag = "1" if is_ok else "0"
                        tsv_lines.append(f"{raw_idx}\t{ok_flag}\t{clean_truth}\t{clean_pred}\n")
                    
                    with open(tsv_part_file, "a", encoding="utf-8") as f:
                        for line in tsv_lines:
                            f.write(line)

                    processed_count += len(batch_responses)
                    current_acc = acc_count / processed_count
                    iterator.set_postfix(acc=f"{current_acc:.2%}")

            print(f"[Worker-{rank}] Done. Final Acc: {acc_count}/{total_samples} = {acc_count/total_samples:.2%}")
            return_dict[rank] = str(tsv_part_file)

        except Exception as e:
            print(f"[Worker-{rank}] Error: {e}")
            import traceback
            traceback.print_exc()
            return_dict[rank] = None
    
    def generate_optimization_strategies(self, data_path: str, out_dir: str, label: str, batch_size: int = 4, gpus: str = "0,1,2", save_all: bool = False):
        # 使用模块级别的Args类
        args = Args(label, data_path, out_dir, batch_size, gpus, save_all)
        mp.set_start_method('spawn', force=True)
        
        gpu_list = [int(x) for x in args.gpus.split(",")]
        
        out_dir = self.ensure_dir(args.out_dir)
        sub_dir = out_dir / ("all" if args.save_all else "wrong")
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Output Dir : {out_dir}")
        print(f"[INFO] Strategy   : Strict 8192 length + EOS Cleaning")
        
        # 检查data_path是否是目录且包含IR文件
        data_path_obj = Path(args.data_path)
        dataset_data = []
        
        if data_path_obj.is_dir():
            # 查找目录中的IR文件
            ir_files = list(data_path_obj.glob("*.ll"))
            
            if ir_files:
                print(f"[INFO] Found {len(ir_files)} IR files in directory {data_path_obj}")
                # 创建临时数据集
                for i, ir_file in enumerate(ir_files):
                    with open(ir_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # 构建prompt格式，参考原始数据集的结构
                    prompt = f"[INST]Given the following LLVM IR, propose key optimization transformation steps to outperform LLVM -O3.\nWrite your answer inside a single <code>...</code> block.\nInside <code>, write ONLY <step></step> blocks.\nEach step MUST follow this format:\n<step>\n**Transformation**: [Brief name of the optimization]\n**Change**: [A short description of the change applied to the code]\n</step>\nDo NOT output optimized IR.\n\n<ir>{content}</ir>[/INST]\n"
                    dataset_data.append({"prompt": prompt})
                
                total_samples = len(dataset_data)
                print(f"[INFO] Total samples: {total_samples}")
            else:
                # 如果目录中没有IR文件，尝试作为预格式化数据集加载
                try:
                    full_dataset = datasets.load_from_disk(args.data_path)
                    dataset_data = [item['prompt'] for item in full_dataset]
                    total_samples = len(dataset_data)
                    print(f"[INFO] Loaded pre-formatted dataset with {total_samples} samples")
                except Exception as e:
                    print(f"[ERROR] Failed to load dataset: {e}")
                    raise
        else:
            # 尝试作为预格式化数据集加载
            full_dataset = datasets.load_from_disk(args.data_path)
            dataset_data = [item for item in full_dataset]
            total_samples = len(dataset_data)
            print(f"[INFO] Total samples: {total_samples}")

        indices = list(range(total_samples))
        chunk_size = math.ceil(total_samples / len(gpu_list))
        chunks = [indices[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]
        
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        
        for rank in range(len(gpu_list)):
            temp_f = out_dir / f"temp_part_{rank}_{args.label}.tsv"
            if temp_f.exists():
                os.remove(temp_f)

        start_time = time.time()
        for rank, gpu_id in enumerate(gpu_list):
            if rank >= len(chunks):
                break
            p = mp.Process(
                target=self.gpu_worker,
                args=(rank, gpu_id, args, dataset_data, chunks[rank], return_dict)
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
            
        print(f"[INFO] Finished in {time.time() - start_time:.2f}s")

        all_dfs = []
        temp_files = []
        for rank in range(len(processes)):
            fpath = return_dict.get(rank)
            if fpath and os.path.exists(fpath):
                try:
                    all_dfs.append(pd.read_csv(fpath, sep='\t'))
                    temp_files.append(fpath)
                except:
                    pass
        
        if all_dfs:
            final_df = pd.concat(all_dfs).sort_values("idx")
            final_out = out_dir / f"pred_{args.label}.tsv"
            final_df.to_csv(final_out, sep='\t', index=False)
            
            acc_rate = final_df['ok'].mean()
            print(f"\n[SUCCESS] Final Accuracy: {acc_rate:.2%}")
            
            for f in temp_files:
                try:
                    os.remove(f)
                except:
                    pass
        else:
            print("[ERROR] No results generated.")
        
        return out_dir
