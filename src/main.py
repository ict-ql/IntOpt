import argparse
import os
import yaml
import shutil
import re
from pathlib import Path
from typing import Optional

from modules.initial_optimization import InitialOptimizationGenerator
from modules.strategy_mapping import StrategyMapping
from modules.strategy_refinement import StrategyRefinement
from modules.llm_optimization import LLM_Optimizer
from modules.post_processing import PostProcessor

class Config:
    def __init__(self, config_file=None, **kwargs):
        # 加载默认配置
        default_config = {
            'model_path': "/home/amax/wzh/Models/llm-compiler-13b-ftd",
            'adapter_path': "/home/amax/yangz/Projects/2025-IR-Optset-Plus/05-Checkpoints/ICML/icml-13b-ftd-step-5k-4000-final",
            'passregistry_def': "/home/amax/yangz/Env/llvm-project/llvm/lib/Passes/PassRegistry.def",
            'llvm_lib_root': "/home/amax/yangz/Env/llvm-project/llvm/lib",
            'opt_bin': "/home/amax/yangz/Env/llvm-project/build/bin/opt",
            'll_dir': "/home/amax/yangz/Projects/2025-IR-Dataset/ICML/qiu-extend-5k",
            'base_url': "https://cloud.infini-ai.com/maas/v1",
            'api_key': "sk-wa5ipdlq2zi2tmjt",
            'batch_size': 4,
            'gpus': "0,1,2"
        }
        
        # 从配置文件加载
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
            # 合并配置
            if 'model' in yaml_config:
                default_config.update(yaml_config['model'])
            if 'llvm' in yaml_config:
                default_config.update(yaml_config['llvm'])
            if 'llm' in yaml_config:
                default_config.update(yaml_config['llm'])
            if 'run' in yaml_config:
                default_config.update(yaml_config['run'])
        
        # 从命令行参数覆盖
        default_config.update(kwargs)
        
        # 设置属性
        for key, value in default_config.items():
            setattr(self, key, value)

class IR_Optimizer:
    def __init__(self, config):
        self.config = config
        self.initial_optimizer = InitialOptimizationGenerator(
            model_path=config.model_path,
            adapter_path=config.adapter_path
        )
        self.strategy_mapper = StrategyMapping(
            passregistry_def=config.passregistry_def,
            llvm_lib_root=config.llvm_lib_root,
            opt_bin=config.opt_bin
        )
        self.strategy_refiner = StrategyRefinement(
            opt_bin=config.opt_bin
        )
        self.llm_optimizer = LLM_Optimizer(
            base_url=config.base_url,
            api_key=config.api_key
        )
        self.post_processor = PostProcessor()
    
    def optimize_single_file(self, input_file: str, output_dir: str):
        """优化单个LLVM IR文件"""
        input_path = Path(input_file)
        if not input_path.exists():
            raise SystemExit(f"Input file not found: {input_file}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Optimizing single file: {input_file}")
        print(f"Output directory: {output_dir}")
        
        # 步骤1: 创建临时目录和文件结构
        temp_dir = output_path / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 步骤2: 创建临时数据集结构（包含IR文件）
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 将输入IR文件复制到临时数据集目录
        temp_input_in_dataset = dataset_dir / f"{input_path.name}"
        shutil.copy2(input_file, temp_input_in_dataset)
        
        # 步骤3: 生成初始优化策略
        print("Step 1: Initial optimization strategy generation")
        initial_out_dir = temp_dir / "step1_initial"
        
        # 调用初始优化策略生成模块
        initial_out_dir = self.initial_optimizer.generate_optimization_strategies(
            data_path=str(dataset_dir),
            out_dir=str(initial_out_dir),
            batch_size=1, # because just single file
            gpus=self.config.gpus,
            save_all=False
        )
        
        # 步骤4: 优化策略映射
        print("\nStep 2: Strategy mapping")
        mapping_out_dir = temp_dir / "step2_mapping"
        
        # 调用策略映射模块
        mapping_out_dir = self.strategy_mapper.map_strategies_to_passes(
            in_dir=str(initial_out_dir),
            out_dir=str(mapping_out_dir),
            topk=3,
            emit="tokens"
        )
        
        # 步骤5: 优化策略精化
        print("\nStep 3: Strategy refinement")
        refinement_out_dir = temp_dir / "step3_refinement"
        
        # 调用策略精化模块
        refinement_out_dir = self.strategy_refiner.refine_strategies(
            in_dir=mapping_out_dir,
            ll_dir=dataset_dir,
            initial_dir=initial_out_dir,
            out_dir=str(refinement_out_dir),
            timeout=self.config.timeout,
            verify_timeout=self.config.verify_timeout
        )
        
        #调用LLM实现strategy refinement
        refinement_out_dir = self.llm_optimizer.optimize_with_llm(
            in_dir=refinement_out_dir,
            out_dir=str(refinement_out_dir),
            model=self.config.llm_model,
            api_mode=self.config.api_mode,
            workers=1
        )

        # 步骤6: LLM调用优化
        print("\nStep 4: LLM optimization")
        llm_out_dir = temp_dir / "step4_realization"
        llm_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 实现prompt生成功能
        llm_out_dir = self.generate_optimized_prompt(refinement_out_dir, llm_out_dir)
        
        # 调用LLM优化模块
        llm_out_dir = self.llm_optimizer.optimize_with_llm(
            in_dir=llm_out_dir,
            out_dir=str(llm_out_dir),
            model=self.config.llm_model,
            api_mode=self.config.api_mode,
            workers=1
        )
        
        # 后处理一下
        print("\nStep 5: Post processing")
        llm_out_dir = self.post_processor.post_process(
            in_dir=str(llm_out_dir)
        )

        # 步骤7: 处理优化结果
        output_file = output_path / f"{input_path.stem}.optimized.ll"
        # 查找优化后的文件并复制到输出目录
        optimized_files = Path(llm_out_dir).glob("*.model.predict.ll")
        if optimized_files:
            first_file = next(optimized_files)
            with open(first_file, "r", encoding="utf-8") as f:
                code = f.read()

            code = self.get_code(code)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(code)

            print(f"Optimization completed. Output file: {output_file}")
            return output_path
        else:
            print(f"[WARN]: There is no optimized IR file for {input_path}")
            return None
    
    def optimize_batch(self, data_path: str, output_dir: str):
        """批量优化LLVM IR文件"""
        # 步骤1: 初始优化策略生成
        print("Step 1: Initial optimization strategy generation")
        initial_out_dir = self.initial_optimizer.generate_optimization_strategies(
            data_path=data_path,
            out_dir=os.path.join(output_dir, "step1_initial"),
            label="icml-13b-ftd-step-5k-4000-final",
            batch_size=self.config.batch_size,
            gpus=self.config.gpus,
            save_all=False
        )
        
        # 步骤2: 优化策略映射
        print("\nStep 2: Strategy mapping")
        mapping_out_dir = self.strategy_mapper.map_strategies_to_passes(
            in_dir=os.path.join(initial_out_dir, "wrong"),
            out_dir=os.path.join(output_dir, "step2_mapping"),
            topk=3,
            emit="tokens"
        )
        
        # @TODO: 记得改，这个也都对不上
        # 步骤3: 优化策略精化
        print("\nStep 3: Strategy refinement")
        refinement_out_dir = self.strategy_refiner.refine_strategies(
            in_dir=mapping_out_dir,
            ll_dir=self.config.ll_dir,
            out_dir=os.path.join(output_dir, "step3_refinement"),
            timeout=30,
            verify_timeout=4
        )
        
        # 步骤4: LLM调用优化
        print("\nStep 4: LLM optimization")
        llm_out_dir = self.llm_optimizer.optimize_with_llm(
            in_dir=refinement_out_dir,
            out_dir=os.path.join(output_dir, "step4_llm"),
            model="gpt-5",
            api_mode="auto",
            workers=50
        )
        
        return llm_out_dir
    
    def _unescape_html(self, s: str) -> str:
        HTML_UNESCAPE = (
            ("&lt;", "<"),
            ("&gt;", ">"),
            ("&amp;", "&"),
        )
        for a, b in HTML_UNESCAPE:
            s = s.replace(a, b)
        return s

    def _clean_block(self, s: str) -> str:
        s = self._unescape_html(s or "")
        s = s.strip("\n\r\t ")
        return s

    def get_advice(self, text):
        code_blocks: List[str] = []
        ADVICE_RE = re.compile(r"<advice>(.*?)</advice>", re.DOTALL | re.IGNORECASE)
        for m in ADVICE_RE.finditer(text):
            blk = self._clean_block(m.group(1))
            # Filter out the common placeholder caught from "single <code>...</code> block"
            if blk == "..." or not blk:
                continue
            code_blocks.append(blk)
        if len(code_blocks) == 0:
            return None
        assert (len(code_blocks) == 1)
        return code_blocks[0]
    
    def get_code(self, text):
        code_blocks: List[str] = []
        ADVICE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
        for m in ADVICE_RE.finditer(text):
            blk = self._clean_block(m.group(1))
            # Filter out the common placeholder caught from "single <code>...</code> block"
            if blk == "..." or not blk:
                continue
            code_blocks.append(blk)
        if len(code_blocks) == 0:
            return None
        assert (len(code_blocks) == 1)
        return code_blocks[0]

    def generate_optimized_prompt(self, refinement_out_dir, llm_out_dir):
        advice_info_old_header = "You may refer to the following advice, but feel free to adapt, extend, or deviate from it as you see fit."
        advice_info_new_header = "You can refer to the following advice."
        old_footer = "Please output the final optimization advice wrapped in <advice>...</advice> and the full optimized LLVM IR wrapped in <code>...</code>."
        new_footer = "Please output the full optimized LLVM IR wrapped in <code>...</code>."


        """生成优化后的prompt"""
        import re
        
        # 步骤1: 提取advice内容
        print("Create optimization prompt...")
        
        # 查找所有*.prompt.ll文件
        prompt_files = list(Path(refinement_out_dir).glob("*.prompt.ll"))
        for file in prompt_files:
            model_pred_name = file.stem[:-len(".prompt")]
            if not Path(f"{file.parent}/{model_pred_name}.model.predict.ll").is_file():
                print(f"[WARN] There is no refined strategy in {file.parent} for {model_pred_name}")
                continue
            with open(f"{file.parent}/{model_pred_name}.model.predict.ll", "r") as f:
                ctx = f.read()
                new_advice = self.get_advice(ctx)
            with open(file, "r") as f:
                ctx = f.read()
                old_advice = self.get_advice(ctx)
                if old_advice == None:
                    ctx = ctx.replace("</code>\n", f"</code>\n\n{advice_info_new_header}\n<advice>\n{new_advice}\n</advice>\n")
                    # print(ctx)
                else:
                    ctx = ctx.replace(old_advice, new_advice)
                ctx = ctx.replace(advice_info_old_header, advice_info_new_header)
                ctx = ctx.replace(old_footer, new_footer)
            # 保存新prompt到输出目录
            output_prompt_file = f"{llm_out_dir}/{model_pred_name}.prompt.ll"
            with open(output_prompt_file, "w") as f:
                f.write(ctx)

        print(f"New prompt saved to: {llm_out_dir}")
        return llm_out_dir

    def run(self, input_path: str, output_dir: str, mode: str):
        """运行优化流程"""
        if mode == "single":
            return self.optimize_single_file(input_path, output_dir)
        elif mode == "batch":
            return self.optimize_batch(input_path, output_dir)
        else:
            raise SystemExit(f"Invalid mode: {mode}")

def parse_args():
    parser = argparse.ArgumentParser(description="LLVM IR Optimizer")
    parser.add_argument("--mode", choices=["single", "batch"], required=True, help="Optimization mode: single file or batch")
    parser.add_argument("--input", required=True, help="Input file (single mode) or data directory (batch mode)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="../config/config.yaml", help="Path to configuration file")
    
    # 模型配置
    parser.add_argument("--model_path", help="Path to the base model")
    parser.add_argument("--adapter_path", help="Path to the adapter")
    
    # LLVM配置
    parser.add_argument("--passregistry_def", help="Path to PassRegistry.def")
    parser.add_argument("--llvm_lib_root", help="Path to LLVM lib directory")
    parser.add_argument("--opt_bin", help="Path to opt binary")
    parser.add_argument("--ll_dir", help="Path to LLVM IR files directory")
    
    # LLM配置
    parser.add_argument("--base_url", help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", help="API key")
    
    # 运行配置
    parser.add_argument("--batch_size", type=int, help="Batch size for initial optimization")
    parser.add_argument("--gpus", help="GPUs to use for initial optimization")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 过滤出非None的参数
    config_kwargs = {k: v for k, v in vars(args).items() if v is not None and k not in ['mode', 'input', 'output', 'config']}
    
    # 创建配置对象
    config = Config(config_file=args.config, **config_kwargs)
    
    optimizer = IR_Optimizer(config)
    result_dir = optimizer.run(args.input, args.output, args.mode)
    if result_dir != None:
        print(f"\nOptimization completed successfully!")
        print(f"Result directory: {result_dir}")

if __name__ == "__main__":
    main()
