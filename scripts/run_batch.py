#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
示例脚本：批量优化LLVM IR文件

使用方法：
    python run_batch.py --input /path/to/data_dir --output /path/to/output_dir

该脚本演示了如何使用重构后的系统批量优化LLVM IR文件。
"""

import argparse
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="批量优化LLVM IR文件")
    parser.add_argument("--input", required=True, help="输入数据目录路径")
    parser.add_argument("--output", required=True, help="输出目录路径")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查输入目录是否存在
    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_dir():
        print(f"错误：输入目录不存在: {args.input}")
        return
    
    # 检查输出目录是否存在，不存在则创建
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 构建命令
    cmd = f"python ../src/main.py \
        --mode batch \
        --input {args.input} \
        --output {args.output}"
    
    print("执行命令:")
    print(cmd)
    print("\n开始批量优化...")
    
    # 执行命令
    os.system(cmd)
    
    print("\n批量优化完成！")
    print(f"优化结果保存在: {args.output}")

if __name__ == "__main__":
    main()
