# LLVM IR Optimizer

一个基于LLM的LLVM IR优化系统，支持单个文件和批处理模式的优化。

## 功能特点

- **模块化设计**：采用模块化架构，提高代码可维护性和可扩展性
- **多模式支持**：支持单个文件优化和批处理模式
- **完整的优化流程**：包含初始优化策略生成、优化策略映射、优化策略精化和LLM调用优化四个阶段
- **可配置性**：通过配置文件管理模型路径、数据路径等参数

## 目录结构

```
intop/
├── config/           # 配置文件
├── scripts/          # 示例脚本
├── src/              # 源代码
│   ├── modules/      # 核心模块
│   └── main.py       # 主入口
└── README.md         # 文档
```

## 核心模块

1. **initial_optimization.py**：初始优化策略生成模块，使用fine-tuned的llm-compiler-13b-ftd模型处理输入IR
2. **strategy_mapping.py**：优化策略映射模块，使用TF-IDF向量化和余弦相似度匹配查找最相近的Top-K pass token
3. **strategy_refinement.py**：优化策略精化模块，生成refine strategy所需的prompt，包含LLVM对应Transformation的analysis信息
4. **llm_optimization.py**：LLM调用优化模块，调用gpt-5模型获取refined后的strategy

## 配置文件

配置文件位于 `config/config.yaml`，包含以下配置项：

- **model**：模型路径配置
- **llvm**：LLVM相关配置
- **llm**：LLM相关配置
- **run**：运行配置
- **paths**：路径配置

## 使用方法

### 优化单个文件

```bash
cd scripts
python run_single_file.py --input /path/to/input.ll --output /path/to/output_dir
```

### 批量优化文件

```bash
cd scripts
python run_batch.py --input /path/to/data_dir --output /path/to/output_dir
```

### 直接使用主入口

```bash
cd src
python main.py --mode single --input /path/to/input.ll --output /path/to/output_dir
```

```bash
cd src
python main.py --mode batch --input /path/to/data_dir --output /path/to/output_dir
```

```bash
cd src
# the '/path/to/data_dir_unopt' stores AA.ll, the '/path/to/data_dir_opt' stores AA.optimized.ll
python main.py --mode diff_test --input /path/to/data_dir_unopt:/path/to/data_dir_opt --output /path/to/output_dir
```

## 依赖项

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- scikit-learn
- OpenAI
- BeautifulSoup4
- requests
- tqdm
- pandas

## 注意事项

1. 确保配置文件中的路径设置正确
2. 批量优化模式需要提供符合格式的数据集
3. 单个文件优化模式正在开发中，目前主要支持批处理模式

## 原始功能保留

重构后的系统完整保留了原始工程的核心功能：

1. 使用fine-tuned的llm-compiler-13b-ftd模型处理输入IR
2. 提取每个<step>中的Transformation + Change作为查询内容
3. 应用TF-IDF向量化处理
4. 使用余弦相似度(cosine similarity)查找最相近的Top-K pass token
5. 生成refine strategy所需的prompt，包含LLVM对应Transformation的analysis信息
6. 调用gpt-5模型获取refined后的strategy
7. 将refined strategy、analysis信息及未优化的IR整合为完整prompt
