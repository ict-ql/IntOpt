# IntOpt: Intent-Driven IR Optimization with Large Language Models

IntOpt is an automated LLVM IR optimization pipeline that uses a fine-tuned code LLM to generate optimization strategies, maps them to concrete LLVM passes, refines the strategies with analysis feedback, and finally calls a frontier LLM to produce optimized IR. It also includes differential testing and performance benchmarking to validate correctness and measure speedup.

## Pipeline Overview

```
Input .ll ──► Strategy Generation ──► Strategy Mapping ──► Strategy Refinement
                  (fine-tuned LLM)     (TF-IDF + cosine)    (LLVM analysis passes)
                                                                     │
                                                                     ▼
                                                            LLM Refinement
                                                            (frontier LLM)
                                                                     │
                                                                     ▼
              Output .optimized.ll ◄── Post-Processing ◄── LLM Realization
                                       (SSA cleanup,        (frontier LLM)
                                        const prop, etc.)
```

**Step 1 — Strategy Generation**: A fine-tuned `llm-compiler-13b-ftd` model reads the input IR and proposes high-level optimization intents (e.g., "loop unrolling", "dead code elimination").

**Step 2 — Strategy Mapping**: Each intent is vectorized with TF-IDF and matched against LLVM's `PassRegistry.def` via cosine similarity to find the top-K relevant analysis/transform passes.

**Step 3 — Strategy Refinement**: The matched passes are executed on the input IR via `opt`, and the resulting analysis output is injected into a prompt. A frontier LLM refines the strategy with this concrete feedback.

**Step 4 — LLM Realization**: The refined strategy, analysis info, and original IR are assembled into a final prompt. A frontier LLM produces the optimized IR.

**Step 5 — Post-Processing**: Cleans up LLM artifacts — strips markdown fences, removes redundant IR attributes, eliminates trivial bitcasts/SSA copies, and propagates constants.

## Project Structure

```
intop/
├── config/
│   └── config.yaml              # All configuration (model, LLVM, LLM, diff/perf testing)
├── scripts/
│   ├── run_single_file.py       # Convenience wrapper for single-file mode
│   └── run_batch.py             # Convenience wrapper for batch mode
├── src/
│   ├── main.py                  # CLI entry point & pipeline orchestrator (IROptimizer)
│   └── modules/
│       ├── strategy_generator.py  # Step 1: fine-tuned LLM inference
│       ├── strategy_mapping.py    # Step 2: TF-IDF pass matching
│       ├── strategy_refinement.py # Step 3: LLVM analysis + prompt construction
│       ├── llm_client.py          # Async batch LLM API client
│       ├── post_processing.py     # Step 5: IR cleanup transforms
│       ├── diff_testing.py        # Differential testing (libFuzzer)
│       ├── perf_testing.py        # Performance benchmarking
│       └── utils.py               # Shared helpers (log, tag extraction, etc.)
├── test/                          # Sample input .ll files
└── test.opt/                      # Sample pipeline output (step1–step4)
```

## Modes

### `single` — Optimize a single IR file

```bash
cd src
python main.py --mode single --input /path/to/input.ll --output /path/to/output_dir
```

### `batch` — Optimize a directory of IR files

```bash
cd src
python main.py --mode batch --input /path/to/data_dir --output /path/to/output_dir
```

### `diff_test` — Differential testing

Validates correctness of optimized IR by generating libFuzzer harnesses, compiling them with the original and optimized IR, and fuzzing to detect mismatches.

```bash
cd src
# original_dir contains *.ll, optimized_dir contains *.optimized.ll
python main.py --mode diff_test \
    --input /path/to/original_dir:/path/to/optimized_dir \
    --output /path/to/output_dir
```

Smart resumption: if `bins/`, `harness/`, or `combined/` already exist under the output directory, the pipeline skips completed stages.

Sub-steps:
1. **Prepare combined IR** — Merges each (original, optimized) pair into a single module with `_opt`-suffixed function names, using brace-counting IR parsing.
2. **Generate harnesses** — Asks a frontier LLM to produce `fuzz.cc` for each combined IR.
3. **Build binaries** — `llc` + `clang++ -fsanitize=fuzzer,address,undefined`.
4. **Run fuzzing** — Executes each binary with configurable runs/timeout. Supports parallel execution.

### `perf_test` — Performance benchmarking

Measures speedup of optimized IR vs. original. Requires diff testing to pass first (runs it automatically if needed).

```bash
cd src
python main.py --mode perf_test \
    --input /path/to/original_dir:/path/to/optimized_dir \
    --output /path/to/output_dir
```

Sub-steps:
1. **Collect corpus** — Runs diff-test fuzzing binaries briefly to generate diverse inputs (skipped if `corpus_dir` is set in config).
2. **Generate bench harnesses** — Transforms `fuzz.cc` → `bench.cc` with `clock_gettime` timing instrumentation around each `(base, opt)` call pair.
3. **Build bench binaries** — `llc` + `clang++` without sanitizers for accurate timing.
4. **Run benchmarks** — Executes each binary on every corpus file, records per-corpus metrics, and averages them.

Outputs:
- `perf_report.csv` — One row per test case (averaged speedup).
- `perf_report_detail.csv` — One row per (test case, corpus file).

### intrinsic advisor

#### 创建knowledge base
* 只处理 1 个 intrinsic，调试用
python scripts/build_intrinsic_kb.py --config config/config.yaml --limit 1 --archs x86,generic

* 处理 5 个
python scripts/build_intrinsic_kb.py --config config/config.yaml --limit 5

* 全量
python scripts/build_intrinsic_kb.py --config config/config.yaml

### ask llm
* 直接问
python scripts/ask_llm.py --config config/config.yaml -p "What does llvm.fma do?"

* 把 prompt 文件丢进去
python scripts/ask_llm.py --config config/config.yaml --file test/gemm/temp/step3_refinement/gemm_int8.prompt.ll

* 交互模式（输入完按 Ctrl+D）
python scripts/ask_llm.py --config config/config.yaml

* 换模型
python scripts/ask_llm.py --config config/config.yaml --model deepseek-v3.2 -p "hello"


## Configuration

All settings live in `config/config.yaml`:

| Section | Key fields |
|---|---|
| `model` | `model_path`, `adapter_path` — fine-tuned model for strategy generation |
| `llvm` | `passregistry_def`, `llvm_lib_root`, `opt_bin`, `ll_dir` |
| `llm` | `base_url`, `api_key`, `llm_model`, `api_mode`, `workers`, `max_output_tokens` |
| `run` | `batch_size`, `gpus`, `timeout`, `verify_timeout` |
| `diff_testing` | `llc`, `clangxx`, `fuzz_runs`, `fuzz_timeout`, `build_workers`, `fuzz_workers` |
| `perf_testing` | `corpus_dir`, `corpus_time`, `bench_iters`, `bench_timeout`, `bench_workers` |

## Dependencies

- Python 3.8+
- PyTorch, Transformers, PEFT
- scikit-learn
- OpenAI SDK
- LLVM toolchain (`opt`, `llc`, `clang++`)
- libFuzzer (for diff testing)
