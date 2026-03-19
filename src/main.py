"""LLVM IR optimisation pipeline — orchestrates strategy generation,
mapping, refinement, LLM-based rewriting, and post-processing."""

import argparse
import os
import shutil
import yaml
from pathlib import Path

from modules.strategy_generator import StrategyGenerator
from modules.strategy_mapping import StrategyMapping
from modules.strategy_refinement import StrategyRefinement
from modules.llm_client import LLMClient
from modules.post_processing import PostProcessor
from modules.diff_testing import DiffTester
from modules.perf_testing import PerfTester
from modules.utils import extract_single_block, log


# ======================================================================
# Configuration
# ======================================================================

class Config:
    """Merge defaults ← YAML file ← CLI overrides into a flat namespace."""

    _DEFAULTS = {
        "model_path": "",
        "adapter_path": "",
        "passregistry_def": "",
        "llvm_lib_root": "",
        "opt_bin": "",
        "ll_dir": "", #IR dir
        "base_url": "",
        "api_key": "",
        "batch_size": 4,
        "gpus": "0,1,2",
    }

    _YAML_SECTIONS = ("model", "llvm", "llm", "run", "diff_testing", "perf_testing")

    def __init__(self, config_file=None, **kwargs):
        cfg = dict(self._DEFAULTS)
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                y = yaml.safe_load(f) or {}
            for sec in self._YAML_SECTIONS:
                if sec in y:
                    cfg.update(y[sec])
        cfg.update(kwargs)
        for k, v in cfg.items():
            setattr(self, k, v)


# ======================================================================
# Pipeline orchestrator
# ======================================================================

class IROptimizer:
    def __init__(self, config: Config):
        self.config = config
        self.strategy_gen = StrategyGenerator(
            model_path=config.model_path,
            adapter_path=config.adapter_path,
        )
        self.strategy_map = StrategyMapping(
            passregistry_def=config.passregistry_def,
            llvm_lib_root=config.llvm_lib_root,
            opt_bin=config.opt_bin,
        )
        self.strategy_refine = StrategyRefinement(opt_bin=config.opt_bin)
        self.llm = LLMClient(base_url=config.base_url, api_key=config.api_key)
        self.post_proc = PostProcessor()
        self.diff_tester = DiffTester(
            llm=self.llm,
            llc=getattr(config, "llc",
                        "/home/amax/yangz/Env/llvm-project/build/bin/llc"),
            clangxx=getattr(config, "clangxx",
                            "/home/amax/yangz/Env/llvm-project/build/bin/clang++"),
        )
        self.perf_tester = PerfTester(
            llc=getattr(config, "llc",
                        "/home/amax/yangz/Env/llvm-project/build/bin/llc"),
            clangxx=getattr(config, "clangxx",
                            "/home/amax/yangz/Env/llvm-project/build/bin/clang++"),
        )

    # ------------------------------------------------------------------
    # Prompt rewriting (inject refined advice into the realization prompt)
    # ------------------------------------------------------------------

    @staticmethod
    def _rewrite_prompts(refinement_dir, out_dir) -> str:
        """Replace old advice in prompts with the LLM-refined advice,
        and switch the footer to request only <code> output."""

        OLD_HEADER = ("You may refer to the following advice, but feel free "
                      "to adapt, extend, or deviate from it as you see fit.")
        NEW_HEADER = "You can refer to the following advice."
        OLD_FOOTER = ("Please output the final optimization advice wrapped in "
                      "<advice>...</advice> and the full optimized LLVM IR "
                      "wrapped in <code>...</code>.")
        NEW_FOOTER = ("Please output the full optimized LLVM IR wrapped in "
                      "<code>...</code>.")

        log("Rewriting prompts for final LLM pass ...")
        refinement_dir = Path(refinement_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for prompt_file in refinement_dir.glob("*.prompt.ll"):
            prefix = prompt_file.stem[: -len(".prompt")]
            pred_file = prompt_file.parent / f"{prefix}.model.predict.ll"
            if not pred_file.is_file():
                log(f"  WARN: no refined prediction for {prefix}, skipping")
                continue

            new_advice = extract_single_block(
                pred_file.read_text(encoding="utf-8"), "advice",
            )
            ctx = prompt_file.read_text(encoding="utf-8")
            old_advice = extract_single_block(ctx, "advice")

            if old_advice is None:
                # No advice block yet — inject one after </code>
                ctx = ctx.replace(
                    "</code>\n",
                    f"</code>\n\n{NEW_HEADER}\n<advice>\n{new_advice}\n</advice>\n",
                )
            else:
                ctx = ctx.replace(old_advice, new_advice or "")

            ctx = ctx.replace(OLD_HEADER, NEW_HEADER)
            ctx = ctx.replace(OLD_FOOTER, NEW_FOOTER)

            (out_dir / f"{prefix}.prompt.ll").write_text(ctx, encoding="utf-8")

        log(f"Rewritten prompts saved to: {out_dir}")
        return str(out_dir)

    # ------------------------------------------------------------------
    # Single-file mode
    # ------------------------------------------------------------------

    def optimize_single_file(self, input_file: str, output_dir: str):
        input_path = Path(input_file)
        if not input_path.exists():
            raise SystemExit(f"Input file not found: {input_file}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        temp = output_path / "temp"

        # Prepare a one-file dataset directory
        dataset_dir = temp / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_file, dataset_dir / input_path.name)

        cfg = self.config

        # Step 1 — generate initial optimisation strategies
        log("Step 1: Strategy generation")
        initial_dir = self.strategy_gen.generate(
            data_path=str(dataset_dir),
            out_dir=str(temp / "step1_initial"),
            batch_size=1,
            gpus=cfg.gpus,
        )

        # Step 2 — map strategies to LLVM analysis passes
        log("Step 2: Strategy mapping")
        mapping_dir = self.strategy_map.map_strategies(
            in_dir=str(initial_dir),
            out_dir=str(temp / "step2_mapping"),
            topk=3,
            emit="tokens",
        )

        # Step 3 — refine strategies (add analysis info to prompts)
        log("Step 3: Strategy refinement")
        refine_dir = self.strategy_refine.refine(
            in_dir=mapping_dir,
            ll_dir=str(dataset_dir),
            initial_dir=str(initial_dir),
            out_dir=str(temp / "step3_refinement"),
            timeout=cfg.timeout,
            verify_timeout=cfg.verify_timeout,
        )

        # Step 3b — LLM-based strategy refinement
        refine_dir = self.llm.batch_query(
            in_dir=refine_dir,
            out_dir=refine_dir,
            model=cfg.llm_model,
            api_mode=cfg.api_mode,
            workers=1,
        )

        # Step 4 — rewrite prompts and call LLM for final IR
        log("Step 4: LLM realization")
        realize_dir = str(temp / "step4_realization")
        realize_dir = self._rewrite_prompts(refine_dir, realize_dir)
        realize_dir = self.llm.batch_query(
            in_dir=realize_dir,
            out_dir=realize_dir,
            model=cfg.llm_model,
            api_mode=cfg.api_mode,
            workers=1,
        )

        # Step 5 — post-process generated IR
        log("Step 5: Post-processing")
        realize_dir = self.post_proc.run(in_dir=realize_dir)

        # Extract final optimised .ll
        output_file = output_path / f"{input_path.stem}.optimized.ll"
        pred_files = list(Path(realize_dir).glob("*.model.predict.ll"))
        if pred_files:
            code = extract_single_block(
                pred_files[0].read_text(encoding="utf-8"), "code",
            )
            if code:
                output_file.write_text(code, encoding="utf-8")
                log(f"Output: {output_file}")
                return str(output_path)

        log(f"WARN: no optimised IR produced for {input_path.name}")
        return None

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------

    def optimize_batch(self, data_path: str, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        n_files = sum(1 for _ in Path(data_path).glob("*.ll"))

        # Step 1
        log("Step 1: Strategy generation")
        initial_dir = self.strategy_gen.generate(
            data_path=data_path,
            out_dir=os.path.join(output_dir, "step1_initial"),
            batch_size=min(cfg.batch_size, n_files),
            gpus=cfg.gpus,
        )

        # Step 2
        log("Step 2: Strategy mapping")
        mapping_dir = self.strategy_map.map_strategies(
            in_dir=str(initial_dir),
            out_dir=os.path.join(output_dir, "step2_mapping"),
            topk=3,
            emit="tokens",
        )

        # Step 3
        log("Step 3: Strategy refinement")
        refine_dir = self.strategy_refine.refine(
            in_dir=mapping_dir,
            ll_dir=data_path,
            initial_dir=str(initial_dir),
            out_dir=os.path.join(output_dir, "step3_refinement"),
            timeout=cfg.timeout,
            verify_timeout=cfg.verify_timeout,
        )

        # Step 3b — LLM-based strategy refinement
        refine_dir = self.llm.batch_query(
            in_dir=refine_dir,
            out_dir=refine_dir,
            model=cfg.llm_model,
            api_mode=cfg.api_mode,
            workers=min(cfg.workers, n_files),
        )

        # Step 4
        log("Step 4: LLM realization")
        realize_dir = os.path.join(output_dir, "step4_realization")
        realize_dir = self._rewrite_prompts(refine_dir, realize_dir)
        realize_dir = self.llm.batch_query(
            in_dir=realize_dir,
            out_dir=realize_dir,
            model=cfg.llm_model,
            api_mode=cfg.api_mode,
            workers=min(cfg.workers, n_files),
        )

        # Step 5
        log("Step 5: Post-processing")
        realize_dir = self.post_proc.run(in_dir=realize_dir)

        # Extract final .ll files
        pred_files = list(Path(realize_dir).glob("*.model.predict.ll"))
        if pred_files:
            for pf in pred_files:
                code = extract_single_block(pf.read_text(encoding="utf-8"), "code")
                if code:
                    stem = pf.name.replace(".model.predict.ll", "")
                    (output_path / f"{stem}.optimized.ll").write_text(
                        code, encoding="utf-8",
                    )
            log(f"Output directory: {output_dir}")
            return output_dir

        log(f"WARN: no optimised IR produced for {data_path}")
        return None

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def run(self, input_path: str, output_dir: str, mode: str):
        if mode == "single":
            return self.optimize_single_file(input_path, output_dir)
        elif mode == "batch":
            return self.optimize_batch(input_path, output_dir)
        elif mode == "diff_test":
            return self.diff_test(input_path, output_dir)
        elif mode == "perf_test":
            return self.perf_test(input_path, output_dir)
        else:
            raise SystemExit(f"Invalid mode: {mode}")

    # ------------------------------------------------------------------
    # Diff testing mode
    # ------------------------------------------------------------------

    def diff_test(self, input_path: str, output_dir: str):
        """Run differential testing on optimised IR.

        *input_path* can be:
          - A directory already containing a 'bins/' subfolder → skip straight to fuzzing.
          - A directory containing a 'harness/' subfolder → build + fuzz.
          - A directory with *.optimized.ll (and a sibling original dir) → full pipeline.
          - Two colon-separated paths  original_dir:optimized_dir  → full pipeline.
        """
        cfg = self.config
        work_dir = Path(output_dir) / "diff_test"
        work_dir.mkdir(parents=True, exist_ok=True)

        fuzz_runs = getattr(cfg, "fuzz_runs", 200000)
        fuzz_timeout = getattr(cfg, "fuzz_timeout", 600)
        build_workers = getattr(cfg, "build_workers", 16)
        fuzz_workers = getattr(cfg, "fuzz_workers", 1)

        # Check if bins already exist → just fuzz
        bin_dir = work_dir / "bins"
        if bin_dir.is_dir() and any(bin_dir.rglob("*_fuzz")):
            log("Found existing fuzzing binaries, running fuzzing directly")
            results = self.diff_tester.run_fuzzing(
                str(bin_dir), fuzz_runs=fuzz_runs, fuzz_timeout=fuzz_timeout,
                workers=fuzz_workers,
            )
            self._write_fuzz_report(results, work_dir)
            return str(work_dir)

        # Check if harnesses already exist → build + fuzz
        harness_dir = work_dir / "harness"
        combined_dir = work_dir / "combined"
        if harness_dir.is_dir() and any(harness_dir.glob("*.fuzz.cc")):
            log("Found existing harnesses, building and fuzzing")
            if not combined_dir.is_dir():
                raise SystemExit(
                    "harness/ exists but combined/ is missing — "
                    "cannot build without combined IR"
                )
            self.diff_tester.build_binaries(
                str(combined_dir), str(harness_dir), str(bin_dir),
                workers=build_workers,
            )
            results = self.diff_tester.run_fuzzing(
                str(bin_dir), fuzz_runs=fuzz_runs, fuzz_timeout=fuzz_timeout,
                workers=fuzz_workers,
            )
            self._write_fuzz_report(results, work_dir)
            return str(work_dir)

        # Full pipeline: need original_dir and optimized_dir
        if ":" in input_path:
            original_dir, optimized_dir = input_path.split(":", 1)
        else:
            # Assume input_path is the output_dir from a previous optimize run
            # containing *.optimized.ll, and look for an 'input' or 'll_dir'
            optimized_dir = input_path
            original_dir = getattr(cfg, "ll_dir", "")
            if not original_dir or not Path(original_dir).is_dir():
                raise SystemExit(
                    "Cannot determine original IR directory. Either pass "
                    "original_dir:optimized_dir or set ll_dir in config."
                )

        log("Running full diff-testing pipeline")
        results = self.diff_tester.run_full(
            original_dir=original_dir,
            optimized_dir=optimized_dir,
            work_dir=str(work_dir),
            model=getattr(cfg, "llm_model", "gpt-5"),
            api_mode=getattr(cfg, "api_mode", "auto"),
            workers=getattr(cfg, "workers", 50),
            build_workers=build_workers,
            fuzz_runs=fuzz_runs,
            fuzz_timeout=fuzz_timeout,
            fuzz_workers=fuzz_workers,
        )
        self._write_fuzz_report(results, work_dir)
        return str(work_dir)

    @staticmethod
    def _write_fuzz_report(results: dict, work_dir: Path) -> None:
        """Write a simple CSV summary of fuzzing results."""
        import csv
        report = work_dir / "fuzz_report.csv"
        with report.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "status", "exit_code"])
            w.writeheader()
            for stem, info in sorted(results.items()):
                w.writerow({
                    "file": stem,
                    "status": info["status"],
                    "exit_code": info["exit_code"],
                })
        log(f"Fuzzing report: {report}")

    # ------------------------------------------------------------------
    # Performance testing mode
    # ------------------------------------------------------------------

    def perf_test(self, input_path: str, output_dir: str):
        """Run performance benchmarking on optimised IR.

        Requires diff testing to have passed first.  If diff_test results
        already exist under *output_dir*/diff_test/, reuses them; otherwise
        runs the full diff_test pipeline first.

        *input_path* follows the same convention as diff_test().
        """
        cfg = self.config
        work_dir = Path(output_dir)
        dt_dir = work_dir / "diff_test"
        perf_dir = work_dir / "perf_test"
        perf_dir.mkdir(parents=True, exist_ok=True)

        # Ensure diff testing has been done
        fuzz_bin_dir = dt_dir / "bins"
        harness_dir = dt_dir / "harness"
        combined_dir = dt_dir / "combined"

        if not (fuzz_bin_dir.is_dir() and any(fuzz_bin_dir.rglob("*_fuzz"))):
            log("No diff-test results found, running diff_test first ...")
            self.diff_test(input_path, output_dir)

        if not fuzz_bin_dir.is_dir() or not any(fuzz_bin_dir.rglob("*_fuzz")):
            raise SystemExit("Diff testing did not produce any fuzzing binaries")

        # Check diff-test results — only benchmark cases that passed
        fuzz_report = dt_dir / "fuzz_report.csv"
        passed_stems = None
        if fuzz_report.exists():
            import csv
            with fuzz_report.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                passed_stems = {
                    row["file"] for row in reader if row["status"] == "PASS"
                }
            log(f"Diff-test: {len(passed_stems)} cases passed, "
                f"benchmarking only those")

        # Run perf pipeline
        build_workers = getattr(cfg, "build_workers", 16)
        corpus_time = getattr(cfg, "corpus_time", 10)
        corpus_workers = getattr(cfg, "corpus_workers", 0)
        bench_iters = getattr(cfg, "bench_iters", 1000000)
        bench_timeout = getattr(cfg, "bench_timeout", 300)
        bench_workers = getattr(cfg, "bench_workers", 1)
        corpus_dir = getattr(cfg, "corpus_dir", "")

        results = self.perf_tester.run_full(
            combined_dir=str(combined_dir),
            harness_dir=str(harness_dir),
            fuzz_bin_dir=str(fuzz_bin_dir),
            work_dir=str(perf_dir),
            corpus_dir=corpus_dir,
            corpus_time=corpus_time,
            corpus_workers=corpus_workers,
            build_workers=build_workers,
            bench_iters=bench_iters,
            bench_timeout=bench_timeout,
            bench_workers=bench_workers,
        )

        # Filter to only diff-test-passed cases if we have the report
        if passed_stems is not None:
            results = {
                k: v for k, v in results.items() if k in passed_stems
            }

        self._write_perf_report(results, perf_dir)
        return str(perf_dir)

    @staticmethod
    def _write_perf_report(results: dict, work_dir: Path) -> None:
        """Write CSV summaries of performance results.

        Produces two files:
          - perf_report.csv       — one row per stem (averaged over corpus)
          - perf_report_detail.csv — one row per (stem, corpus_file)
        """
        import csv

        # Summary report (averaged)
        report = work_dir / "perf_report.csv"
        fields = ["file", "status", "n_corpus", "baseline_ns", "opt_ns", "speedup"]
        with report.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for stem, info in sorted(results.items()):
                w.writerow({
                    "file": stem,
                    "status": info.get("status", ""),
                    "n_corpus": info.get("n_corpus", 0),
                    "baseline_ns": f"{info.get('baseline_ns', 0):.2f}",
                    "opt_ns": f"{info.get('opt_ns', 0):.2f}",
                    "speedup": f"{info.get('speedup', 0):.4f}",
                })
        log(f"Performance report: {report}")

        # Detail report (per corpus file)
        detail = work_dir / "perf_report_detail.csv"
        detail_fields = [
            "file", "corpus", "status", "baseline_ns", "opt_ns", "speedup",
        ]
        with detail.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=detail_fields)
            w.writeheader()
            for stem, info in sorted(results.items()):
                for entry in info.get("per_corpus", []):
                    w.writerow({
                        "file": stem,
                        "corpus": entry.get("corpus", ""),
                        "status": entry.get("status", ""),
                        "baseline_ns": f"{entry.get('baseline_ns', 0):.2f}",
                        "opt_ns": f"{entry.get('opt_ns', 0):.2f}",
                        "speedup": f"{entry.get('speedup', 0):.4f}",
                    })
        log(f"Performance detail: {detail}")


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="LLVM IR Optimizer")
    p.add_argument("--mode", choices=["single", "batch", "diff_test", "perf_test"], required=True)
    p.add_argument("--input", required=True,
                   help="Input .ll file (single), directory (batch), "
                        "or original_dir:optimized_dir (diff_test)")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--config", default="../config/config.yaml")

    p.add_argument("--model_path")
    p.add_argument("--adapter_path")
    p.add_argument("--passregistry_def")
    p.add_argument("--llvm_lib_root")
    p.add_argument("--opt_bin")
    p.add_argument("--ll_dir")
    p.add_argument("--base_url")
    p.add_argument("--api_key")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--gpus")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k not in ("mode", "input", "output", "config")
    }
    config = Config(config_file=args.config, **overrides)
    optimizer = IROptimizer(config)
    result = optimizer.run(args.input, args.output, args.mode)
    if result:
        log(f"All done.  Result: {result}")


if __name__ == "__main__":
    main()
