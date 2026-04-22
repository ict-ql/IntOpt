"""LLVM IR optimisation pipeline — orchestrates strategy generation,
mapping, refinement, LLM-based rewriting, and post-processing."""

import argparse
import os
import re
import shutil
from typing import List
import yaml
from pathlib import Path

from modules.strategy_generator import StrategyGenerator
from modules.strategy_mapping import StrategyMapping
from modules.strategy_refinement import StrategyRefinement
from modules.llm_client import LLMClient
from modules.post_processing import PostProcessor
from modules.verification import Verifier
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

    _YAML_SECTIONS = ("model", "llvm", "llm", "run", "diff_testing", "perf_testing", "alive2", "intrinsic_advisor")

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
        self.diff_tester = Verifier(
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
        self._intrinsic_advisor = None
        self._host_cpu = ""
        self._host_features_str = ""

    def _get_intrinsic_advice(
        self, ir_text: str,
        cache_dir: str = "", cache_key: str = "",
    ) -> str:
        """Retrieve intrinsic suggestions if enabled in config.

        *cache_dir*: directory to cache IR summaries.
        *cache_key*: filename stem for the cache file."""
        cfg = self.config
        if not getattr(cfg, "intrinsic_enabled", False):
            log("Intrinsic advisor: disabled (intrinsic_enabled=false)")
            return ""
        kb_path = getattr(cfg, "intrinsic_kb_path", "")
        emb_model = getattr(cfg, "intrinsic_embedding_model", "")
        if not kb_path:
            log("Intrinsic advisor: skipped (intrinsic_kb_path not set)")
            return ""
        if not Path(kb_path).exists():
            log(f"Intrinsic advisor: skipped (kb_path not found: {kb_path})")
            return ""
        if not emb_model:
            log("Intrinsic advisor: skipped (intrinsic_embedding_model not set)")
            return ""
        try:
            if self._intrinsic_advisor is None:
                from modules.intrinsic_advisor import IntrinsicAdvisor
                host_features = self._detect_host_features()
                self._intrinsic_advisor = IntrinsicAdvisor(
                    kb_path, emb_model, host_features=host_features,
                    declares_path=getattr(cfg, "intrinsic_declares_path", ""),
                )
            top_k = getattr(cfg, "intrinsic_top_k", 10)
            boost_thresh = getattr(cfg, "intrinsic_relevance_threshold", 0.45)
            suggestions = self._intrinsic_advisor.suggest(
                ir_text, self.llm,
                model=getattr(cfg, "llm_model", "gpt-5"),
                api_mode=getattr(cfg, "api_mode", "auto"),
                top_k=top_k,
                cache_dir=cache_dir,
                cache_key=cache_key,
                boost_threshold=boost_thresh,
            )
            result = self._intrinsic_advisor.format_suggestions(suggestions)
            if result:
                log(f"Intrinsic advisor: {len(suggestions)} suggestions retrieved")
            else:
                log("Intrinsic advisor: no suggestions matched")
            return result
        except Exception as e:
            log(f"WARN: intrinsic advisor failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _detect_host_features(self) -> set:
        """Detect host CPU features using clang -march=native."""
        cfg = self.config
        clang = getattr(cfg, "clangxx", "").replace("clang++", "clang")
        if not clang:
            # Try to derive from llc path
            llc = getattr(cfg, "llc", "")
            if llc:
                clang = llc.replace("llc", "clang")
        if not clang or not Path(clang).exists():
            log("WARN: clang not found, cannot detect host features")
            return set()
        try:
            from modules.intrinsic_advisor import detect_host_features
            march = getattr(cfg, "target_march", "native")
            cpu, features_str, features = detect_host_features(clang, march=march)
            if cpu:
                log(f"Host CPU: {cpu}, features: {len(features)} flags")
                # Cache for later use (attribute injection)
                self._host_cpu = cpu
                self._host_features_str = features_str
            return features
        except Exception as e:
            log(f"WARN: host feature detection failed: {e}")
            return set()

    def inject_target_attributes(self, ir_text: str) -> str:
        """Inject host CPU target-cpu and target-features into IR attributes.

        Finds `attributes #N = { ... }` lines and adds/replaces
        target-cpu and target-features if not already present."""
        if not self._host_cpu and not self._host_features_str:
            # Try to detect if not done yet
            self._detect_host_features()
        if not self._host_cpu:
            return ir_text

        from modules.intrinsic_advisor import features_to_attribute_string
        attr_str = features_to_attribute_string(self._host_cpu, self._host_features_str)
        if not attr_str:
            return ir_text

        lines = ir_text.splitlines()
        result = []
        for line in lines:
            if line.startswith("attributes #") and "{" in line:
                # Check if target-cpu already present
                if '"target-cpu"' not in line:
                    # Insert before the closing }
                    line = line.rstrip()
                    if line.endswith("}"):
                        line = line[:-1].rstrip() + " " + attr_str + " }"
            result.append(line)

        return "\n".join(result)

    # ------------------------------------------------------------------
    # Prompt rewriting (inject refined advice into the realization prompt)
    # ------------------------------------------------------------------

    def _rewrite_prompts(self, refinement_dir, out_dir) -> str:
        """Replace old advice in prompts with the LLM-refined advice,
        and switch the footer to request only <code> output.

        Also: extract intrinsic names mentioned in the advice, look up
        their correct declare signatures, and append them to the prompt
        so the LLM generates correct IR."""

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
                ctx = ctx.replace(
                    "</code>\n",
                    f"</code>\n\n{NEW_HEADER}\n<advice>\n{new_advice}\n</advice>\n",
                )
            else:
                ctx = ctx.replace(old_advice, new_advice or "")

            ctx = ctx.replace(OLD_HEADER, NEW_HEADER)
            ctx = ctx.replace(OLD_FOOTER, NEW_FOOTER)

            # Remove intrinsics block and surrounding instruction text
            ctx = re.sub(
                r"IMPORTANT:.*?</intrinsics>\s*"
                r"(?:In your.*?\n)*\s*",
                "", ctx, flags=re.DOTALL,
            )
            ctx = re.sub(r"<intrinsics>.*?</intrinsics>\s*", "", ctx, flags=re.DOTALL)

            # Extract intrinsic names from the refined advice and attach
            # their correct declare signatures
            if new_advice and self._intrinsic_advisor is not None:
                intrinsic_sigs = self._extract_intrinsic_signatures(new_advice)
                if intrinsic_sigs:
                    sig_block = (
                        "\nThe following intrinsic signatures are referenced "
                        "in the advice above. Use these exact declarations "
                        "in your output IR:\n"
                        + "\n".join(intrinsic_sigs)
                        + "\n"
                    )
                    # Insert before the final instruction line
                    ctx = ctx.replace(NEW_FOOTER, sig_block + "\n" + NEW_FOOTER)
                    log(f"  {prefix}: attached {len(intrinsic_sigs)} intrinsic signatures")

            (out_dir / f"{prefix}.prompt.ll").write_text(ctx, encoding="utf-8")

        log(f"Rewritten prompts saved to: {out_dir}")
        return str(out_dir)

    def _extract_intrinsic_signatures(self, text: str) -> List[str]:
        """Find @llvm.* intrinsic names in text and return their declare lines.

        If an intrinsic is OVERLOADED (valid but type-parameterized), skip
        the signature (the LLM should infer the correct types from context).
        If not found at all, mark as invalid."""
        if self._intrinsic_advisor is None:
            return []
        names = set(re.findall(r"@(llvm\.[a-zA-Z0-9_.]+)", text))
        sigs = []
        seen = set()
        for name in sorted(names):
            decl = self._intrinsic_advisor._get_declare(name)
            if decl == "OVERLOADED":
                # Valid intrinsic, just overloaded — no fixed signature to attach
                continue
            elif decl and decl not in seen:
                sigs.append(decl)
                seen.add(decl)
            elif not decl:
                warning = (
                    f"; WARNING: @{name} does NOT exist in LLVM. "
                    f"Do NOT use this intrinsic — it will cause a compilation error."
                )
                if warning not in seen:
                    sigs.append(warning)
                    seen.add(warning)
                    log(f"  WARN: intrinsic @{name} not found in declares DB (hallucination?)")
        return sigs

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

        # Inject target attributes into input IR before any processing
        if getattr(cfg, "intrinsic_enabled", False) or getattr(cfg, "inject_target_attrs", False):
            input_ll = dataset_dir / input_path.name
            ir_text = input_ll.read_text(encoding="utf-8")
            ir_text = self.inject_target_attributes(ir_text)
            input_ll.write_text(ir_text, encoding="utf-8")
            log(f"Injected target attributes into {input_ll.name}")

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
        ir_text = (dataset_dir / input_path.name).read_text(encoding="utf-8")
        summary_cache_dir = str(temp / "IR_summaries")
        intrinsic_text = self._get_intrinsic_advice(
            ir_text,
            cache_dir=summary_cache_dir,
            cache_key=input_path.stem,
        )
        if intrinsic_text:
            intrinsic_file = temp / "step3_refinement" / f"{input_path.stem}.intrinsic_suggestions.txt"
            intrinsic_file.parent.mkdir(parents=True, exist_ok=True)
            intrinsic_file.write_text(intrinsic_text, encoding="utf-8")
            log(f"Intrinsic suggestions saved to: {intrinsic_file}")
        intrinsic_map = {input_path.stem: intrinsic_text} if intrinsic_text else None
        refine_dir = self.strategy_refine.refine(
            in_dir=mapping_dir,
            ll_dir=str(dataset_dir),
            initial_dir=str(initial_dir),
            out_dir=str(temp / "step3_refinement"),
            timeout=cfg.timeout,
            verify_timeout=cfg.verify_timeout,
            intrinsic_advice_map=intrinsic_map,
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

        # Inject target attributes into all input IR files
        if getattr(cfg, "intrinsic_enabled", False) or getattr(cfg, "inject_target_attrs", False):
            for ll_file in Path(data_path).glob("*.ll"):
                ir_text = ll_file.read_text(encoding="utf-8")
                ir_text = self.inject_target_attributes(ir_text)
                ll_file.write_text(ir_text, encoding="utf-8")
            log(f"Injected target attributes into {n_files} input files")

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
        # Generate per-file intrinsic advice (parallel + cached)
        intrinsic_advice_map = {}
        if getattr(cfg, "intrinsic_enabled", False):
            try:
                if self._intrinsic_advisor is None:
                    from modules.intrinsic_advisor import IntrinsicAdvisor
                    host_features = self._detect_host_features()
                    kb_path = getattr(cfg, "intrinsic_kb_path", "")
                    emb_model = getattr(cfg, "intrinsic_embedding_model", "")
                    self._intrinsic_advisor = IntrinsicAdvisor(
                        kb_path, emb_model, host_features=host_features,
                        declares_path=getattr(cfg, "intrinsic_declares_path", ""),
                    )
                ir_items = []
                for ll_file in sorted(Path(data_path).glob("*.ll")):
                    ir_items.append({
                        "key": ll_file.stem,
                        "ir_text": ll_file.read_text(encoding="utf-8"),
                    })
                if ir_items:
                    summary_cache = os.path.join(output_dir, "IR_summaries")
                    batch_results = self._intrinsic_advisor.batch_suggest(
                        ir_items, self.llm,
                        model=getattr(cfg, "llm_model", "gpt-5"),
                        api_mode=getattr(cfg, "api_mode", "auto"),
                        top_k=getattr(cfg, "intrinsic_top_k", 10),
                        cache_dir=summary_cache,
                        workers=min(getattr(cfg, "workers", 50), len(ir_items)),
                        boost_threshold=getattr(cfg, "intrinsic_relevance_threshold", 0.45),
                    )
                    for key, suggestions in batch_results.items():
                        text = self._intrinsic_advisor.format_suggestions(suggestions)
                        if text:
                            intrinsic_advice_map[key] = text
                    log(f"Intrinsic advisor: generated advice for "
                        f"{len(intrinsic_advice_map)}/{len(ir_items)} files")
            except Exception as e:
                log(f"WARN: batch intrinsic advisor failed: {e}")

        refine_dir = self.strategy_refine.refine(
            in_dir=mapping_dir,
            ll_dir=data_path,
            initial_dir=str(initial_dir),
            out_dir=os.path.join(output_dir, "step3_refinement"),
            timeout=cfg.timeout,
            verify_timeout=cfg.verify_timeout,
            intrinsic_advice_map=intrinsic_advice_map if intrinsic_advice_map else None,
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

    def run(self, input_path: str, output_dir: str, mode: str, step: int = 0):
        if mode == "single":
            return self.optimize_single_file(input_path, output_dir)
        elif mode == "batch":
            return self.optimize_batch(input_path, output_dir)
        elif mode == "diff_test":
            return self.diff_test(input_path, output_dir)
        elif mode == "perf_test":
            return self.perf_test(input_path, output_dir)
        elif mode == "verify":
            return self.verify(input_path, output_dir)
        elif mode == "step":
            return self.run_step(input_path, output_dir, step)
        else:
            raise SystemExit(f"Invalid mode: {mode}")

    # ------------------------------------------------------------------
    # Step-by-step interactive mode (for VSCode extension)
    # ------------------------------------------------------------------

    def run_step(self, input_file: str, output_dir: str, step: int):
        """Run a single pipeline step.  The extension calls this repeatedly.

        Step 0: prepare (copy input to dataset dir)
        Step 1: strategy generation
        Step 2: strategy mapping
        Step 3: strategy refinement (analysis + LLM refinement)
        Step 4: LLM realization
        Step 5: post-processing + extract final IR
        """
        import json

        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        temp = output_path / "temp"
        cfg = self.config

        dataset_dir = temp / "dataset"

        if step == 0:
            # Prepare dataset directory
            dataset_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_file, dataset_dir / input_path.name)

            # Inject target attributes into input IR
            if getattr(cfg, "intrinsic_enabled", False) or getattr(cfg, "inject_target_attrs", False):
                input_ll = dataset_dir / input_path.name
                ir_text = input_ll.read_text(encoding="utf-8")
                ir_text = self.inject_target_attributes(ir_text)
                input_ll.write_text(ir_text, encoding="utf-8")
                log(f"Injected target attributes into {input_ll.name}")

            log("STEP_RESULT:" + json.dumps({
                "step": 0, "status": "ok",
                "message": f"Prepared input: {input_path.name}",
                "dataset_dir": str(dataset_dir),
            }))
            return str(dataset_dir)

        elif step == 1:
            # Strategy generation
            log("Step 1: Strategy generation")
            initial_dir = self.strategy_gen.generate(
                data_path=str(dataset_dir),
                out_dir=str(temp / "step1_initial"),
                batch_size=1,
                gpus=cfg.gpus,
            )
            # Read the generated strategy — extract <code> block content
            pred_files = list(Path(initial_dir).glob("*.model.predict.ll"))
            content = ""
            if pred_files:
                raw = pred_files[0].read_text(encoding="utf-8")
                content = extract_single_block(raw, "code") or raw
            log("STEP_RESULT:" + json.dumps({
                "step": 1, "status": "ok",
                "title": "Initial Optimization Strategy",
                "content": content,
                "file": str(pred_files[0]) if pred_files else "",
                "dir": str(initial_dir),
            }))
            return str(initial_dir)

        elif step == 2:
            # Strategy mapping — produces prompt with <analysis> block
            initial_dir = str(temp / "step1_initial")
            log("Step 2: Strategy mapping")
            mapping_dir = self.strategy_map.map_strategies(
                in_dir=initial_dir,
                out_dir=str(temp / "step2_mapping"),
                topk=3,
                emit="tokens",
            )
            # Read the mapping/analysis result (the prompt file has analysis)
            prompt_files = list(Path(mapping_dir).glob("*.prompt.ll"))
            content = ""
            analysis = ""
            if prompt_files:
                raw = prompt_files[0].read_text(encoding="utf-8")
                content = raw
                analysis = extract_single_block(raw, "analysis") or ""
            log("STEP_RESULT:" + json.dumps({
                "step": 2, "status": "ok",
                "title": "Strategy Mapping & Analysis",
                "content": content,
                "analysis": analysis,
                "file": str(prompt_files[0]) if prompt_files else "",
                "dir": str(mapping_dir),
            }))
            return str(mapping_dir)

        elif step == 3:
            # Step 3a: Strategy refinement — build prompt with analysis
            # Optionally include intrinsic suggestions from KB
            mapping_dir = str(temp / "step2_mapping")
            initial_dir = str(temp / "step1_initial")

            # Intrinsic advisor: retrieve relevant intrinsics if enabled
            intrinsic_text = ""
            if getattr(cfg, "intrinsic_enabled", False):
                kb_path = getattr(cfg, "intrinsic_kb_path", "")
                emb_model = getattr(cfg, "intrinsic_embedding_model", "")
                if kb_path and Path(kb_path).exists() and emb_model:
                    try:
                        from modules.intrinsic_advisor import IntrinsicAdvisor
                        host_features = self._detect_host_features()
                        advisor = IntrinsicAdvisor(
                            kb_path, emb_model, host_features=host_features,
                            declares_path=getattr(cfg, "intrinsic_declares_path", ""),
                        )
                        ir_file = list(Path(str(dataset_dir)).glob("*.ll"))
                        if ir_file:
                            ir_text = ir_file[0].read_text(encoding="utf-8")
                            summary_cache = str(temp / "IR_summaries")
                            suggestions = advisor.suggest(
                                ir_text, self.llm,
                                model=getattr(cfg, "llm_model", "gpt-5"),
                                api_mode=getattr(cfg, "api_mode", "auto"),
                                top_k=getattr(cfg, "intrinsic_top_k", 10),
                                cache_dir=summary_cache,
                                cache_key=ir_file[0].stem,
                                boost_threshold=getattr(cfg, "intrinsic_relevance_threshold", 0.45),
                            )
                            intrinsic_text = advisor.format_suggestions(suggestions)
                            if intrinsic_text:
                                log(f"  Retrieved {len(suggestions)} intrinsic suggestions")
                    except Exception as e:
                        log(f"  WARN: intrinsic advisor failed: {e}")

            log("Step 3: Strategy refinement (building prompt)")
            intrinsic_map = {input_path.stem: intrinsic_text} if intrinsic_text else None
            refine_dir = self.strategy_refine.refine(
                in_dir=mapping_dir,
                ll_dir=str(dataset_dir),
                initial_dir=initial_dir,
                out_dir=str(temp / "step3_refinement"),
                timeout=cfg.timeout,
                verify_timeout=cfg.verify_timeout,
                intrinsic_advice_map=intrinsic_map,
            )

            # Output analysis from prompt for user review BEFORE LLM call
            prompt_files = list(Path(refine_dir).glob("*.prompt.ll"))
            analysis = ""
            intrinsics_block = ""
            prompt_file = ""
            if prompt_files:
                prompt_file = str(prompt_files[0])
                raw = prompt_files[0].read_text(encoding="utf-8")
                analysis = extract_single_block(raw, "analysis") or ""
                intrinsics_block = extract_single_block(raw, "intrinsics") or ""

            # Combine analysis + intrinsics for display
            display_content = analysis
            if intrinsics_block:
                display_content += "\n\n--- Intrinsic Suggestions ---\n" + intrinsics_block

            log("STEP_RESULT:" + json.dumps({
                "step": 3, "status": "ok",
                "title": "Analysis & Intrinsic Suggestions (editable before LLM refinement)",
                "content_type": "ir",
                "content": display_content,
                "file": prompt_file,
                "dir": str(refine_dir),
            }))
            return str(refine_dir)

        elif step == 4:
            # Step 3b: LLM-based strategy refinement (uses the prompt user may have edited)
            refine_dir = str(temp / "step3_refinement")
            log("Step 3b: LLM strategy refinement")
            refine_dir = self.llm.batch_query(
                in_dir=refine_dir,
                out_dir=refine_dir,
                model=cfg.llm_model,
                api_mode=cfg.api_mode,
                workers=1,
            )
            # Show only <advice> block from refined strategy
            pred_files = list(Path(refine_dir).glob("*.model.predict.ll"))
            advice = ""
            if pred_files:
                raw = pred_files[0].read_text(encoding="utf-8")
                advice = extract_single_block(raw, "advice") or ""
            log("STEP_RESULT:" + json.dumps({
                "step": 4, "status": "ok",
                "title": "Refined Optimization Strategy (Advice)",
                "content_type": "markdown",
                "content": advice,
                "file": str(pred_files[0]) if pred_files else "",
                "dir": str(refine_dir),
            }))
            return str(refine_dir)

        elif step == 5:
            # LLM realization
            refine_dir = str(temp / "step3_refinement")
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
            # Show only <code> block from generated IR
            pred_files = list(Path(realize_dir).glob("*.model.predict.ll"))
            code = ""
            if pred_files:
                raw = pred_files[0].read_text(encoding="utf-8")
                code = extract_single_block(raw, "code") or raw
            log("STEP_RESULT:" + json.dumps({
                "step": 5, "status": "ok",
                "title": "LLM-Generated Optimized IR",
                "content_type": "ir",
                "content": code,
                "file": str(pred_files[0]) if pred_files else "",
                "dir": str(realize_dir),
            }))
            return str(realize_dir)

        elif step == 6:
            # Post-processing + extract
            realize_dir = str(temp / "step4_realization")
            log("Step 5: Post-processing")
            realize_dir = self.post_proc.run(in_dir=realize_dir)

            output_file = output_path / f"{input_path.stem}.optimized.ll"
            pred_files = list(Path(realize_dir).glob("*.model.predict.ll"))
            code = ""
            if pred_files:
                code = extract_single_block(
                    pred_files[0].read_text(encoding="utf-8"), "code",
                ) or ""
                if code:
                    output_file.write_text(code, encoding="utf-8")

            log("STEP_RESULT:" + json.dumps({
                "step": 5, "status": "ok" if code else "fail",
                "title": "Final Optimized IR",
                "content_type": "ir",
                "content": code,
                "output_file": str(output_file) if code else "",
            }))
            return str(output_file) if code else None

        elif step == 7:
            # Verify: alive2 first, fallback to diff fuzzing if needed
            log("Step 6: Verification")
            output_file = output_path / f"{input_path.stem}.optimized.ll"
            if not output_file.exists():
                log("STEP_RESULT:" + json.dumps({
                    "step": 7, "status": "fail",
                    "title": "Verification",
                    "content": "No optimized IR found to verify",
                }))
                return None

            # Prepare combined IR (rebuild each time to pick up edits)
            orig_file = dataset_dir / input_path.name
            verify_dir = temp / "verify"
            combined_dir = verify_dir / "combined"
            # Clean old verify artifacts for retry
            alive2_dir = verify_dir / "alive2"
            for d in [combined_dir, alive2_dir, verify_dir / "harness", verify_dir / "bins"]:
                if d.is_dir():
                    shutil.rmtree(d)
            combined_dir.mkdir(parents=True, exist_ok=True)
            from modules.verification import build_combined_ir
            orig_ir = orig_file.read_text(encoding="utf-8")
            opt_ir = output_file.read_text(encoding="utf-8")
            combined = build_combined_ir(orig_ir, opt_ir)
            stem = input_path.stem
            case_dir = combined_dir / stem
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "combined.ll").write_text(combined, encoding="utf-8")

            summary_parts = []

            # Phase 1: alive2
            alive2_dir = verify_dir / "alive2"
            alive2_bin = getattr(cfg, "alive2_bin",
                                 "/home/amax/yangz/Env/alive2/build/alive-tv")
            alive2_timeout = getattr(cfg, "alive2_timeout", 60)

            log("Running alive2 verification ...")
            alive2_results = self.diff_tester.run_alive2(
                str(combined_dir), str(alive2_dir),
                alive2_bin=alive2_bin,
                timeout=alive2_timeout,
                workers=1,
                strict=getattr(cfg, "alive2_strict", False),
            )

            a2_info = alive2_results.get(stem, {})
            a2_status = a2_info.get("status", "UNKNOWN")
            log_file = alive2_dir / stem / "alive2_log.txt"
            a2_log = ""
            if log_file.exists():
                a2_log = log_file.read_text(encoding="utf-8")[-2000:]

            summary_parts.append(f"[Alive2] {a2_status}")
            if a2_status == "PASS":
                summary_parts.append("Transformation verified correct by alive2!")
            else:
                summary_parts.append(f"Alive2 did not pass ({a2_status}), running diff fuzzing ...")

            final_status = a2_status

            # Phase 2: diff fuzzing fallback if alive2 didn't pass
            if a2_status != "PASS":
                log("Alive2 did not pass, falling back to diff fuzzing ...")
                harness_dir = str(verify_dir / "harness")
                harness_cfg = getattr(cfg, "harness_dir", "")

                # For single-file mode, check if config harness dir has
                # a matching harness; if not, generate one for this case only
                if harness_cfg and Path(harness_cfg).is_dir():
                    matching = Path(harness_cfg) / f"{stem}.fuzz.cc"
                    if matching.exists():
                        # Copy just the matching harness to local dir
                        Path(harness_dir).mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(matching), harness_dir)
                    else:
                        self.diff_tester.generate_harnesses(
                            str(combined_dir), harness_dir,
                            model=getattr(cfg, "llm_model", "gpt-5"),
                            api_mode=getattr(cfg, "api_mode", "auto"),
                            workers=1,
                        )
                else:
                    self.diff_tester.generate_harnesses(
                        str(combined_dir), harness_dir,
                        model=getattr(cfg, "llm_model", "gpt-5"),
                        api_mode=getattr(cfg, "api_mode", "auto"),
                        workers=1,
                    )

                bin_dir = str(verify_dir / "bins")
                self.diff_tester.build_binaries(
                    str(combined_dir), harness_dir, bin_dir,
                    workers=getattr(cfg, "build_workers", 16),
                )

                fuzz_results = self.diff_tester.run_fuzzing(
                    bin_dir,
                    fuzz_runs=getattr(cfg, "fuzz_runs", 200000),
                    fuzz_timeout=getattr(cfg, "fuzz_timeout", 600),
                    workers=1,
                )

                fuzz_info = fuzz_results.get(stem, {})
                fuzz_status = fuzz_info.get("status", "UNKNOWN")
                fuzz_tail = fuzz_info.get("output_tail", "")[-1000:]

                summary_parts.append(f"\n[Diff Fuzzing] {fuzz_status}")
                if fuzz_status == "PASS":
                    summary_parts.append("Diff fuzzing passed!")
                    final_status = "PASS"
                else:
                    summary_parts.append(f"Diff fuzzing: {fuzz_status}")
                    final_status = fuzz_status

                if fuzz_tail:
                    summary_parts.append(f"\n--- Fuzzing Log (tail) ---\n{fuzz_tail}")

            if a2_log:
                summary_parts.append(f"\n--- Alive2 Log ---\n{a2_log}")

            log("STEP_RESULT:" + json.dumps({
                "step": 7, "status": "ok",
                "title": f"Verification: {final_status}",
                "content": "\n".join(summary_parts),
                "verify_status": final_status,
                "output_file": str(output_file),
            }))
            return final_status

        elif step == 8:
            # Performance testing
            log("Step 7: Performance testing")
            output_file = output_path / f"{input_path.stem}.optimized.ll"
            if not output_file.exists():
                log("STEP_RESULT:" + json.dumps({
                    "step": 8, "status": "fail",
                    "title": "Performance Test",
                    "content": "No optimized IR found",
                }))
                return None

            stem = input_path.stem
            combined_dir = str(temp / "verify" / "combined")
            perf_dir = temp / "perf_test"
            perf_dir.mkdir(parents=True, exist_ok=True)

            # Need fuzz binary for corpus collection
            # Build harness + fuzz binary
            harness_dir = str(perf_dir / "harness")
            harness_cfg = getattr(cfg, "harness_dir", "")

            # For single-file mode, only use matching harness from config dir
            if harness_cfg and Path(harness_cfg).is_dir():
                matching = Path(harness_cfg) / f"{stem}.fuzz.cc"
                if matching.exists():
                    Path(harness_dir).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(matching), harness_dir)
                else:
                    self.diff_tester.generate_harnesses(
                        combined_dir, harness_dir,
                        model=getattr(cfg, "llm_model", "gpt-5"),
                        api_mode=getattr(cfg, "api_mode", "auto"),
                        workers=1,
                    )
            else:
                self.diff_tester.generate_harnesses(
                    combined_dir, harness_dir,
                    model=getattr(cfg, "llm_model", "gpt-5"),
                    api_mode=getattr(cfg, "api_mode", "auto"),
                    workers=1,
                )

            fuzz_bin_dir = str(perf_dir / "fuzz_bins")
            self.diff_tester.build_binaries(
                combined_dir, harness_dir, fuzz_bin_dir,
                workers=getattr(cfg, "build_workers", 16),
            )

            # Run perf — filter config dirs to matching stem for single-file mode
            perf_harness_cfg = getattr(cfg, "perf_harness_dir", "")
            corpus_cfg = getattr(cfg, "corpus_dir", "")

            # If perf_harness_dir has a matching bench harness, copy just that one
            local_perf_harness = ""
            if perf_harness_cfg and Path(perf_harness_cfg).is_dir():
                matching = Path(perf_harness_cfg) / f"{stem}.bench.cc"
                if matching.exists():
                    local_ph = perf_dir / "bench_harness_local"
                    local_ph.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(matching), str(local_ph))
                    local_perf_harness = str(local_ph)

            # If corpus_dir has a matching corpus subdir, use just that
            local_corpus = ""
            if corpus_cfg and Path(corpus_cfg).is_dir():
                matching_corpus = Path(corpus_cfg) / stem
                if matching_corpus.is_dir() and any(matching_corpus.iterdir()):
                    local_cd = perf_dir / "corpus_local"
                    local_cd.mkdir(parents=True, exist_ok=True)
                    dst = local_cd / stem
                    if not dst.exists():
                        shutil.copytree(str(matching_corpus), str(dst))
                    local_corpus = str(local_cd)

            results = self.perf_tester.run_full(
                combined_dir=combined_dir,
                harness_dir=harness_dir,
                fuzz_bin_dir=fuzz_bin_dir,
                work_dir=str(perf_dir),
                corpus_dir=local_corpus,
                perf_harness_dir=local_perf_harness,
                corpus_time=getattr(cfg, "corpus_time", 10),
                corpus_workers=getattr(cfg, "corpus_workers", 0),
                build_workers=getattr(cfg, "build_workers", 16),
                bench_iters=getattr(cfg, "bench_iters", 10000),
                bench_timeout=getattr(cfg, "bench_timeout", 60),
                bench_workers=1,
            )

            info = results.get(stem, {})
            status = info.get("status", "UNKNOWN")
            speedup = info.get("speedup", 1.0)
            n_corpus = info.get("n_corpus", 0)
            baseline_ns = info.get("baseline_ns", 0)
            opt_ns = info.get("opt_ns", 0)

            summary = f"Status: {status}\n"
            summary += f"Speedup: {speedup:.3f}x\n"
            summary += f"Corpus files: {n_corpus}\n"
            summary += f"Baseline: {baseline_ns:.2f} ns\n"
            summary += f"Optimized: {opt_ns:.2f} ns"

            log("STEP_RESULT:" + json.dumps({
                "step": 8, "status": "ok",
                "title": f"Performance: {speedup:.3f}x speedup",
                "content": summary,
            }))
            return str(perf_dir)

        elif step == 9:
            # Apply optimized IR back to original file + opt verify
            # Expects --input to be the ORIGINAL full .ll file path
            # (the extension passes the original source file)
            import subprocess as _sp

            log("Step 8: Apply optimized IR to original")
            stem = input_path.stem
            output_file = output_path / f"{stem}.optimized.ll"
            if not output_file.exists():
                log("STEP_RESULT:" + json.dumps({
                    "step": 9, "status": "fail",
                    "title": "Apply",
                    "content": "No optimized IR found",
                }))
                return None

            opt_bin = getattr(cfg, "opt_bin",
                              "/home/amax/yangz/Env/llvm-project/build/bin/opt")
            llvm_link = opt_bin.replace("/opt", "/llvm-link")
            llvm_extract = opt_bin.replace("/opt", "/llvm-extract")
            llvm_as = opt_bin.replace("/opt", "/llvm-as")
            llvm_dis = opt_bin.replace("/opt", "/llvm-dis")

            apply_dir = temp / "apply"
            apply_dir.mkdir(parents=True, exist_ok=True)

            orig_file = str(input_path)
            opt_file = str(output_file)
            merged_ll = str(apply_dir / f"{stem}.merged.ll")
            summary_parts = []

            # Step A: extract function names from optimized IR to know what to replace
            opt_ir = output_file.read_text(encoding="utf-8")
            from modules.verification import _parse_ir_structure, _extract_func_name
            _, opt_defs = _parse_ir_structure(opt_ir)
            func_names = []
            for d in opt_defs:
                fn = _extract_func_name(d)
                if fn:
                    func_names.append(fn)

            if not func_names:
                log("STEP_RESULT:" + json.dumps({
                    "step": 9, "status": "fail",
                    "title": "Apply",
                    "content": "No function definitions found in optimized IR",
                }))
                return None

            summary_parts.append(f"Functions to replace: {', '.join(func_names)}")

            # Step B: assemble both to .bc
            orig_bc = str(apply_dir / "orig.bc")
            opt_bc = str(apply_dir / "opt.bc")
            stripped_bc = str(apply_dir / "stripped.bc")

            def _run(cmd):
                r = _sp.run(cmd, capture_output=True, text=True, timeout=60)
                return r.returncode, (r.stdout or "") + (r.stderr or "")

            rc, out = _run([llvm_as, orig_file, "-o", orig_bc])
            if rc != 0:
                summary_parts.append(f"llvm-as (original) failed: {out[:500]}")
                log("STEP_RESULT:" + json.dumps({
                    "step": 9, "status": "fail",
                    "title": "Apply: llvm-as failed",
                    "content": "\n".join(summary_parts),
                }))
                return None

            rc, out = _run([llvm_as, opt_file, "-o", opt_bc])
            if rc != 0:
                summary_parts.append(f"llvm-as (optimized) failed: {out[:500]}")
                log("STEP_RESULT:" + json.dumps({
                    "step": 9, "status": "fail",
                    "title": "Apply: llvm-as failed",
                    "content": "\n".join(summary_parts),
                }))
                return None

            # Step C: strip the original functions from orig.bc
            delete_args = []
            for fn in func_names:
                delete_args.extend(["--delete", f"--func={fn}"])
            rc, out = _run([llvm_extract, *delete_args, orig_bc, "-o", stripped_bc])
            if rc != 0:
                summary_parts.append(f"llvm-extract --delete failed: {out[:500]}")
                log("STEP_RESULT:" + json.dumps({
                    "step": 9, "status": "fail",
                    "title": "Apply: llvm-extract failed",
                    "content": "\n".join(summary_parts),
                }))
                return None

            # Step D: llvm-link stripped + optimized
            linked_bc = str(apply_dir / "linked.bc")
            rc, out = _run([llvm_link, stripped_bc, opt_bc, "-o", linked_bc])
            if rc != 0:
                summary_parts.append(f"llvm-link failed: {out[:500]}")
                log("STEP_RESULT:" + json.dumps({
                    "step": 9, "status": "fail",
                    "title": "Apply: llvm-link failed",
                    "content": "\n".join(summary_parts),
                }))
                return None

            # Step E: disassemble back to .ll
            rc, out = _run([llvm_dis, linked_bc, "-o", merged_ll])
            if rc != 0:
                summary_parts.append(f"llvm-dis failed: {out[:500]}")

            # Step F: run opt -passes='verify'
            rc, out = _run([opt_bin, "-passes=verify", "-disable-output", merged_ll])
            verify_ok = (rc == 0)
            if verify_ok:
                summary_parts.append("opt -passes='verify': PASS ✅")
            else:
                summary_parts.append(f"opt -passes='verify': FAIL ❌\n{out[:1000]}")

            log("STEP_RESULT:" + json.dumps({
                "step": 9,
                "status": "ok" if verify_ok else "fail",
                "title": f"Apply & Verify: {'PASS' if verify_ok else 'FAIL'}",
                "content": "\n".join(summary_parts),
                "content_type": "text",
                "output_file": merged_ll,
                "verify_ok": verify_ok,
            }))
            return merged_ll

        else:
            raise SystemExit(f"Invalid step: {step}")

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
            harness_dir=getattr(cfg, "harness_dir", ""),
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
    # Verify mode: alive2 → diff_test fallback
    # ------------------------------------------------------------------

    def verify(self, input_path: str, output_dir: str):
        """Run alive2 formal verification first; for cases that fail or
        timeout, fall back to differential testing.

        *input_path* follows the same convention as diff_test().
        """
        cfg = self.config
        work_dir = Path(output_dir) / "verify"
        work_dir.mkdir(parents=True, exist_ok=True)

        combined_dir = work_dir / "combined"
        alive2_dir = work_dir / "alive2"

        # Resolve original_dir / optimized_dir
        if ":" in input_path:
            original_dir, optimized_dir = input_path.split(":", 1)
        else:
            optimized_dir = input_path
            original_dir = getattr(cfg, "ll_dir", "")
            if not original_dir or not Path(original_dir).is_dir():
                raise SystemExit(
                    "Cannot determine original IR directory. Either pass "
                    "original_dir:optimized_dir or set ll_dir in config."
                )

        # Step 1: prepare combined IR (reuse if exists)
        if combined_dir.is_dir() and any(combined_dir.glob("*/combined.ll")):
            log("Reusing existing combined IR")
        else:
            self.diff_tester.prepare_combined(
                original_dir, optimized_dir, str(combined_dir),
            )

        # Step 2: run alive2
        alive2_bin = getattr(cfg, "alive2_bin",
                             "/home/amax/yangz/Env/alive2/build/alive-tv")
        alive2_timeout = getattr(cfg, "alive2_timeout", 60)
        alive2_workers = getattr(cfg, "alive2_workers", 16)
        alive2_strict = getattr(cfg, "alive2_strict", False)

        alive2_results = self.diff_tester.run_alive2(
            str(combined_dir), str(alive2_dir),
            alive2_bin=alive2_bin,
            timeout=alive2_timeout,
            workers=alive2_workers,
            strict=alive2_strict,
        )
        self._write_alive2_report(alive2_results, work_dir)

        # Step 3: collect cases that need diff testing (FAIL / TIMEOUT / ERROR)
        need_diff = {
            stem for stem, info in alive2_results.items()
            if info["status"] not in ("PASS",)
        }

        if not need_diff:
            log("All cases verified by alive2, no diff testing needed")
            return str(work_dir)

        log(f"{len(need_diff)} cases need diff testing fallback")

        # Step 4: diff test only the failed/timeout cases
        dt_dir = work_dir / "diff_test"
        dt_dir.mkdir(parents=True, exist_ok=True)

        harness_dir = getattr(cfg, "harness_dir", "")
        fuzz_runs = getattr(cfg, "fuzz_runs", 200000)
        fuzz_timeout = getattr(cfg, "fuzz_timeout", 600)
        build_workers = getattr(cfg, "build_workers", 16)
        fuzz_workers = getattr(cfg, "fuzz_workers", 1)

        # Build a filtered combined dir with only the cases that need testing
        filtered_combined = dt_dir / "combined"
        filtered_combined.mkdir(parents=True, exist_ok=True)
        for stem in need_diff:
            src = combined_dir / stem
            dst = filtered_combined / stem
            if src.is_dir() and not dst.exists():
                shutil.copytree(str(src), str(dst))

        fuzz_results = self.diff_tester.run_full(
            original_dir=original_dir,
            optimized_dir=optimized_dir,
            work_dir=str(dt_dir),
            harness_dir=harness_dir,
            model=getattr(cfg, "llm_model", "gpt-5"),
            api_mode=getattr(cfg, "api_mode", "auto"),
            workers=getattr(cfg, "workers", 50),
            build_workers=build_workers,
            fuzz_runs=fuzz_runs,
            fuzz_timeout=fuzz_timeout,
            fuzz_workers=fuzz_workers,
        )
        self._write_fuzz_report(fuzz_results, dt_dir)

        # Step 5: merge results
        merged = {}
        for stem, info in alive2_results.items():
            if info["status"] == "PASS":
                merged[stem] = {"status": "PASS", "method": "alive2"}
            elif stem in fuzz_results:
                merged[stem] = {
                    "status": fuzz_results[stem]["status"],
                    "method": "diff_test",
                    "exit_code": fuzz_results[stem].get("exit_code", -1),
                }
            else:
                merged[stem] = {"status": info["status"], "method": "alive2"}

        self._write_verify_report(merged, work_dir)
        n_pass = sum(1 for r in merged.values() if r["status"] == "PASS")
        log(f"Verification complete. PASS={n_pass}/{len(merged)}")
        return str(work_dir)

    @staticmethod
    def _write_alive2_report(results: dict, work_dir: Path) -> None:
        """Write CSV summary of alive2 results."""
        import csv
        report = work_dir / "alive2_report.csv"
        with report.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "status"])
            w.writeheader()
            for stem, info in sorted(results.items()):
                w.writerow({"file": stem, "status": info["status"]})
        log(f"Alive2 report: {report}")

    @staticmethod
    def _write_verify_report(results: dict, work_dir: Path) -> None:
        """Write merged verification report (alive2 + diff_test)."""
        import csv
        report = work_dir / "verify_report.csv"
        with report.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "status", "method"])
            w.writeheader()
            for stem, info in sorted(results.items()):
                w.writerow({
                    "file": stem,
                    "status": info["status"],
                    "method": info.get("method", ""),
                })
        log(f"Verification report: {report}")

    # ------------------------------------------------------------------
    # Performance testing mode
    # ------------------------------------------------------------------

    def perf_test(self, input_path: str, output_dir: str):
        """Run performance benchmarking on optimised IR.

        Requires verification (alive2 + diff_test) to have passed first.
        If verify results already exist under *output_dir*/verify/, reuses
        them; otherwise runs the full verify pipeline first.

        For cases that only passed via alive2 (no diff_test binary), we
        build harness + fuzz binary so corpus collection can work.

        *input_path* follows the same convention as diff_test().
        """
        cfg = self.config
        work_dir = Path(output_dir)
        verify_dir = work_dir / "verify"
        perf_dir = work_dir / "perf_test"
        perf_dir.mkdir(parents=True, exist_ok=True)

        # Ensure verification has been done
        verify_report = verify_dir / "verify_report.csv"
        if not verify_report.exists():
            log("No verify results found, running verify first ...")
            self.verify(input_path, output_dir)

        if not verify_report.exists():
            raise SystemExit("Verification did not produce a report")

        # Read verify results
        import csv
        passed_stems = set()
        all_stems = set()
        with verify_report.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_stems.add(row["file"])
                if row["status"] == "PASS":
                    passed_stems.add(row["file"])
        log(f"Verify: {len(passed_stems)}/{len(all_stems)} cases passed, "
            f"benchmarking passed cases only")

        if not passed_stems:
            log("No cases passed verification, nothing to benchmark")
            return str(perf_dir)

        # Locate combined IR (always produced by verify)
        combined_dir = verify_dir / "combined"
        if not combined_dir.is_dir():
            raise SystemExit("No combined IR found under verify/")

        # Locate existing fuzz binaries from diff_test (if any)
        dt_bin_dir = verify_dir / "diff_test" / "bins"
        if not (dt_bin_dir.is_dir() and any(dt_bin_dir.rglob("*_fuzz"))):
            dt_bin_dir = work_dir / "diff_test" / "bins"

        # Check which passed stems already have a fuzz binary
        existing_bins = set()
        if dt_bin_dir.is_dir():
            for b in dt_bin_dir.rglob("*_fuzz"):
                if b.is_file():
                    existing_bins.add(b.parent.name)

        missing_stems = passed_stems - existing_bins

        # For alive2-only cases, build harness + fuzz binary so corpus
        # collection works for them too
        harness_cfg = getattr(cfg, "harness_dir", "")
        build_workers = getattr(cfg, "build_workers", 16)
        perf_bin_dir = perf_dir / "fuzz_bins"

        if missing_stems:
            log(f"{len(missing_stems)} alive2-only cases need fuzz binaries "
                f"for corpus collection, building ...")

            # Generate harnesses for missing cases only
            perf_harness_dir = perf_dir / "harness"

            if harness_cfg and Path(harness_cfg).is_dir():
                # Check if provided harness dir covers the missing stems
                has_harness = {
                    p.name[: -len(".fuzz.cc")]
                    for p in Path(harness_cfg).glob("*.fuzz.cc")
                }
                still_missing = missing_stems - has_harness
                if not still_missing:
                    perf_harness_dir = Path(harness_cfg)
                else:
                    # Generate only for truly missing ones
                    self.diff_tester.generate_harnesses(
                        str(combined_dir), str(perf_harness_dir),
                        model=getattr(cfg, "llm_model", "gpt-5"),
                        api_mode=getattr(cfg, "api_mode", "auto"),
                        workers=getattr(cfg, "workers", 50),
                    )
            else:
                self.diff_tester.generate_harnesses(
                    str(combined_dir), str(perf_harness_dir),
                    model=getattr(cfg, "llm_model", "gpt-5"),
                    api_mode=getattr(cfg, "api_mode", "auto"),
                    workers=getattr(cfg, "workers", 50),
                )

            self.diff_tester.build_binaries(
                str(combined_dir), str(perf_harness_dir), str(perf_bin_dir),
                workers=build_workers,
            )

        # Merge all fuzz binary dirs: existing diff_test bins + newly built
        # Use a unified dir so perf_tester sees everything
        unified_bin_dir = perf_dir / "all_fuzz_bins"
        unified_bin_dir.mkdir(parents=True, exist_ok=True)

        # Symlink existing bins
        if dt_bin_dir.is_dir():
            for stem_dir in dt_bin_dir.iterdir():
                if stem_dir.is_dir() and stem_dir.name in passed_stems:
                    dst = unified_bin_dir / stem_dir.name
                    if not dst.exists():
                        dst.symlink_to(stem_dir.resolve())

        # Symlink newly built bins
        if perf_bin_dir.is_dir():
            for stem_dir in perf_bin_dir.iterdir():
                if stem_dir.is_dir() and stem_dir.name in passed_stems:
                    dst = unified_bin_dir / stem_dir.name
                    if not dst.exists():
                        dst.symlink_to(stem_dir.resolve())

        harness_dir = Path(harness_cfg) if harness_cfg and Path(harness_cfg).is_dir() else (
            perf_dir / "harness" if (perf_dir / "harness").is_dir() else
            verify_dir / "diff_test" / "harness"
        )

        # Run perf pipeline
        corpus_time = getattr(cfg, "corpus_time", 10)
        corpus_workers = getattr(cfg, "corpus_workers", 0)
        bench_iters = getattr(cfg, "bench_iters", 1000000)
        bench_timeout = getattr(cfg, "bench_timeout", 300)
        bench_workers = getattr(cfg, "bench_workers", 1)
        corpus_dir = getattr(cfg, "corpus_dir", "")
        perf_harness_dir = getattr(cfg, "perf_harness_dir", "")

        results = self.perf_tester.run_full(
            combined_dir=str(combined_dir),
            harness_dir=str(harness_dir),
            fuzz_bin_dir=str(unified_bin_dir),
            work_dir=str(perf_dir),
            corpus_dir=corpus_dir,
            perf_harness_dir=perf_harness_dir,
            corpus_time=corpus_time,
            corpus_workers=corpus_workers,
            build_workers=build_workers,
            bench_iters=bench_iters,
            bench_timeout=bench_timeout,
            bench_workers=bench_workers,
        )

        # Ensure all passed_stems have an entry; missing ones get speedup=1
        filtered = {}
        for stem in passed_stems:
            if stem in results:
                filtered[stem] = results[stem]
            else:
                filtered[stem] = {
                    "status": "NO_BENCH",
                    "speedup": 1.0,
                    "baseline_ns": 0,
                    "opt_ns": 0,
                    "n_corpus": 0,
                    "per_corpus": [],
                }
        results = filtered

        # Add verify-failed cases with speedup=0
        failed_stems = all_stems - passed_stems
        for stem in failed_stems:
            results[stem] = {
                "status": "VERIFY_FAIL",
                "speedup": 0,
                "baseline_ns": 0,
                "opt_ns": 0,
                "n_corpus": 0,
                "per_corpus": [],
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
    p.add_argument("--mode", choices=["single", "batch", "diff_test", "perf_test", "verify", "step"], required=True)
    p.add_argument("--step", type=int, default=0,
                   help="Step number for interactive mode (0-5)")
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
        if v is not None and k not in ("mode", "input", "output", "config", "step")
    }
    config = Config(config_file=args.config, **overrides)
    optimizer = IROptimizer(config)
    result = optimizer.run(args.input, args.output, args.mode, step=args.step)
    if result:
        log(f"All done.  Result: {result}")


if __name__ == "__main__":
    main()
