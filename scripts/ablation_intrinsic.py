#!/usr/bin/env python3
"""Intrinsic suggestion ablation: for cases that had intrinsic suggestions,
re-run step3b→step4→step5 WITHOUT the intrinsic block, then compare.

Usage:
  python scripts/ablation_intrinsic.py \
    --experiment-dir test/TSVC_2/intoptplus_results \
    --config config/config.yaml
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.llm_client import LLMClient
from modules.post_processing import PostProcessor
from modules.utils import extract_single_block, log


def find_intrinsic_cases(step3_dir: str) -> list:
    """Find prompt files that contain <intrinsics> blocks."""
    cases = []
    for f in sorted(Path(step3_dir).glob("*.prompt.ll")):
        text = f.read_text(encoding="utf-8")
        if "<intrinsics>" in text:
            stem = f.stem.replace(".prompt", "")
            cases.append(stem)
    return cases


def strip_intrinsics(text: str) -> str:
    """Remove the intrinsic block and surrounding instruction text from a prompt."""
    # Remove "The following hardware intrinsics..." through "</intrinsics>"
    text = re.sub(
        r"The following hardware intrinsics.*?</intrinsics>\s*",
        "", text, flags=re.DOTALL,
    )
    # Fallback patterns
    text = re.sub(
        r"IMPORTANT:.*?</intrinsics>\s*(?:In your.*?\n)*\s*",
        "", text, flags=re.DOTALL,
    )
    text = re.sub(r"<intrinsics>.*?</intrinsics>\s*", "", text, flags=re.DOTALL)
    return text


def main():
    p = argparse.ArgumentParser(description="Intrinsic ablation experiment")
    p.add_argument("--experiment-dir", required=True,
                   help="Directory with step3_refinement/, step4_realization/, etc.")
    p.add_argument("--config", default="../config/config.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="Only list affected cases, don't re-run")
    p.add_argument("--tsvc-src", default="test/TSVC_2/src",
                   help="TSVC source dir (for -I include path)")
    p.add_argument("--split-c", default="test/TSVC_2/split/c",
                   help="Dir with {name}_main.c and globals.c")
    args = p.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    llm_cfg = cfg.get("llm", {})

    exp_dir = Path(args.experiment_dir)
    step3_dir = exp_dir / "step3_refinement"
    ablation_dir = exp_dir / "ablation_no_intrinsic"

    # Step 1: Find cases with intrinsic suggestions
    cases = find_intrinsic_cases(str(step3_dir))
    log(f"Found {len(cases)} cases with intrinsic suggestions")

    if not cases:
        log("Nothing to ablate")
        return

    if args.dry_run:
        for c in cases:
            log(f"  {c}")
        return

    # Step 2: Create ablation step3 directory with intrinsics stripped
    abl_step3 = ablation_dir / "step3_refinement"
    abl_step4 = ablation_dir / "step4_realization"
    abl_opt = ablation_dir / "opt"

    # Check what already exists and skip completed steps
    has_step3 = abl_step3.is_dir() and any(abl_step3.glob("*.model.predict.ll"))
    has_step4 = abl_step4.is_dir() and any(abl_step4.glob("*.model.predict.ll"))
    has_opt = abl_opt.is_dir() and any(abl_opt.glob("*.optimized.ll"))

    llm = LLMClient(
        base_url=llm_cfg.get("base_url", ""),
        api_key=llm_cfg.get("api_key", ""),
    )

    if not has_step3:
        abl_step3.mkdir(parents=True, exist_ok=True)
        for stem in cases:
            src_prompt = step3_dir / f"{stem}.prompt.ll"
            dst_prompt = abl_step3 / f"{stem}.prompt.ll"
            text = src_prompt.read_text(encoding="utf-8")
            dst_prompt.write_text(strip_intrinsics(text), encoding="utf-8")
        log(f"Created {len(cases)} stripped prompts in {abl_step3}")

        log("Running LLM refinement without intrinsics ...")
        llm.batch_query(
            in_dir=str(abl_step3),
            out_dir=str(abl_step3),
            model=llm_cfg.get("llm_model", "gpt-5"),
            api_mode=llm_cfg.get("api_mode", "auto"),
            workers=min(llm_cfg.get("workers", 50), len(cases)),
            max_output_tokens=llm_cfg.get("max_output_tokens", 8192),
        )
    else:
        log("Step 3 already done, skipping")

    if not has_step4:
        log("Rewriting prompts for realization ...")
        abl_step4.mkdir(parents=True, exist_ok=True)

        OLD_HEADER = ("You may refer to the following advice, but feel free "
                      "to adapt, extend, or deviate from it as you see fit.")
        NEW_HEADER = "You can refer to the following advice."
        OLD_FOOTER = ("Please output the final optimization advice wrapped in "
                      "<advice>...</advice> and the full optimized LLVM IR "
                      "wrapped in <code>...</code>.")
        NEW_FOOTER = ("Please output the full optimized LLVM IR wrapped in "
                      "<code>...</code>.")

        for stem in cases:
            prompt_file = abl_step3 / f"{stem}.prompt.ll"
            pred_file = abl_step3 / f"{stem}.model.predict.ll"
            if not pred_file.exists():
                log(f"  WARN: no prediction for {stem}, skipping")
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

            (abl_step4 / f"{stem}.prompt.ll").write_text(ctx, encoding="utf-8")

        log("Running LLM realization ...")
        llm.batch_query(
            in_dir=str(abl_step4),
            out_dir=str(abl_step4),
            model=llm_cfg.get("llm_model", "gpt-5"),
            api_mode=llm_cfg.get("api_mode", "auto"),
            workers=min(llm_cfg.get("workers", 50), len(cases)),
            max_output_tokens=llm_cfg.get("max_output_tokens", 8192),
        )
    else:
        log("Step 4 already done, skipping")

    if not has_opt:
        log("Post-processing ...")
        post = PostProcessor()
        post.run(in_dir=str(abl_step4))

        abl_opt.mkdir(parents=True, exist_ok=True)
        for pred in abl_step4.glob("*.model.predict.ll"):
            stem = pred.name.replace(".model.predict.ll", "")
            code = extract_single_block(pred.read_text(encoding="utf-8"), "code")
            if code:
                (abl_opt / f"{stem}.optimized.ll").write_text(code, encoding="utf-8")
    else:
        log("Optimized files already exist, skipping")

    n_out = len(list(abl_opt.glob("*.optimized.ll")))
    log(f"Ablation: {n_out}/{len(cases)} optimized files in {abl_opt}")

    # Step 6: TSVC-style compile + run + compare
    tsvc_cfg = cfg.get("diff_testing", {})
    llc = tsvc_cfg.get("llc", "/home/amax/yangz/Env/llvm-project/build/bin/llc")
    clang = tsvc_cfg.get("clangxx", "/home/amax/yangz/Env/llvm-project/build/bin/clang++").replace("clang++", "clang")
    llc_extra = tsvc_cfg.get("llc_extra_args", "")
    link_extra = tsvc_cfg.get("link_extra_files", "")
    tsvc_src = args.tsvc_src
    split_c = args.split_c

    if not tsvc_src or not split_c:
        log("Skipping TSVC eval (--tsvc-src and --split-c not provided)")
        log(f"Run manually:\n  python scripts/compare_tsvc.py --baseline ... --intopt ...")
        return

    import subprocess
    eval_dir = ablation_dir / "tsvc_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    abl_bin_dir = eval_dir / "bins"
    abl_bin_dir.mkdir(parents=True, exist_ok=True)

    abl_tsv = eval_dir / "ablation_no_intrinsic.tsv"
    with_tsv = eval_dir / "with_intrinsic.tsv"
    baseline_tsv = eval_dir / "baseline_o3.tsv"

    def compile_and_run(opt_dir, tsv_path, label):
        results = []
        ok = fail = 0
        for opt_ll in sorted(opt_dir.glob("*.optimized.ll")):
            stem = opt_ll.stem.replace(".optimized", "")
            main_c = Path(split_c) / f"{stem}_main.c"
            if not main_c.exists():
                continue

            bin_path = abl_bin_dir / f"{stem}_{label}"

            # llc → .s
            asm = abl_bin_dir / f"{stem}_{label}.s"
            llc_cmd = [llc, "-O3"] + (llc_extra.split() if llc_extra else []) + [str(opt_ll), "-o", str(asm)]
            r = subprocess.run(llc_cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                results.append(f"{stem}\tCOMPILE_FAIL\t0")
                fail += 1
                continue

            # link
            link_cmd = [clang, "-O3", "-Wno-everything",
                        f"-I{tsvc_src}", str(asm), str(main_c)]
            if link_extra:
                link_cmd.extend(link_extra.split())
            link_cmd.extend(["-lm", "-o", str(bin_path)])
            r = subprocess.run(link_cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                results.append(f"{stem}\tLINK_FAIL\t0")
                fail += 1
                continue

            # run
            try:
                r = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=120)
                out = r.stdout.strip().split("\n")[-1] if r.stdout else f"{stem}\tRUN_FAIL\t0"
                results.append(out)
                ok += 1
            except Exception:
                results.append(f"{stem}\tRUN_FAIL\t0")
                fail += 1

        with open(tsv_path, "w") as f:
            f.write("name\ttime\tchecksum\n")
            for line in results:
                f.write(line + "\n")
        log(f"  {label}: ok={ok} fail={fail} → {tsv_path}")

    # Only compile+run if TSV files don't exist yet
    if not abl_tsv.exists():
        log("Compiling & running ablation (no intrinsic) binaries ...")
        compile_and_run(abl_opt, abl_tsv, "no_intr")
    else:
        log(f"Ablation TSV exists, skipping: {abl_tsv}")

    if not with_tsv.exists():
        log("Compiling & running with-intrinsic binaries ...")
        with_opt = eval_dir / "with_intrinsic_opt"
        with_opt.mkdir(parents=True, exist_ok=True)
        for stem in cases:
            src = exp_dir / f"{stem}.optimized.ll"
            if src.exists():
                shutil.copy2(str(src), str(with_opt / f"{stem}.optimized.ll"))
        compile_and_run(with_opt, with_tsv, "with_intr")
    else:
        log(f"With-intrinsic TSV exists, skipping: {with_tsv}")

    if not baseline_tsv.exists():
        log("Running O3 baseline ...")
        baseline_bin_dir = Path(args.split_c).parent / "bin"
        baseline_results = []
        for stem in cases:
            bin_path = baseline_bin_dir / stem
            if bin_path.exists():
                try:
                    r = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=120)
                    out = r.stdout.strip().split("\n")[-1] if r.stdout else f"{stem}\tRUN_FAIL\t0"
                    baseline_results.append(out)
                except Exception:
                    baseline_results.append(f"{stem}\tRUN_FAIL\t0")
        with open(baseline_tsv, "w") as f:
            f.write("name\ttime\tchecksum\n")
            for line in baseline_results:
                f.write(line + "\n")
    else:
        log(f"Baseline TSV exists, skipping: {baseline_tsv}")

    # Compare all three
    log("\n=== Ablation Comparison ===")
    log(f"python scripts/compare_tsvc.py --baseline {baseline_tsv} --intopt {abl_tsv}")
    log(f"python scripts/compare_tsvc.py --baseline {baseline_tsv} --intopt {with_tsv}")

    # ── Comparison table ──
    import math, csv

    def _load_tsv(path):
        d = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("name"):
                    continue
                parts = [p.strip() for p in line.split("\t") if p.strip()]
                if len(parts) < 2:
                    continue
                name = parts[0]
                if any(x in line for x in ("COMPILE_FAIL", "LINK_FAIL", "RUN_FAIL")):
                    d[name] = {"time": -1, "checksum": "0"}
                    continue
                nums = []
                for pp in parts[1:]:
                    try: nums.append(float(pp))
                    except ValueError: pass
                if len(nums) >= 2:
                    d[name] = {"time": nums[-2], "checksum": str(nums[-1])}
                elif len(nums) == 1:
                    d[name] = {"time": nums[0], "checksum": "0"}
        return d

    def _ck_ok(x, y):
        """Check if checksums match. Both must have valid time (>0) to count as correct."""
        if x.get("time", -1) <= 0 or y.get("time", -1) <= 0:
            return False
        try:
            xf = float(x.get("checksum", "0"))
            yf = float(y.get("checksum", "0"))
            return abs(xf - yf) < max(abs(xf) * 1e-4, 1e-6)
        except Exception:
            return False

    base = _load_tsv(str(baseline_tsv))
    abl = _load_tsv(str(abl_tsv))
    wit = _load_tsv(str(with_tsv))

    # Print to console
    header = (f"{'benchmark':>12}  {'O3_time(s)':>10}  {'no_intr_time(s)':>15}  "
              f"{'w_intr_time(s)':>14}  {'speedup_no_intr':>15}  "
              f"{'speedup_w_intr':>14}  {'correct_no_intr':>15}  {'correct_w_intr':>14}")
    print(f"\n{'='*130}")
    print("Intrinsic Suggestion Ablation Study")
    print(f"{'='*130}")
    print(header)
    print("-" * 130)

    sp_no_list, sp_w_list = [], []
    ck_no_ok_cnt = ck_w_ok_cnt = 0
    rows = []

    for stem in sorted(cases):
        b = base.get(stem, {})
        a = abl.get(stem, {})
        w = wit.get(stem, {})
        bt = b.get("time", -1)
        at = a.get("time", -1)
        wt = w.get("time", -1)

        sp_no = bt / at if bt > 0 and at > 0 else 0
        sp_w = bt / wt if bt > 0 and wt > 0 else 0

        cn = "PASS" if _ck_ok(b, a) else ("COMPILE_FAIL" if at < 0 else "WRONG")
        cw = "PASS" if _ck_ok(b, w) else ("COMPILE_FAIL" if wt < 0 else "WRONG")
        if cn == "PASS": ck_no_ok_cnt += 1
        if cw == "PASS": ck_w_ok_cnt += 1
        if sp_no > 0: sp_no_list.append(sp_no)
        if sp_w > 0: sp_w_list.append(sp_w)

        at_s = f"{at:.4f}" if at > 0 else "FAIL"
        wt_s = f"{wt:.4f}" if wt > 0 else "FAIL"
        sp_no_s = f"{sp_no:.3f}x" if sp_no > 0 else "N/A"
        sp_w_s = f"{sp_w:.3f}x" if sp_w > 0 else "N/A"

        print(f"{stem:>12}  {bt:>10.4f}  {at_s:>15}  {wt_s:>14}  "
              f"{sp_no_s:>15}  {sp_w_s:>14}  {cn:>15}  {cw:>14}")

        rows.append({
            "benchmark": stem,
            "o3_time_s": f"{bt:.4f}" if bt > 0 else "N/A",
            "no_intrinsic_time_s": at_s,
            "with_intrinsic_time_s": wt_s,
            "speedup_no_intrinsic": f"{sp_no:.4f}" if sp_no > 0 else "N/A",
            "speedup_with_intrinsic": f"{sp_w:.4f}" if sp_w > 0 else "N/A",
            "correct_no_intrinsic": cn,
            "correct_with_intrinsic": cw,
        })

    print("-" * 130)
    n = len(cases)
    geo_no = math.exp(sum(math.log(s) for s in sp_no_list) / len(sp_no_list)) if sp_no_list else 0
    geo_w = math.exp(sum(math.log(s) for s in sp_w_list) / len(sp_w_list)) if sp_w_list else 0
    print(f"{'SUMMARY':>12}  {'':>10}  {'geomean:':>15}  {'geomean:':>14}  "
          f"{geo_no:>14.4f}x  {geo_w:>13.4f}x  "
          f"{ck_no_ok_cnt:>11}/{n:<3}  {ck_w_ok_cnt:>10}/{n:<3}")
    print(f"{'':>12}  {'':>10}  {'compiled:':>15}  {'compiled:':>14}  "
          f"{len(sp_no_list):>11}/{n:<3}   {len(sp_w_list):>10}/{n:<3}")
    print(f"{'='*130}\n")

    # Write CSV to eval_dir
    csv_path = eval_dir / "ablation_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "benchmark", "o3_time_s",
            "no_intrinsic_time_s", "with_intrinsic_time_s",
            "speedup_no_intrinsic", "speedup_with_intrinsic",
            "correct_no_intrinsic", "correct_with_intrinsic",
        ])
        w.writeheader()
        w.writerows(rows)
        # Summary row
        w.writerow({
            "benchmark": "GEOMEAN",
            "o3_time_s": "",
            "no_intrinsic_time_s": "",
            "with_intrinsic_time_s": "",
            "speedup_no_intrinsic": f"{geo_no:.4f}" if geo_no else "N/A",
            "speedup_with_intrinsic": f"{geo_w:.4f}" if geo_w else "N/A",
            "correct_no_intrinsic": f"{ck_no_ok_cnt}/{n}",
            "correct_with_intrinsic": f"{ck_w_ok_cnt}/{n}",
        })
    log(f"Comparison CSV: {csv_path}")


if __name__ == "__main__":
    main()
