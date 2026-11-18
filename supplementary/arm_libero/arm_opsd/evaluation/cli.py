"""Evaluation commands: python -m arm_opsd.evaluation.cli <command> (console script: arm-opsd-eval).

  fetch-student         a released student (arm_libero.json + student_adapter.pt) from the Hugging Face model repository
  fetch-libero-pro      the official LIBERO-Pro2.0 evaluation package
  prepare-prompts       T5 embeddings of every LIBERO-Pro2.0 instruction of the plan
  prepare-long-prompts  T5 embeddings of the ten LIBERO-Long instructions
  evaluate              LIBERO-Pro2.0, one policy (base_1_2 or student_1_2)
  evaluate-long         standard LIBERO-Long, one policy
  official-summary      LIBERO-Pro2.0 BASE / static / runtime group rates of one results directory
  report / report-long  one paired comparison (LIBERO-Pro2.0 / LIBERO-Long)
  paired-table          every comparison of 'Paired comparisons on LIBERO' from a map of results directories
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_REPO, EVAL_REVISION = "zhouxueyang/LIBERO-Pro2.0-Eval", "d1933228f8be59f5d8fa1d8e2e251b463dfb64bc"
PAIRED_TABLE = ROOT / "configs/evaluation/paired_libero.json"


def write_new(path, value, text=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    if text is not None:
        path.with_suffix(".txt").write_text(text)


def paired_table(results, plan=None, spec=PAIRED_TABLE, *, bootstrap_samples=10000, seed=42):
    """Rows of the supplementary table 'Paired comparisons on LIBERO'. `results` maps a policy name to {"libero_pro2": dir, "libero_long": dir};
    a comparison whose results are missing is skipped. Delta = first minus second, in points."""
    from arm_opsd.evaluation.libero_long import libero_long_report
    from arm_opsd.evaluation.libero_pro import libero_pro_report
    spec = json.loads(Path(spec).read_text())
    rows, skipped = [], []
    for item in spec["comparisons"]:
        benchmark, first, second = item["benchmark"], item["first"], item["second"]
        dirs = [results.get(name, {}).get(benchmark) for name in (first, second)]
        if not all(dirs) or (benchmark == "libero_pro2" and not plan):
            skipped.append(item["label"])
            continue
        if benchmark == "libero_pro2":
            report = libero_pro_report(dirs[1], dirs[0], plan, bootstrap_samples=bootstrap_samples, seed=seed)["overall"]
            ci = report["paired_macro_delta_ci95"]
        else:
            report = libero_long_report(dirs[1], dirs[0], bootstrap_samples=bootstrap_samples, seed=seed)
            ci = report["macro_delta_ci95"]
        rows.append({"benchmark": benchmark, "comparison": item["label"], "n": report["n"],
                     "delta_points": 100 * report["macro_delta"], "plus": report["improved"],
                     "minus": report["regressed"], "ci95_points": [100 * x for x in ci] if ci else None,
                     "mcnemar_p": report["mcnemar_exact_p"]})
    lines = [f"{'benchmark':<12} {'comparison':<26} {'delta':>6} {'+/-':>9} {'95% CI':>16} {'p':>9}"]
    for row in rows:
        ci = "[{:.1f}, {:.1f}]".format(*row["ci95_points"]) if row["ci95_points"] else "-"
        lines.append(f"{row['benchmark']:<12} {row['comparison']:<26} {row['delta_points']:+6.1f} "
                     f"{row['plus']:>4}/{row['minus']:<4} {ci:>16} {row['mcnemar_p']:9.2g}")
    if skipped:
        lines.append(f"skipped (results missing): {', '.join(skipped)}")
    return {"rows": rows, "skipped": skipped}, "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="arm-opsd-eval", description="LIBERO-Long and LIBERO-Pro2.0 evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch-student")
    p.add_argument("--name", default="opsd_s300_ema", help="entry of configs/evaluation/released_students.json")
    p.add_argument("--output", required=True)
    p.add_argument("--repo")
    p.add_argument("--path")
    p.add_argument("--revision", help="default: the entry's revision (the HF `main` branch)")
    p = sub.add_parser("fetch-libero-pro")
    p.add_argument("--output", required=True)
    p = sub.add_parser("prepare-prompts")
    p.add_argument("--model-root", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--instructions", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("prepare-long-prompts")
    p.add_argument("--model-root", required=True)
    p.add_argument("--reset-manifest", required=True)
    p.add_argument("--output", required=True)
    for name in ("evaluate", "evaluate-long"):
        p = sub.add_parser(name)
        p.add_argument("--policy", choices=("base_1_2", "student_1_2"), required=True)
        p.add_argument("--student", help="student export directory (arm_libero.json + student_adapter.pt, or a published export.json + adapter.pt)")
        p.add_argument("--model-root", required=True)
        p.add_argument("--prompts", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--seed", type=int, default=42, help="master seed of the generation noise")
        p.add_argument("--max-episodes", type=int)
        if name == "evaluate":
            p.add_argument("--plan", required=True)
            p.add_argument("--instructions", required=True)
            p.add_argument("--run-id", required=True, help="LIBERO-Pro2.0 run submitted from --plan for this policy")
            p.add_argument("--pro-python", required=True, help="Python of the LIBERO-Pro2.0 client environment")
            p.add_argument("--runner-id")
        else:
            p.add_argument("--reset-manifest", required=True)
            p.add_argument("--cpu-python", required=True)
            p.add_argument("--worker-env", required=True)
    p = sub.add_parser("official-summary")
    p.add_argument("--results", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("report")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("report-long")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("paired-table")
    p.add_argument("--results", required=True, help='JSON: {"<policy>": {"libero_pro2": DIR, "libero_long": DIR}}')
    p.add_argument("--plan", help="LIBERO-Pro2.0 plan.json (needed for its rows)")
    p.add_argument("--spec", default=str(PAIRED_TABLE))
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "fetch-student":
        from arm_opsd.evaluation.policies import fetch_student
        print(fetch_student(args.name, args.output, repo=args.repo, path=args.path, revision=args.revision))
    elif args.command == "fetch-libero-pro":
        from huggingface_hub import snapshot_download
        snapshot_download(EVAL_REPO, repo_type="dataset", revision=EVAL_REVISION, local_dir=args.output)
        print(json.dumps({"repo": EVAL_REPO, "revision": EVAL_REVISION, "download": args.output}, indent=2))
    elif args.command == "prepare-prompts":
        from arm_opsd.evaluation.libero_pro import prepare_libero_pro_prompts
        print(json.dumps(prepare_libero_pro_prompts(args.model_root, args.instructions, args.plan, args.output), indent=2))
    elif args.command == "prepare-long-prompts":
        from arm_opsd.assets.base_model import prepare_prompt_bank
        from arm_opsd.assets.resets import load_test_manifest
        cases = [c for c in load_test_manifest(args.reset_manifest)["cases"] if c["split"] == "test"]
        print(json.dumps(prepare_prompt_bank(args.model_root, cases, args.output), indent=2))
    elif args.command == "evaluate":
        from arm_opsd.evaluation.libero_pro import evaluate_libero_pro
        print(json.dumps(evaluate_libero_pro(model_root=args.model_root, prompts=args.prompts, plan_path=args.plan,
            instructions_path=args.instructions, run_id=args.run_id, policy=args.policy, output_dir=args.output,
            pro_python=args.pro_python, student=args.student, seed=args.seed, runner_id=args.runner_id,
            max_episodes=args.max_episodes), indent=2))
    elif args.command == "evaluate-long":
        from arm_opsd.evaluation.libero_long import evaluate_libero_long
        evaluate_libero_long(model_root=args.model_root, prompts=args.prompts, reset_manifest=args.reset_manifest,
            cpu_python=args.cpu_python, worker_env=args.worker_env, policy=args.policy, output_dir=args.output,
            student=args.student, seed=args.seed, max_episodes=args.max_episodes)
    elif args.command == "official-summary":
        from arm_opsd.evaluation.libero_pro import libero_pro_official_summary, official_summary_text
        result = libero_pro_official_summary(args.results, args.plan)
        write_new(args.output, result, official_summary_text(result))
        print(official_summary_text(result))
    elif args.command == "report":
        from arm_opsd.evaluation.libero_pro import libero_pro_report
        result = libero_pro_report(args.baseline, args.candidate, args.plan)
        write_new(args.output, result)
        print(json.dumps({k: result["overall"][k] for k in ("macro_delta", "paired_macro_delta_ci95", "improved",
                                                             "regressed", "mcnemar_exact_p")}, indent=2))
    elif args.command == "report-long":
        from arm_opsd.evaluation.libero_long import libero_long_report
        result = libero_long_report(args.baseline, args.candidate)
        write_new(args.output, result, result["text"])
        print(result["text"])
    elif args.command == "paired-table":
        result, text = paired_table(json.loads(Path(args.results).read_text()), args.plan, args.spec)
        write_new(args.output, result, text)
        print(text)


if __name__ == "__main__":
    main()
