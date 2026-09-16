#!/usr/bin/env python3
"""Create a compact paired report for one single-task checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("dino", "vae", "dual")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_case_results(path: Path) -> dict[int, int]:
    results: dict[int, int] = {}
    for result_path in sorted((path / "libero_object").glob("gpu*_task*_results.json")):
        match = re.search(r"_task(\d+)_results$", result_path.stem)
        if match:
            payload = load_json(result_path)
            results[int(match.group(1))] = int(payload.get("successes", 0) > 0)
    return results


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap(left: list[int], right: list[int], seed: int, draws: int = 10_000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(left)
    deltas = []
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        delta = sum(right[i] - left[i] for i in indices) / n
        deltas.append(100.0 * delta)
    deltas.sort()
    return deltas[int(0.025 * draws)], deltas[int(0.975 * draws) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("evaluate_results/single_task"))
    parser.add_argument("--tag", default="step400_seed42")
    parser.add_argument(
        "--classification-path",
        type=Path,
        default=Path("/home/pai/zxw/LIBERO-plus/libero/libero/benchmark/task_classification.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.result_root.resolve()
    output = args.output or root / f"comparison_{args.tag}.md"
    classification_raw = load_json(args.classification_path.resolve())
    categories = {
        int(entry["id"]) - 1: str(entry.get("category", "Unknown"))
        for entry in classification_raw["libero_object"]
    }

    clean: dict[str, dict[str, Any]] = {}
    plus: dict[str, dict[str, Any]] = {}
    cases: dict[str, dict[int, int]] = {}
    for variant in VARIANTS:
        clean[variant] = load_json(
            root / f"{variant}_clean_{args.tag}/libero_object/gpu0_task3_results.json"
        )
        plus[variant] = load_json(root / f"{variant}_plus_140_{args.tag}/plus_summary.json")
        cases[variant] = load_case_results(root / f"{variant}_plus_140_{args.tag}")

    common = sorted(set(cases["vae"]) & set(cases["dual"]))
    if not common:
        raise RuntimeError("No paired VAE/Dual Plus cases found")
    vae_values = [cases["vae"][i] for i in common]
    dual_values = [cases["dual"][i] for i in common]
    both_success = sum(v and d for v, d in zip(vae_values, dual_values))
    dual_only = sum((not v) and d for v, d in zip(vae_values, dual_values))
    vae_only = sum(v and (not d) for v, d in zip(vae_values, dual_values))
    both_fail = len(common) - both_success - dual_only - vae_only
    ci_low, ci_high = paired_bootstrap(vae_values, dual_values, args.seed)

    category_rows = []
    by_category: dict[str, list[int]] = defaultdict(list)
    for task_id in common:
        by_category[categories.get(task_id, "Unknown")].append(task_id)
    for category in sorted(by_category):
        ids = by_category[category]
        vae_rate = 100 * sum(cases["vae"][i] for i in ids) / len(ids)
        dual_rate = 100 * sum(cases["dual"][i] for i in ids) / len(ids)
        category_rows.append((category, len(ids), vae_rate, dual_rate, dual_rate - vae_rate))

    lines = [
        f"# 单任务消融评测汇总：{args.tag}",
        "",
        "| Variant | Clean | LIBERO-Plus | Robustness drop |",
        "| --- | ---: | ---: | ---: |",
    ]
    summary_json: dict[str, Any] = {"tag": args.tag, "variants": {}}
    for variant in VARIANTS:
        clean_success = int(clean[variant]["successes"])
        clean_total = int(clean[variant]["total_episodes"])
        total = plus[variant]["leaderboard"]["Total"]
        plus_success = int(total["Successes"])
        plus_total = int(total["Episodes"])
        clean_rate = 100 * clean_success / clean_total
        plus_rate = 100 * plus_success / plus_total
        lines.append(
            f"| {variant} | {clean_success}/{clean_total} = {clean_rate:.2f}% | "
            f"{plus_success}/{plus_total} = {plus_rate:.2f}% | {clean_rate - plus_rate:.2f}pp |"
        )
        summary_json["variants"][variant] = {
            "clean_successes": clean_success,
            "clean_episodes": clean_total,
            "plus_successes": plus_success,
            "plus_episodes": plus_total,
        }

    lines += [
        "",
        "## Dual-Space 与 VAE 配对比较",
        "",
        f"- 配对 cases：{len(common)}",
        f"- 两者都成功：{both_success}",
        f"- 仅 Dual-Space 成功：{dual_only}",
        f"- 仅 VAE 成功：{vae_only}",
        f"- 两者都失败：{both_fail}",
        f"- McNemar exact p：{exact_mcnemar(dual_only, vae_only):.6g}",
        f"- Dual−VAE paired bootstrap 95% CI：[{ci_low:.2f}, {ci_high:.2f}]pp",
        "",
        "| Category | N | VAE | Dual | Dual−VAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, count, vae_rate, dual_rate, delta in category_rows:
        lines.append(f"| {category} | {count} | {vae_rate:.2f}% | {dual_rate:.2f}% | {delta:+.2f}pp |")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_json["paired_vae_dual"] = {
        "cases": len(common),
        "both_success": both_success,
        "dual_only": dual_only,
        "vae_only": vae_only,
        "both_fail": both_fail,
        "mcnemar_exact_p": exact_mcnemar(dual_only, vae_only),
        "bootstrap_delta_pp_95ci": [ci_low, ci_high],
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
