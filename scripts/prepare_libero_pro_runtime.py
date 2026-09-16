#!/usr/bin/env python3
"""Prepare the small, reproducible LIBERO-PRO runtime used by evaluation.

LIBERO-PRO ships the swap/object/language/task BDDL and init-state folders, but
the environment-perturbed folder is not present in the checkout.  This helper
creates a runtime LIBERO config, links the existing folders, generates only
the three training-task environment BDDL files, and (when requested) creates
their initial states with the LIBERO-PRO generator.

The runtime is deliberately kept outside the source checkout.  It can safely
be cached under an evaluation result directory and reused by resumed runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


OBJECT_TASK_NAMES = [
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
]

SUITE_BY_PERTURBATION = {
    "env": "libero_object_env",
    "swap": "libero_object_swap",
    "object": "libero_object_object",
    "language": "libero_object_lan",
    "task": "libero_object_task",
}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_environment_perturbator(source_root: Path):
    perturbation_path = source_root / "perturbation.py"
    if not perturbation_path.is_file():
        raise FileNotFoundError(
            f"LIBERO-PRO perturbation.py is missing: {perturbation_path}"
        )
    spec = importlib.util.spec_from_file_location("libero_pro_perturbation", perturbation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {perturbation_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BDDLParser, module.EnvironmentReplacePerturbator


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _ensure_directory_link(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"LIBERO-PRO directory is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)

    if os.path.lexists(target):
        if target.is_symlink() and _same_path(target, source):
            return
        if target.is_dir() and not target.is_symlink():
            # A previous invocation may have materialized a directory instead
            # of a symlink.  Preserve it and validate the selected files later.
            return
        raise RuntimeError(f"Runtime path exists but is not a directory: {target}")

    target.symlink_to(source, target_is_directory=True)


def _write_libero_config(source_root: Path, runtime_root: Path) -> Path:
    config_dir = runtime_root / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    source_libero_root = source_root / "libero" / "libero"
    config = {
        "benchmark_root": str(source_libero_root.resolve()),
        "bddl_files": str((runtime_root / "bddl_files").resolve()),
        "init_states": str((runtime_root / "init_files").resolve()),
        "assets": str((source_libero_root / "assets").resolve()),
        "datasets": str((source_root / "libero" / "datasets").resolve()),
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _selected_tasks(task_ids: list[int]) -> list[str]:
    if task_ids != [3, 1, 9]:
        raise ValueError(
            "This evaluation is intentionally restricted to training task ids [3, 1, 9]. "
            f"Received {task_ids}."
        )
    return [OBJECT_TASK_NAMES[index] for index in task_ids]


def _ensure_selected_files(
    bddl_dir: Path,
    init_dir: Path,
    task_names: list[str],
    *,
    require_init: bool,
) -> None:
    for task_name in task_names:
        bddl_path = bddl_dir / f"{task_name}.bddl"
        if not bddl_path.is_file() or bddl_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing PRO BDDL file: {bddl_path}")
        if require_init:
            init_path = init_dir / f"{task_name}.pruned_init"
            if not init_path.is_file() or init_path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing PRO init-state file: {init_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_environment_bddl(
    source_root: Path,
    runtime_bddl_dir: Path,
    task_names: list[str],
    seed: int,
) -> list[str]:
    config_path = source_root / "libero_ood" / "ood_environment.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing LIBERO-PRO environment config: {config_path}")

    parser_cls, perturbator_cls = _load_environment_perturbator(source_root)
    source_bddl_dir = source_root / "libero" / "libero" / "bddl_files" / "libero_object"
    runtime_bddl_dir.mkdir(parents=True, exist_ok=True)
    changed_tasks: list[str] = []
    for task_name in task_names:
        source_path = source_bddl_dir / f"{task_name}.bddl"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source LIBERO-Object BDDL file: {source_path}")
        source_content = source_path.read_text(encoding="utf-8")
        parser = parser_cls(source_content)
        perturbator = perturbator_cls(parser, str(config_path))
        transformed = perturbator.perturb(
            task_suite_name="libero_object",
            task_name=task_name,
            seed=seed,
        )
        target_path = runtime_bddl_dir / source_path.name
        old_content = target_path.read_text(encoding="utf-8") if target_path.is_file() else None
        if old_content != transformed:
            target_path.write_text(transformed, encoding="utf-8")
            changed_tasks.append(task_name)

        if "living_room_table" not in transformed:
            raise RuntimeError(
                f"Environment perturbation did not produce living_room_table: {target_path}"
            )

    return changed_tasks


def _generate_missing_init_states(
    source_root: Path,
    runtime_root: Path,
    config_path: Path,
    task_names: list[str],
    init_count: int,
) -> None:
    if not task_names:
        return
    generator = source_root / "notebooks" / "generate_init_states.py"
    if not generator.is_file():
        raise FileNotFoundError(f"Missing LIBERO-PRO init-state generator: {generator}")

    pending_root = runtime_root / ".pending_init_states"
    pending_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="generation-", dir=str(pending_root)) as temp_dir:
        temp_root = Path(temp_dir)
        temp_bddl_dir = temp_root / "bddl"
        temp_init_dir = temp_root / "init"
        temp_bddl_dir.mkdir()
        temp_init_dir.mkdir()

        source_env_bddl_dir = runtime_root / "bddl_files" / "libero_object_env"
        for task_name in task_names:
            shutil.copy2(
                source_env_bddl_dir / f"{task_name}.bddl",
                temp_bddl_dir / f"{task_name}.bddl",
            )

        child_env = os.environ.copy()
        child_env["LIBERO_ROOT"] = str(source_root)
        child_env["LIBERO_CONFIG_PATH"] = str(config_path.parent)
        current_pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = os.pathsep.join(
            item for item in [str(source_root), current_pythonpath] if item
        )
        command = [
            sys.executable,
            str(generator),
            "--bddl_base_dir",
            str(temp_bddl_dir),
            "--output_dir",
            str(temp_init_dir),
            "--num_inits",
            str(init_count),
            "--height",
            "128",
            "--width",
            "128",
        ]
        print("[PRO] generating environment init states:", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=child_env)

        target_dir = runtime_root / "init_files" / "libero_object_env"
        target_dir.mkdir(parents=True, exist_ok=True)
        for task_name in task_names:
            generated = temp_init_dir / f"{task_name}.pruned_init"
            if not generated.is_file() or generated.stat().st_size == 0:
                raise RuntimeError(f"Init-state generator produced no file: {generated}")
            os.replace(generated, target_dir / generated.name)


def prepare_runtime(
    source_root: Path,
    runtime_root: Path,
    perturbations: list[str],
    task_ids: list[int],
    init_count: int,
    seed: int,
    skip_init_generation: bool,
) -> dict:
    source_root = source_root.resolve()
    runtime_root = runtime_root.resolve()
    task_names = _selected_tasks(task_ids)
    if not perturbations:
        raise ValueError("At least one PRO perturbation is required")
    unknown = [item for item in perturbations if item not in SUITE_BY_PERTURBATION]
    if unknown:
        raise ValueError(f"Unknown PRO perturbation(s): {unknown}")
    if init_count <= 0:
        raise ValueError(f"init_count must be positive, got {init_count}")

    runtime_root.mkdir(parents=True, exist_ok=True)
    config_path = _write_libero_config(source_root, runtime_root)
    runtime_bddl_root = runtime_root / "bddl_files"
    runtime_init_root = runtime_root / "init_files"
    runtime_bddl_root.mkdir(parents=True, exist_ok=True)
    runtime_init_root.mkdir(parents=True, exist_ok=True)

    source_bddl_root = source_root / "libero" / "libero" / "bddl_files"
    source_init_root = source_root / "libero" / "libero" / "init_files"
    generated_env_bddl = False
    changed_env_tasks: list[str] = []

    for perturbation in perturbations:
        suite = SUITE_BY_PERTURBATION[perturbation]
        runtime_bddl_dir = runtime_bddl_root / suite
        runtime_init_dir = runtime_init_root / suite
        if perturbation == "env":
            before = {
                task_name: (runtime_bddl_dir / f"{task_name}.bddl").is_file()
                for task_name in task_names
            }
            changed_env_tasks.extend(
                _generate_environment_bddl(
                    source_root,
                    runtime_bddl_dir,
                    task_names,
                    seed,
                )
            )
            generated_env_bddl = True
            # Missing BDDL files also need init states even if the generated
            # text happened to match a previous partial runtime.
            changed_env_tasks.extend(
                task_name for task_name, existed in before.items() if not existed
            )
            runtime_init_dir.mkdir(parents=True, exist_ok=True)
        else:
            _ensure_directory_link(source_bddl_root / suite, runtime_bddl_dir)
            _ensure_directory_link(source_init_root / suite, runtime_init_dir)

        _ensure_selected_files(
            runtime_bddl_dir,
            runtime_init_dir,
            task_names,
            require_init=perturbation != "env",
        )

    if generated_env_bddl:
        missing_env_init = []
        env_init_dir = runtime_init_root / "libero_object_env"
        for task_name in task_names:
            init_path = env_init_dir / f"{task_name}.pruned_init"
            if task_name in set(changed_env_tasks) or not init_path.is_file() or init_path.stat().st_size == 0:
                missing_env_init.append(task_name)
        if skip_init_generation:
            print(
                "[PRO] skipped init-state generation; missing env init files:",
                ", ".join(missing_env_init) or "none",
                flush=True,
            )
        else:
            _generate_missing_init_states(
                source_root,
                runtime_root,
                config_path,
                missing_env_init,
                init_count,
            )
        _ensure_selected_files(
            runtime_bddl_root / "libero_object_env",
            env_init_dir,
            task_names,
            require_init=not skip_init_generation,
        )

    metadata = {
        "source_root": str(source_root),
        "runtime_root": str(runtime_root),
        "config_path": str(config_path),
        "task_ids": task_ids,
        "task_names": task_names,
        "perturbations": perturbations,
        "suite_by_perturbation": {
            key: SUITE_BY_PERTURBATION[key] for key in perturbations
        },
        "environment_init_count_requested": init_count,
        "environment_bddl_sha256": {
            task_name: _sha256(
                runtime_bddl_root
                / "libero_object_env"
                / f"{task_name}.bddl"
            )
            for task_name in task_names
        }
        if generated_env_bddl
        else {},
    }
    (runtime_root / "runtime_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[PRO] runtime ready: {runtime_root}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--perturbations", default="env,swap,object,language,task")
    parser.add_argument("--task-ids", default="3,1,9")
    parser.add_argument("--init-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-init-generation",
        action="store_true",
        help="Only prepare BDDL/config; useful for CPU-only static checks.",
    )
    args = parser.parse_args()
    perturbations = _parse_csv(args.perturbations)
    task_ids = [int(item) for item in _parse_csv(args.task_ids)]
    prepare_runtime(
        source_root=args.source_root,
        runtime_root=args.runtime_root,
        perturbations=perturbations,
        task_ids=task_ids,
        init_count=args.init_count,
        seed=args.seed,
        skip_init_generation=args.skip_init_generation,
    )


if __name__ == "__main__":
    main()
