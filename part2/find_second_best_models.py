import os
from pathlib import Path
from typing import Dict, List, Tuple
import re
from datetime import datetime


def parse_checkpoint_name(name: str) -> Tuple[datetime, float, str, int]:
    match = re.search(r"(best|results)_(\d+\.\d+)_(\d+)_single_ppo_(\d+\.\d+)", name)
    if match:
        checkpoint_type, timestamp_str, epoch_str, score_str = match.groups()
        timestamp = datetime.fromtimestamp(float(timestamp_str))
        score = float(score_str)
        epoch = int(epoch_str)
        return timestamp, score, checkpoint_type, epoch
    return None, None, None, None


def find_second_best_models(grid_search_run: str) -> List[Dict[str, Dict[str, str]]]:
    base_path = Path("runs") / grid_search_run
    if not base_path.exists():
        raise ValueError(f"Grid search run directory not found: {base_path}")

    results = []

    config_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("c")
    ]
    print(f"Found {len(config_dirs)} configuration directories")

    for config_dir in config_dirs:
        weights_dir = config_dir / "weights"
        if not weights_dir.exists():
            continue

        print(f"\nProcessing {config_dir.name}")
        best_checkpoints = []
        results_checkpoints = []

        for checkpoint in weights_dir.iterdir():
            if not checkpoint.is_dir():
                continue

            timestamp, score, checkpoint_type, epoch = parse_checkpoint_name(
                checkpoint.name
            )
            if timestamp and score and checkpoint_type:
                if checkpoint_type == "best":
                    best_checkpoints.append((checkpoint, score))
                elif checkpoint_type == "results":
                    results_checkpoints.append((checkpoint, epoch))
        config_name = config_dir.name
        has_arrows = "with_arrows" in config_name
        submission_name = (
            "submission_single.py" if has_arrows else "submission_single_no_arrows.py"
        )

        config_entry = {
            "submission_file": submission_name,
            "second_best": None,
            "second_last": None,
        }

        if len(best_checkpoints) >= 2:
            best_checkpoints.sort(key=lambda x: x[1], reverse=True)
            second_best = best_checkpoints[1]

            print(
                f"Best checkpoint (score: {best_checkpoints[0][1]}): {best_checkpoints[0][0]}"
            )
            print(f"Second best checkpoint (score: {second_best[1]}): {second_best[0]}")

            config_entry["second_best"] = str(second_best[0])
        else:
            raise ValueError(
                f"Not enough checkpoints found for {config_dir.name}. Expected at least 2 best checkpoints."
            )

        if len(results_checkpoints) >= 2:
            results_checkpoints.sort(key=lambda x: x[1])
            second_last_results = results_checkpoints[-2]

            print(
                f"Last results checkpoint (epoch: {results_checkpoints[-1][1]}): {results_checkpoints[-1][0]}"
            )
            print(
                f"Second last results checkpoint (epoch: {second_last_results[1]}): {second_last_results[0]}"
            )

            config_entry["second_last"] = str(second_last_results[0])
        else:
            raise ValueError(
                f"Not enough checkpoints found for {config_dir.name}. Expected at least 2 best and 2 results checkpoints."
            )

        results.append(config_entry)

    return results
