import os
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, Optional, Any, Union


def parse_filename(filename: str) -> Optional[Dict[str, Union[str, float, bool, int]]]:
    # Extract config, checkpoint and score from filename
    # Format: config__checkpoint____score.txt
    # Example: c2000_yKL_r512_with_arrows__results_1746903951.944895_980_single_ppo_79.8____43.64.txt
    parts = filename.split("____")
    if len(parts) != 2:
        print(f"Invalid filename format (no score separator): {filename}")
        return None

    config_checkpoint = parts[0]
    score_str = parts[1].split(".txt")[0]

    # Split config and checkpoint
    config_checkpoint_parts = config_checkpoint.split("__")
    if len(config_checkpoint_parts) != 2:
        print(f"Invalid filename format (no config/checkpoint separator): {filename}")
        return None

    config = config_checkpoint_parts[0]
    checkpoint = config_checkpoint_parts[1]

    try:
        score = float(score_str)
    except ValueError:
        print(f"Invalid score format: {score_str}")
        return None

    # Extract features from config
    has_arrows = "with_arrows" in config
    max_cycles_match = re.search(r"c(\d+)", config)
    rollout_length_match = re.search(r"r(\d+)", config)

    if not max_cycles_match or not rollout_length_match:
        print(f"Could not extract cycles or rollout length from config: {config}")
        return None

    max_cycles = int(max_cycles_match.group(1))
    rollout_length = int(rollout_length_match.group(1))
    has_kl = "yKL" in config

    return {
        "config": config,
        "checkpoint": checkpoint,
        "score": score,
        "has_arrows": has_arrows,
        "max_cycles": max_cycles,
        "rollout_length": rollout_length,
        "has_kl": has_kl,
    }


def analyze_results():
    # Read all files from eval_scores_1000
    scores_dir = Path("eval_scores_10000000")
    if not scores_dir.exists():
        print(f"Directory {scores_dir} not found!")
        return

    # Parse all files
    results = []
    for file in scores_dir.glob("*.txt"):
        result = parse_filename(file.name)
        if result:
            results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv("eval_analysis.csv", index=False)

    # Print analysis
    print("\n=== Evaluation Results Analysis ===\n")

    # Overall statistics
    print("Overall Statistics:")
    print(f"Number of configurations: {len(df['config'].unique())}")
    print(f"Average score: {df['score'].mean():.2f}")
    print(f"Best score: {df['score'].max():.2f}")
    print(f"Worst score: {df['score'].min():.2f}\n")

    # Compare with/without arrows
    print("Comparison with/without arrows:")
    arrow_stats = df.groupby("has_arrows")["score"].agg(["mean", "std", "count"])
    print(arrow_stats)
    print()

    # Compare KL loss
    print("Comparison with/without KL loss:")
    kl_stats = df.groupby("has_kl")["score"].agg(["mean", "std", "count"])
    print(kl_stats)
    print()

    # Compare rollout lengths
    print("Comparison by rollout length:")
    rollout_stats = df.groupby("rollout_length")["score"].agg(["mean", "std", "count"])
    print(rollout_stats)
    print()

    # Compare max cycles
    print("Comparison by max cycles:")
    cycles_stats = df.groupby("max_cycles")["score"].agg(["mean", "std", "count"])
    print(cycles_stats)
    print()

    # Top 5 configurations
    print("Top 5 configurations:")
    top_configs = df.nlargest(5, "score")[["config", "checkpoint", "score"]]
    print(top_configs)
    print()

    # Save detailed analysis to file
    with open(f"{scores_dir}/eval_analysis.txt", "w") as f:
        f.write("=== Detailed Evaluation Analysis ===\n\n")
        f.write("Overall Statistics:\n")
        f.write(f"Number of configurations: {len(df['config'].unique())}\n")
        f.write(f"Average score: {df['score'].mean():.2f}\n")
        f.write(f"Best score: {df['score'].max():.2f}\n")
        f.write(f"Worst score: {df['score'].min():.2f}\n\n")

        f.write("Comparison with/without arrows:\n")
        f.write(str(arrow_stats) + "\n\n")

        f.write("Comparison with/without KL loss:\n")
        f.write(str(kl_stats) + "\n\n")

        f.write("Comparison by rollout length:\n")
        f.write(str(rollout_stats) + "\n\n")

        f.write("Comparison by max cycles:\n")
        f.write(str(cycles_stats) + "\n\n")

        f.write("Top 5 configurations:\n")
        f.write(str(top_configs) + "\n")


if __name__ == "__main__":
    analyze_results()
