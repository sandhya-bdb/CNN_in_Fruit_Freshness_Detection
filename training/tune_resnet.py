import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from train_resnet import TrainingConfig, run_training


SEARCH_SPACE = {
    "lr": [1e-4, 5e-5, 1e-5],
    "dropout": [0.2, 0.3, 0.4, 0.5],
    "weight_decay": [0.0, 1e-5, 1e-4],
    "batch_size": [16, 32, 64],
}



def parse_args():
    parser = argparse.ArgumentParser(description="Random search tuner for ResNet50 fruit freshness model")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="tuning_runs")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--preset",
        choices=["tuned", "realistic_eval"],
        default="realistic_eval",
    )
    return parser.parse_args()



def sample_trial(rng: random.Random):
    return {
        "lr": rng.choice(SEARCH_SPACE["lr"]),
        "dropout": rng.choice(SEARCH_SPACE["dropout"]),
        "weight_decay": rng.choice(SEARCH_SPACE["weight_decay"]),
        "batch_size": rng.choice(SEARCH_SPACE["batch_size"]),
    }


if __name__ == "__main__":
    args = parse_args()
    rng = random.Random(args.seed)

    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    all_results = []

    for trial in range(1, args.trials + 1):
        sampled = sample_trial(rng)
        trial_dir = root_out / f"trial_{trial:02d}"

        cfg = TrainingConfig(
            dataset_dir=args.dataset_dir,
            output_dir=str(trial_dir),
            preset=args.preset,
            epochs=args.epochs,
            batch_size=sampled["batch_size"],
            lr=sampled["lr"],
            weight_decay=sampled["weight_decay"],
            dropout=sampled["dropout"],
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed + trial,
        )

        print(f"\nTrial {trial}/{args.trials} -> {sampled}")
        result = run_training(cfg)

        all_results.append(
            {
                "trial": trial,
                "hyperparameters": sampled,
                "best_val_acc": result["best_val_acc"],
                "test_accuracy": result["test_accuracy"],
                "best_epoch": result["best_epoch"],
                "config": asdict(cfg),
            }
        )

    all_results.sort(key=lambda x: x["best_val_acc"], reverse=True)

    summary_path = root_out / "tuning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    best = all_results[0] if all_results else None
    if best:
        print("\nBest trial:")
        print(best)
    print(f"Saved summary to: {summary_path}")
