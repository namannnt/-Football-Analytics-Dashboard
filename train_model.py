"""CLI for chronological training and evaluation of the pace classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from ml_dataset import MODEL_FEATURES, TARGET, build_ml_dataset, chronological_split, read_serving_matches
from model import save_metadata, score_on_pace, train_on_pace_model


def evaluate(truth, probabilities) -> dict[str, object]:
    predicted = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(truth, predicted),
        "precision": precision_score(truth, predicted, zero_division=0),
        "recall": recall_score(truth, predicted, zero_division=0),
        "f1": f1_score(truth, predicted, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
    }
    metrics["roc_auc"] = roc_auc_score(truth, probabilities) if truth.nunique() == 2 else None
    return metrics


def train(matches: pd.DataFrame, model_path: str, metadata_path: str) -> dict[str, object]:
    dataset = build_ml_dataset(matches)
    training, testing = chronological_split(dataset)
    model = train_on_pace_model(training, model_path)
    scored = score_on_pace(model, testing)
    metrics = evaluate(scored[TARGET], scored.on_pace_probability)
    model_version = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:12]
    metadata = {
        "model_version": model_version,
        "label_definition": "Top half of final season table, ordered by points, goal difference, goals scored, then team name",
        "training_policy": "First-half pre-match observations; newest season held out",
        "features": MODEL_FEATURES,
        "training_rows": len(training),
        "test_rows": len(testing),
        "training_seasons": sorted(training.season.astype(str).unique().tolist()),
        "test_seasons": sorted(testing.season.astype(str).unique().tolist()),
        "demonstration_only": len(testing) < 100,
        "metrics": metrics,
    }
    save_metadata(metadata, metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("demo", "postgres", "csv"), default="demo")
    parser.add_argument("--matches", help="Match CSV used when --source=csv")
    parser.add_argument("--model", default="artifacts/on_pace.joblib")
    parser.add_argument("--metadata", default="artifacts/on_pace.metadata.json")
    args = parser.parse_args()
    if args.source == "postgres":
        matches = read_serving_matches()
    elif args.source == "csv":
        if not args.matches:
            parser.error("--matches is required when --source=csv")
        matches = pd.read_csv(args.matches)
    else:
        matches = pd.read_csv(Path(__file__).resolve().parent / "data" / "demo_matches.csv")
    metadata = train(matches, args.model, args.metadata)
    print(json.dumps(metadata, indent=2))
    if metadata["demonstration_only"]:
        print("Metrics are demonstration-only because the held-out sample has fewer than 100 observations.")


if __name__ == "__main__":
    main()
