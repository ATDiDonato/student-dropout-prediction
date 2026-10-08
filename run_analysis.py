"""Run the complete staged student-dropout experiment from the repository root."""

from pathlib import Path

from src.experiment import run_all


if __name__ == "__main__":
    outputs = run_all(Path(__file__).resolve().parent)
    print("\nVerified held-out test metrics:")
    print(outputs["metrics"].to_string(index=False))
    print("\nLike-for-like metrics on the Stage 3-eligible test cohort:")
    print(outputs["common_cohort"].to_string(index=False))

