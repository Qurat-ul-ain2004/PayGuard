
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

# Load predictions and expected labels
results = pd.read_csv(folder / "triage_results.csv")
expected = pd.read_csv(folder / "expected_labels.csv")

# Match records by complaint ID
df = results.merge(
    expected,
    on="id",
    how="inner",
    validate="one_to_one"
)

if len(df) != len(expected) or len(df) != len(results):
    raise ValueError("Missing or unmatched complaint IDs")

# Compare department predictions
df["routing_correct"] = (
    df["department"] == df["expected_department"]
)

routing_accuracy = df["routing_correct"].mean()

# Convert human review values to booleans
def to_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() == "true":
        return True
    if str(value).strip().lower() == "false":
        return False
    raise ValueError(f"Invalid human review value: {value}")

df["predicted_review"] = df["needs_human_review"].apply(to_bool)
df["expected_review"] = df["expected_human_review"].apply(to_bool)

# Evaluate mandatory human-review detection
tp = ((df["predicted_review"]) & (df["expected_review"])).sum()
fn = ((~df["predicted_review"]) & (df["expected_review"])).sum()
fp = ((df["predicted_review"]) & (~df["expected_review"])).sum()

review_recall = tp / (tp + fn) if (tp + fn) else 0
review_precision = tp / (tp + fp) if (tp + fp) else 0

print("\n===== PAYGUARD TRIAGE EVALUATION =====")
print("Total complaints:", len(df))
print(f"Routing Accuracy: {routing_accuracy:.2%}")
print(f"Human Review Recall: {review_recall:.2%}")
print(f"Human Review Precision: {review_precision:.2%}")

print("\nIncorrectly Routed Complaints:")
wrong = df[~df["routing_correct"]]
print(
    wrong[
        ["id", "complaint", "department", "expected_department"]
    ].to_string(index=False)
)

print("\nMissed Human Review Cases:")
missed = df[df["expected_review"] & ~df["predicted_review"]]
print(
    missed[
        ["id", "complaint", "department"]
    ].to_string(index=False)
)
