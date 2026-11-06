
import pandas as pd
import time
from pathlib import Path
from triage import triage_complaint

# File locations
project_root = Path(__file__).resolve().parent.parent
input_path = project_root / "data" / "cfpb_sample_400.csv"
output_path = project_root / "data" / "cfpb_triage_results.csv"

# Read the prepared CFPB sample
df = pd.read_csv(input_path)

# Start with only 5 complaints
df = df.head(300)

results = []

for _, row in df.iterrows():
    complaint_id = row["Complaint ID"]
    complaint = row["Consumer complaint narrative"]

    print(f"Processing complaint {complaint_id}...")

    start_time = time.perf_counter()

    try:
        result = triage_complaint(complaint)
        status = "success"

    except Exception as error:
        print(f"Failed: {type(error).__name__}")

        # Safe fallback if the LLM fails
        result = {
            "category": "general_other",
            "urgency": "high",
            "summary": "Automated triage unavailable.",
            "requires_human": True,
            "confidence_note": "API or validation failure; manual review required."
        }
        status = "fallback"

    latency = time.perf_counter() - start_time

    results.append({
        "complaint_id": complaint_id,
        "category": result["category"],
        "urgency": result["urgency"],
        "summary": result["summary"],
        "requires_human": result["requires_human"],
        "confidence_note": result["confidence_note"],
        "status": status,
        "latency_seconds": round(latency, 2)
    })

# Save results
pd.DataFrame(results).to_csv(output_path, index=False)

print("\nBatch processing complete!")
print("Complaints processed:", len(results))
print("Successful:", sum(r["status"] == "success" for r in results))
print("Fallbacks:", sum(r["status"] == "fallback" for r in results))
print("Results saved to:", output_path)
