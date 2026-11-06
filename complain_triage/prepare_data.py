
import pandas as pd
from pathlib import Path

data_folder = Path(__file__).resolve().parent.parent / "data"


df = pd.read_csv(
    data_folder / "CCDB_Export_15_January_2026_through_February_2026.csv",
    usecols=[
        "Complaint ID",
        "Date received",
        "Product",
        "Issue",
        "Consumer complaint narrative"
    ]
)

print("Original rows:", len(df))


df = df.dropna(subset=["Consumer complaint narrative"])

df["Consumer complaint narrative"] = (
    df["Consumer complaint narrative"].str.strip()
)

df = df[df["Consumer complaint narrative"].str.len() >= 30]

df = df.drop_duplicates(subset=["Complaint ID"])
df = df.drop_duplicates(subset=["Consumer complaint narrative"])

print("Valid complaints:", len(df))


sample = df.sample(n=400, random_state=42)

sample.to_csv(data_folder / "cfpb_sample_400.csv", index=False)

print("Sample saved successfully!")
print("Selected complaints:", len(sample))
print("Date range:", sample["Date received"].min(),
      "to", sample["Date received"].max())
