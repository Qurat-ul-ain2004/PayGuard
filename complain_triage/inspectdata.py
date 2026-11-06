import pandas as pd
from pathlib import Path
data_folder = Path(__file__).resolve().parent.parent / "data"
csv_files = [
    file for file in data_folder.glob("*.csv")
    if file.name.lower() != "creditcard.csv"
]

print("Available CSV files in the data folder:")
for file in csv_files:
    print(f"- {file.name}")

if len(csv_files) == 0:
    print("No CSV files found in the data folder.")

df = pd.read_csv(
    csv_files[0],
    nrows=5,
    low_memory=False
)
print("\nDataset columns:", df.columns.tolist())
print("\nDataset preview:")
print(
df.head()[
       [
           col for col in df.columns
           if "nara" not in col.lower()
       ]
].to_string(index=False)
)
