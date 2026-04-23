import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

CSV = Path("MultiagentSystem/agents/unbias_agent/CarpeNoctom_dataset_h1.csv")

df = pd.read_csv(CSV)

df["pred"] = pd.NA
df.loc[df["confidence"] > 50, "pred"] = "LONG"
df.loc[df["confidence"] < 50, "pred"] = "SHORT"

mask = df["pred"].notna() & df["y_true"].notna()
y_true = df.loc[mask, "y_true"]
y_pred = df.loc[mask, "pred"]

labels = ["LONG", "SHORT"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{l}" for l in labels],
    columns=[f"pred_{l}" for l in labels],
)

print(f"=== rule: confidence > 50 -> LONG, confidence < 50 -> SHORT ===")
print(f"used rows: {mask.sum()} / {len(df)} (skipped: confidence == 50 or y_true=NaN)")
print()
print("y_true distribution:")
print(y_true.value_counts())
print()
print("y_pred distribution:")
print(y_pred.value_counts())
print()
print(cm_df.to_string())
print()
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
