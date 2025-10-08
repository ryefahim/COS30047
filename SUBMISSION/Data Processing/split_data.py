import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path 

DATA = Path("data")

# Load defensively; keep labels small ints
df = pd.read_csv(DATA / "CombinedDataSet.csv",
                 dtype={"fake": "Int8", "id": "string", "text": "string"})


# Basic validations
required = {"id", "text", "fake"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")
if df["fake"].isna().any():
    raise ValueError("Found missing labels in 'fake' column.")
if not set(df["fake"].unique()) <= {0, 1}:
    raise ValueError(f"Unexpected label values: {df['fake'].unique().tolist()}")

# Cast to non-nullable for sklearn/Numpy
df["fake"] = df["fake"].astype("uint8")

# 80/20 train+val / test
temp_X, X_test, temp_y, y_test = train_test_split(
    df[["id", "text"]], df["fake"],
    test_size=0.20, random_state=42, stratify=df["fake"]
)


# from the remaining 80%, carve out 10% as val (i.e., 72/8/20 overall)
X_train, X_val, y_train, y_val = train_test_split(
    temp_X, temp_y,
    test_size=0.10, random_state=42, stratify=temp_y
)

# save splits
(DATA / "splits").mkdir(parents=True, exist_ok=True)
train = X_train.assign(fake=y_train.values)
val   = X_val.assign(fake=y_val.values)
test  = X_test.assign(fake=y_test.values)

train.to_csv(DATA / "splits" / "train.csv", index=False)
val.to_csv(DATA / "splits" / "val.csv", index=False)
test.to_csv(DATA / "splits" / "test.csv", index=False)

# quick sanity checks
print("Train:", train["fake"].value_counts(normalize=True).round(4).to_dict(), len(train))
print("Val  :", val["fake"].value_counts(normalize=True).round(4).to_dict(), len(val))
print("Test :", test["fake"].value_counts(normalize=True).round(4).to_dict(), len(test))