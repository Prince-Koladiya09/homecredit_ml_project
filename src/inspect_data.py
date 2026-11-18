"""Generate a compact, reproducible dataset profile using only competition files."""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
train = pd.read_csv(ROOT / "input/train.csv")
test = pd.read_csv(ROOT / "input/test.csv")
profile = {
    "train_shape": list(train.shape), "test_shape": list(test.shape),
    "target_counts": {str(k): int(v) for k, v in train.TARGET.value_counts().sort_index().items()},
    "target_rate": float(train.TARGET.mean()),
    "train_duplicate_ids": int(train.SK_ID_CURR.duplicated().sum()),
    "test_duplicate_ids": int(test.SK_ID_CURR.duplicated().sum()),
    "train_test_id_overlap": int(len(set(train.SK_ID_CURR) & set(test.SK_ID_CURR))),
    "missing_fraction": train.isna().mean().sort_values(ascending=False).round(4).to_dict(),
    "train_test_column_match_excluding_target": set(train.columns)-{"TARGET"} == set(test.columns),
}
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs/data_profile.json").write_text(json.dumps(profile, indent=2))
print(json.dumps(profile, indent=2))
