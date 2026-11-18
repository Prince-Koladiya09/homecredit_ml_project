"""Validate packaged data, saved model and submission artifacts."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))  # the estimator was serialized with features as a top-level module

train = pd.read_csv(ROOT/'input/train.csv')
test = pd.read_csv(ROOT/'input/test.csv')
sample = pd.read_csv(ROOT/'input/sample_submission.csv')
sub = pd.read_csv(ROOT/'outputs/submission.csv')
assert train.shape == (171202, 34), train.shape
assert test.shape == (61500, 33), test.shape
assert train.SK_ID_CURR.is_unique and test.SK_ID_CURR.is_unique
assert set(train.SK_ID_CURR).isdisjoint(set(test.SK_ID_CURR))
assert sample.SK_ID_CURR.equals(test.SK_ID_CURR)
assert sub.shape == (61500, 2)
assert sub.columns.tolist() == ['SK_ID_CURR', 'TARGET']
assert sub.SK_ID_CURR.equals(sample.SK_ID_CURR)
assert sub.SK_ID_CURR.is_unique
assert sub.TARGET.notna().all() and np.isfinite(sub.TARGET).all()
assert sub.TARGET.between(0, 1).all()
model = joblib.load(ROOT/'models/homecredit_model.joblib')
metrics = pd.read_json(ROOT/'outputs/validation_metrics.json', typ='series')
assert 'roc_auc' in metrics and 0 <= float(metrics['roc_auc']) <= 1
assert len(list((ROOT/'outputs/eda').glob('*.png'))) >= 10
print('PASS: data schema, IDs, submission, prediction range, saved model load and EDA figures.')
print('Saved model:', type(model).__name__)
print('Submission rows:', len(sub), '| prediction range:', float(sub.TARGET.min()), float(sub.TARGET.max()))
print('Recorded validation ROC-AUC:', float(metrics['roc_auc']))
