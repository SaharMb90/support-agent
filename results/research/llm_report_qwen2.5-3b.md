# E7 · LLM confidence across English and Persian (qwen2.5:3b)

## Verbalised confidence

| Metric | English | Persian |
|---|---|---|
| Accuracy (%) | 93.8 | 76.3 |
| ECE | 0.060 | 0.219 |
| AURC | 0.037 | 0.184 |
| Coverage at 0.3 (%) | 100.0 | 100.0 |
| Error on auto-answers at 0.3 (%) | 6.2 | 23.7 |
| Unusable replies | 0 | 0 |
| Threshold for ≤10% auto-answer error | 0.80 | none reaches it |
| Coverage at that threshold (%) | 100.0 | 0.0 |

## Consistency confidence

| Metric | English | Persian |
|---|---|---|
| Accuracy (%) | 93.8 | 76.3 |
| ECE | 0.047 | 0.209 |
| AURC | 0.012 | 0.164 |
| Coverage at 0.3 (%) | 100.0 | 100.0 |
| Error on auto-answers at 0.3 (%) | 6.2 | 23.7 |
| Unusable replies | 0 | 0 |
| Threshold for ≤10% auto-answer error | 0.60 | none reaches it |
| Coverage at that threshold (%) | 100.0 | 0.0 |

## Classifier baseline (from experiments.py, mean over seeds)

| Metric | English | Persian |
|---|---|---|
| Accuracy (%) | 79.2 | 70.3 |
| ECE | 0.258 | 0.186 |
| AURC | 0.051 | 0.102 |
| Error on auto-answers at 0.3 (%) | 13.2 | 22.7 |
| Threshold for ≤10% auto-answer error | 0.35 | 0.45 |
| Coverage at that threshold (%) | 76.5 | 57.6 |

Note: the classifier is evaluated with cross-validation; the LLM is zero-shot on all messages.
