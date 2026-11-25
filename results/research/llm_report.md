# E7 · LLM confidence across English and Persian (qwen2.5:7b)

## Verbalised confidence

| Metric | English | Persian |
|---|---|---|
| Accuracy (%) | 96.9 | 82.8 |
| ECE | 0.078 | 0.101 |
| AURC | 0.003 | 0.038 |
| Coverage at 0.3 (%) | 100.0 | 100.0 |
| Error on auto-answers at 0.3 (%) | 3.1 | 17.2 |
| Unusable replies | 0 | 0 |

## Consistency confidence

| Metric | English | Persian |
|---|---|---|
| Accuracy (%) | 96.9 | 83.9 |
| ECE | 0.033 | 0.153 |
| AURC | 0.006 | 0.089 |
| Coverage at 0.3 (%) | 100.0 | 100.0 |
| Error on auto-answers at 0.3 (%) | 3.1 | 16.1 |
| Unusable replies | 0 | 0 |

## Classifier baseline (from experiments.py, mean over seeds)

| Metric | English | Persian |
|---|---|---|
| Accuracy (%) | 79.2 | 70.3 |
| ECE | 0.258 | 0.186 |
| AURC | 0.051 | 0.102 |
| Error on auto-answers at 0.3 (%) | 13.2 | 22.7 |

Note: the classifier is evaluated with cross-validation; the LLM is zero-shot on all messages.
