# Evaluation report

| Metric | Value |
|---|---|
| test_examples | 30 |
| intent_accuracy | 0.967 |
| routing_accuracy | 1.0 |
| intent_accuracy_by_language | {'en': 1.0, 'fa': 0.923} |
| train_5fold_cv_accuracy | 0.606 |
| latency_ms_p50 | 1.25 |
| latency_ms_p95 | 1.36 |
| llm_used | False |

## Errors

- `فاکتور سفارشم رو ندارم` — expected billing_issue/auto, got order_status/auto (conf 0.334)
