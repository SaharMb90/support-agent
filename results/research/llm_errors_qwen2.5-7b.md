# E7 error analysis (qwen2.5:7b)

## How the errors look, per language

| | English | Persian |
|---|---|---|
| Wrong answers | 3 | 16 |
| … stated confidence ≥ 90% | 0 | 3 |
| … all sampled answers agreed on the same wrong intent | 2 | 13 |

## English: which intents are confused

| True intent | Predicted | Count |
|---|---|---|
| technical_problem | billing_issue | 1 |
| human_agent | technical_problem | 1 |
| human_agent | billing_issue | 1 |

## English: every wrong answer (most confident first)

| Message | True | Predicted | Stated conf. | Sampled answers |
|---|---|---|---|---|
| error 500 when I pay | technical_problem | billing_issue | 0.85 | billing_issue×4, technical_problem×1 |
| I want to file a formal complaint | human_agent | technical_problem | 0.70 | technical_problem×5 |
| I'm going to sue you | human_agent | billing_issue | 0.70 | billing_issue×5 |

## Persian: which intents are confused

| True intent | Predicted | Count |
|---|---|---|
| technical_problem | billing_issue | 6 |
| human_agent | refund_request | 2 |
| order_status | change_address | 1 |
| change_address | billing_issue | 1 |
| cancel_order | human_agent | 1 |
| cancel_order | order_status | 1 |
| greeting | human_agent | 1 |
| human_agent | order_status | 1 |
| billing_issue | order_status | 1 |
| human_agent | billing_issue | 1 |

## Persian: every wrong answer (most confident first)

| Message | True | Predicted | Stated conf. | Sampled answers |
|---|---|---|---|---|
| لطفا پیگیری ویژه کنید | human_agent | order_status | 1.00 | order_status×5 |
| موقع پرداخت سایت خطا میده | technical_problem | billing_issue | 0.95 | billing_issue×5 |
| می‌خوام شکایت رسمی ثبت کنم | human_agent | refund_request | 0.95 | refund_request×5 |
| کد تایید برام نیومد | technical_problem | billing_issue | 0.85 | billing_issue×5 |
| خطای ۵۰۰ موقع پرداخت | technical_problem | billing_issue | 0.85 | billing_issue×5 |
| پشیمون شدم لغوش کنید | cancel_order | human_agent | 0.85 | human_agent×5 |
| سفارش ۱۰۵۶۷ رو نفرستید | cancel_order | order_status | 0.85 | order_status×4, cancel_order×1 |
| سلام کسی هست؟ | greeting | human_agent | 0.85 | human_agent×5 |
| فاکتور سفارشم رو ندارم | billing_issue | order_status | 0.85 | order_status×5 |
| مرسوله من کی تحویل داده میشه | order_status | change_address | 0.75 | change_address×5 |
| ایمیل بازیابی رمز نمیاد | technical_problem | billing_issue | 0.70 | billing_issue×3, technical_problem×2 |
| حسابم قفل شده | technical_problem | billing_issue | 0.70 | billing_issue×5 |
| کد پستی سفارشم غلطه | change_address | billing_issue | 0.70 | change_address×5 |
| ازتون شکایت می‌کنم | human_agent | refund_request | 0.70 | refund_request×5 |
| رمزم رو فراموش کردم ایمیل نمیاد | technical_problem | billing_issue | 0.70 | billing_issue×5 |
| میخوام شکایت کنم | human_agent | billing_issue | 0.70 | billing_issue×5 |
