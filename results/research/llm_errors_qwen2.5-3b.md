# E7 error analysis (qwen2.5:3b)

## How the errors look, per language

| | English | Persian |
|---|---|---|
| Wrong answers | 6 | 22 |
| … stated confidence ≥ 90% | 4 | 20 |
| … all sampled answers agreed on the same wrong intent | 1 | 16 |

## English: which intents are confused

| True intent | Predicted | Count |
|---|---|---|
| technical_problem | human_agent | 2 |
| technical_problem | billing_issue | 2 |
| technical_problem | greeting | 1 |
| human_agent | billing_issue | 1 |

## English: every wrong answer (most confident first)

| Message | True | Predicted | Stated conf. | Sampled answers |
|---|---|---|---|---|
| Password reset email never arrives | technical_problem | human_agent | 1.00 | greeting×3, human_agent×2 |
| my account is locked | technical_problem | human_agent | 1.00 | human_agent×4, technical_problem×1 |
| error 500 when I pay | technical_problem | billing_issue | 1.00 | billing_issue×5 |
| the checkout page gives me an error | technical_problem | billing_issue | 1.00 | billing_issue×4, technical_problem×1 |
| verification code not received | technical_problem | greeting | 0.80 | greeting×4, human_agent×1 |
| I want to file a formal complaint | human_agent | billing_issue | 0.80 | billing_issue×4, human_agent×1 |

## Persian: which intents are confused

| True intent | Predicted | Count |
|---|---|---|
| technical_problem | billing_issue | 5 |
| human_agent | greeting | 3 |
| refund_request | billing_issue | 2 |
| technical_problem | greeting | 2 |
| order_status | change_address | 1 |
| refund_request | change_address | 1 |
| technical_problem | order_status | 1 |
| change_address | technical_problem | 1 |
| cancel_order | greeting | 1 |
| cancel_order | order_status | 1 |
| human_agent | refund_request | 1 |
| human_agent | order_status | 1 |
| human_agent | technical_problem | 1 |
| human_agent | billing_issue | 1 |

## Persian: every wrong answer (most confident first)

| Message | True | Predicted | Stated conf. | Sampled answers |
|---|---|---|---|---|
| کد رهگیری سفارشم رو می‌خوام | order_status | change_address | 1.00 | change_address×3, cancel_order×2 |
| لطفا هزینه سفارش ۱۰۲۳۴ رو برگردونید | refund_request | billing_issue | 1.00 | billing_issue×5 |
| محصول آسیب دیده بود | refund_request | billing_issue | 1.00 | billing_issue×5 |
| می‌خوام این رو پس بفرستم | refund_request | change_address | 1.00 | change_address×5 |
| ایمیل بازیابی رمز نمیاد | technical_problem | billing_issue | 1.00 | billing_issue×5 |
| موقع پرداخت سایت خطا میده | technical_problem | billing_issue | 1.00 | billing_issue×5 |
| صفحه باز نمیشه | technical_problem | greeting | 1.00 | greeting×5 |
| اسباب‌کشی کردم آدرسم رو به‌روز کنید | change_address | technical_problem | 1.00 | technical_problem×5 |
| پشیمون شدم لغوش کنید | cancel_order | greeting | 1.00 | greeting×5 |
| سفارش ۱۰۵۶۷ رو نفرستید | cancel_order | order_status | 1.00 | order_status×5 |
| وصلم کنید به اپراتور | human_agent | greeting | 1.00 | greeting×5 |
| می‌خوام شکایت رسمی ثبت کنم | human_agent | refund_request | 1.00 | refund_request×5 |
| ربات نمی‌خوام آدم می‌خوام | human_agent | greeting | 1.00 | greeting×5 |
| رمزم رو فراموش کردم ایمیل نمیاد | technical_problem | greeting | 1.00 | greeting×4, billing_issue×1 |
| با اپراتور وصلم کنید | human_agent | greeting | 1.00 | greeting×5 |
| خطای ۵۰۰ موقع پرداخت | technical_problem | billing_issue | 0.95 | billing_issue×5 |
| حسابم قفل شده | technical_problem | billing_issue | 0.90 | billing_issue×5 |
| لطفا پیگیری ویژه کنید | human_agent | order_status | 0.90 | order_status×3, greeting×1, billing_issue×1 |
| ازتون شکایت می‌کنم | human_agent | technical_problem | 0.90 | technical_problem×4, greeting×1 |
| میخوام شکایت کنم | human_agent | billing_issue | 0.90 | refund_request×5 |
| کد تایید برام نیومد | technical_problem | billing_issue | 0.85 | billing_issue×5 |
| نمی‌تونم کالا به سبد اضافه کنم | technical_problem | order_status | 0.80 | order_status×4, billing_issue×1 |
