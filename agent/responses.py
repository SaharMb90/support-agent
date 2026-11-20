"""Reply templates in English and Persian. Used directly, or as grounded drafts for an LLM to rephrase."""

T = {
 "ask_order_id": {"en": "Sure — what's your 5-digit order number?",
                  "fa": "حتماً — لطفاً شماره سفارش ۵ رقمی‌تون رو بفرستید."},
 "order_status": {"en": "Order {order_id} ({item}) is {status}. Expected delivery: {eta}.",
                  "fa": "سفارش {order_id} ({item}) در وضعیت «{status}» است. تاریخ تحویل: {eta}."},
 "refund_request": {"en": "I've opened refund request {ticket_id} for order {order_id} ({item}). You'll get an email within 24 hours.",
                    "fa": "درخواست مرجوعی {ticket_id} برای سفارش {order_id} ({item}) ثبت شد. ظرف ۲۴ ساعت ایمیل دریافت می‌کنید."},
 "cancel_ok": {"en": "Order {order_id} has been cancelled. The refund goes back to your card in 3–5 days.",
               "fa": "سفارش {order_id} لغو شد. مبلغ ظرف ۳ تا ۵ روز به کارت شما برمی‌گردد."},
 "cancel_no": {"en": "Order {order_id} is already {status}, so it can't be cancelled. I can start a return instead.",
               "fa": "سفارش {order_id} در وضعیت «{status}» است و قابل لغو نیست. می‌توانم درخواست مرجوعی ثبت کنم."},
 "change_address": {"en": "Ticket {ticket_id}: please reply with the new address for order {order_id} and we'll update it before dispatch.",
                    "fa": "تیکت {ticket_id}: لطفاً آدرس جدید سفارش {order_id} را بفرستید تا قبل از ارسال اصلاح شود."},
 "billing_issue": {"en": "I've logged billing ticket {ticket_id}. Our finance team will check the charge and reply within 1 business day.",
                   "fa": "تیکت مالی {ticket_id} ثبت شد. واحد مالی تراکنش را بررسی و ظرف ۱ روز کاری پاسخ می‌دهد."},
 "technical_problem": {"en": "Sorry about that. Ticket {ticket_id} is open. Meanwhile, try updating the app or clearing your browser cache.",
                       "fa": "متأسفیم. تیکت {ticket_id} ثبت شد. تا آن زمان اپ را به‌روز کنید یا کش مرورگر را پاک کنید."},
 "greeting": {"en": "Hi! I can help with orders, refunds, cancellations, billing and login problems.",
              "fa": "سلام! در پیگیری سفارش، مرجوعی، لغو، مسائل مالی و مشکلات ورود کمکتان می‌کنم."},
 "handoff": {"en": "I'm connecting you to a human agent now; they'll see our conversation so you won't need to repeat yourself.",
             "fa": "شما را به کارشناس پشتیبانی وصل می‌کنم؛ گفتگوی ما را می‌بینند و لازم نیست چیزی را تکرار کنید."},
 "order_not_found": {"en": "I couldn't find order {order_id}, so I'm passing you to a human agent to check it.",
                     "fa": "سفارش {order_id} پیدا نشد؛ شما را به کارشناس پشتیبانی وصل می‌کنم تا بررسی کند."},
}


STATUS_FA = {"shipped": "ارسال شده", "processing": "در حال پردازش", "delivered": "تحویل شده", "cancelled": "لغو شده"}


def render(key: str, lang: str, **slots) -> str:
    if lang == "fa" and "status" in slots:
        slots["status"] = STATUS_FA.get(slots["status"], slots["status"])
    return T[key][lang].format(**slots)
