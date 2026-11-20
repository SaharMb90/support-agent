"""Builds the bilingual (English/Persian) intent dataset used for training and evaluation."""
import json, pathlib

TRAIN = {
 "order_status": [
  "Where is my order?", "Can you track order 10234 for me?", "My package hasn't arrived yet",
  "When will my order be delivered?", "What's the status of order #10567?", "Has my order shipped?",
  "I ordered last week and still nothing", "Is order 10891 on the way?", "tracking info for my delivery please",
  "how long until my parcel arrives",
  "سفارشم کجاست؟", "وضعیت سفارش ۱۰۲۳۴ رو بگید", "بسته‌ام هنوز نرسیده", "سفارشم کی می‌رسه؟",
  "سفارش شماره ۱۰۵۶۷ ارسال شده؟", "کد رهگیری سفارشم رو می‌خوام", "هفته پیش سفارش دادم هنوز خبری نیست",
  "سفارش من در چه مرحله‌ایه؟", "مرسوله من کی تحویل داده میشه", "پیگیری سفارش",
 ],
 "refund_request": [
  "I want a refund", "Can I get my money back for order 10234?", "The item is broken, I want to return it",
  "How do I return a product?", "Please refund my last purchase", "I received the wrong item and want a refund",
  "return request for order 10567", "the product is damaged, money back please", "I'd like to send this back",
  "refund for a defective item",
  "می‌خوام پولم رو پس بگیرم", "کالا خراب رسیده می‌خوام مرجوع کنم", "چطوری کالا رو برگردونم؟",
  "لطفا هزینه سفارش ۱۰۲۳۴ رو برگردونید", "کالای اشتباه برام فرستادید پولم رو می‌خوام",
  "درخواست مرجوعی دارم", "محصول آسیب دیده بود", "استرداد وجه سفارشم", "می‌خوام این رو پس بفرستم",
  "بازگشت کالا",
 ],
 "billing_issue": [
  "I was charged twice", "Why is there an extra charge on my card?", "My invoice is wrong",
  "payment failed but money was taken", "I need a copy of my invoice", "the price I paid is different from the website",
  "double charge on my account", "can you explain this charge", "billing problem with my last payment",
  "I paid but the order says unpaid",
  "دو بار از حسابم پول کم شده", "این مبلغ اضافه برای چیه؟", "فاکتورم اشتباهه", "پرداخت ناموفق بود ولی پول کسر شد",
  "یه نسخه از فاکتور می‌خوام", "مبلغی که پرداختم با قیمت سایت فرق داره", "مشکل پرداخت دارم",
  "پول دادم ولی سفارش پرداخت نشده نشون میده", "کسر وجه اضافه", "صورتحساب",
 ],
 "technical_problem": [
  "I can't log in to my account", "The app keeps crashing", "Password reset email never arrives",
  "the website shows an error at checkout", "I can't add items to my cart", "the page won't load",
  "my account is locked", "verification code not received", "the app is very slow", "error 500 when I pay",
  "نمی‌تونم وارد حسابم بشم", "اپلیکیشن مدام بسته میشه", "ایمیل بازیابی رمز نمیاد", "موقع پرداخت سایت خطا میده",
  "نمی‌تونم کالا به سبد اضافه کنم", "صفحه باز نمیشه", "حسابم قفل شده", "کد تایید برام نیومد",
  "برنامه خیلی کنده", "خطای ۵۰۰ موقع پرداخت",
 ],
 "change_address": [
  "I need to change my delivery address", "Can I update the shipping address for order 10234?",
  "wrong address on my order", "please deliver to a different address", "I moved, update my address",
  "change address for my order", "can you send it to my office instead", "fix the address on order 10891",
  "the postcode on my order is wrong", "update delivery location",
  "می‌خوام آدرس تحویل رو عوض کنم", "آدرس سفارش ۱۰۲۳۴ رو تغییر بدید", "آدرس سفارشم اشتباهه",
  "لطفا به یه آدرس دیگه بفرستید", "اسباب‌کشی کردم آدرسم رو به‌روز کنید", "تغییر آدرس سفارش",
  "به جای خونه بفرستید محل کارم", "آدرس سفارش ۱۰۸۹۱ رو درست کنید", "کد پستی سفارشم غلطه", "ویرایش آدرس ارسال",
 ],
 "cancel_order": [
  "Cancel my order", "I want to cancel order 10234", "please stop my order before it ships",
  "I changed my mind, cancel it", "how do I cancel an order", "cancel the order I just placed",
  "don't send order 10567", "I ordered by mistake, cancel please", "cancellation request", "can I still cancel?",
  "سفارشم رو لغو کنید", "می‌خوام سفارش ۱۰۲۳۴ رو کنسل کنم", "قبل از ارسال سفارش رو متوقف کنید", "پشیمون شدم لغوش کنید",
  "چطوری سفارش رو لغو کنم", "سفارشی که الان دادم رو کنسل کنید", "سفارش ۱۰۵۶۷ رو نفرستید",
  "اشتباهی سفارش دادم لغو کنید", "درخواست لغو", "هنوز میشه کنسل کرد؟",
 ],
 "greeting": [
  "hi", "hello", "good morning", "hey there", "thanks!", "thank you so much", "hi, are you there?", "bye",
  "good evening", "thanks for your help",
  "سلام", "درود", "صبح بخیر", "سلام وقت بخیر", "ممنون", "خیلی ممنونم", "سلام کسی هست؟", "خداحافظ",
  "عصر بخیر", "مرسی از کمکتون",
 ],
 "human_agent": [
  "I want to talk to a human", "connect me to a real person", "let me speak with an agent",
  "this is useless, get me a manager", "I want to file a formal complaint", "can I call support?",
  "I need a person not a bot", "transfer me to customer service", "escalate this please", "I'm going to sue you",
  "می‌خوام با یه آدم صحبت کنم", "وصلم کنید به اپراتور", "با پشتیبان انسانی کار دارم", "این به درد نمی‌خوره مدیر رو می‌خوام",
  "می‌خوام شکایت رسمی ثبت کنم", "شماره تماس پشتیبانی رو بدید", "ربات نمی‌خوام آدم می‌خوام", "انتقالم بدید به کارشناس",
  "لطفا پیگیری ویژه کنید", "ازتون شکایت می‌کنم",
 ],
}

# Held-out test set: (text, expected_intent, expected_route)
TEST = [
 ("hey, where's my stuff? order 10234", "order_status", "auto"),
 ("Did order 10891 ship yet?", "order_status", "auto"),
 ("tracking for 10567 pls", "order_status", "auto"),
 ("سفارش ۱۰۸۹۱ کجاست", "order_status", "auto"),
 ("بسته من کی میرسه؟ سفارش ۱۰۵۶۷", "order_status", "auto"),
 ("my order 99999 hasn't come", "order_status", "handoff"),
 ("the blender arrived cracked, I want my money back, order 10234", "refund_request", "auto"),
 ("return order 10891 please", "refund_request", "auto"),
 ("کالای سفارش ۱۰۵۶۷ معیوب بود پولم رو برگردونید", "refund_request", "auto"),
 ("میخوام مرجوع کنم", "refund_request", "auto"),
 ("you took money from my card two times", "billing_issue", "auto"),
 ("there's a charge I don't recognize", "billing_issue", "auto"),
 ("دوبار پول ازم کم کردید", "billing_issue", "auto"),
 ("فاکتور سفارشم رو ندارم", "billing_issue", "auto"),
 ("login doesn't work", "technical_problem", "auto"),
 ("the checkout page gives me an error", "technical_problem", "auto"),
 ("رمزم رو فراموش کردم ایمیل نمیاد", "technical_problem", "auto"),
 ("اپ باز نمیشه", "technical_problem", "auto"),
 ("send order 10234 to a new address", "change_address", "auto"),
 ("my address is wrong on the order", "change_address", "auto"),
 ("آدرس رو عوض کنید برای سفارش ۱۰۸۹۱", "change_address", "auto"),
 ("please cancel 10567", "cancel_order", "auto"),
 ("I don't want the order anymore, cancel it", "cancel_order", "auto"),
 ("سفارش ۱۰۲۳۴ رو کنسل کنید لطفا", "cancel_order", "auto"),
 ("hello!", "greeting", "auto"),
 ("ممنون از راهنمایی", "greeting", "auto"),
 ("get me a real human now", "human_agent", "handoff"),
 ("I'm furious, this is the third time, I want a manager", "human_agent", "handoff"),
 ("با اپراتور وصلم کنید", "human_agent", "handoff"),
 ("میخوام شکایت کنم", "human_agent", "handoff"),
]

out = pathlib.Path(__file__).parent
with open(out / "train.jsonl", "w", encoding="utf-8") as f:
    for intent, texts in TRAIN.items():
        for t in texts:
            f.write(json.dumps({"text": t, "intent": intent}, ensure_ascii=False) + "\n")
with open(out / "test.jsonl", "w", encoding="utf-8") as f:
    for t, i, r in TEST:
        f.write(json.dumps({"text": t, "intent": i, "route": r}, ensure_ascii=False) + "\n")
print(sum(map(len, TRAIN.values())), "train /", len(TEST), "test")
