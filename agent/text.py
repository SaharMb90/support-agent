"""Text utilities: Persian normalisation, language detection, entity extraction."""
import re

_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ك": "ک", "ة": "ه"})
_ORDER_ID = re.compile(r"(?<!\d)(\d{5})(?!\d)")
_PERSIAN_CHARS = re.compile(r"[؀-ۿ]")
_LATIN_CHARS = re.compile(r"[A-Za-z]")


def normalize(text: str) -> str:
    """Unify digits and Arabic/Persian letter variants, collapse whitespace."""
    text = text.translate(_FA_DIGITS).translate(_ARABIC_TO_PERSIAN)
    text = text.replace("‌", " ")  # zero-width non-joiner -> space
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> str:
    """Return 'fa' if the message is mostly Persian script, else 'en'."""
    fa, en = len(_PERSIAN_CHARS.findall(text)), len(_LATIN_CHARS.findall(text))
    return "fa" if fa > en else "en"


def extract_order_id(text: str) -> str | None:
    match = _ORDER_ID.search(normalize(text))
    return match.group(1) if match else None


_ANGER = [
    "furious", "angry", "ridiculous", "worst", "unacceptable", "scam", "sue", "third time", "lawyer",
    "عصبانی", "مسخره", "افتضاح", "کلاهبردار", "شکایت", "بار سوم", "وکیل",
]


def is_angry(text: str) -> bool:
    low = normalize(text).lower()
    return any(re.search(rf"(?<!\w){re.escape(w)}", low) for w in _ANGER)
