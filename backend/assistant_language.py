"""Language policy for answers, independent from invoice source values."""
import re
import unicodedata


def normalize_text(text):
    text = unicodedata.normalize("NFKD", str(text)).casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("ـ", "").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")


def answer_language(question):
    normalized = normalize_text(question)
    if re.search(r"(?:en|in)\s+(?:arabe|arabic)\b|بالعربية|بالعربي", normalized):
        return "ar"
    if re.search(r"(?:en|in)\s+(?:francais|french)\b|بالفرنسية|بالفرنسي", normalized):
        return "fr"
    return "ar" if re.search(r"[\u0621-\u064a]", normalized) else "fr"


def language_instruction(question):
    language = "arabe" if answer_language(question) == "ar" else "francais"
    return (
        f"La langue de la reponse est le {language}, quelle que soit la langue des sources. "
        "Traduis uniquement l'explication. Conserve exactement les noms propres, references, "
        "dates, montants, devises et citations sources. Distingue HT, TTC, TVA et taux. "
    )


def no_evidence(question):
    return ("لا تسمح المقاطع المسترجعة بالإجابة بدليل موثوق."
            if answer_language(question) == "ar"
            else "Les passages retrouves ne permettent pas de repondre avec une preuve.")


# Expand invoice concepts for French/Arabic lexical retrieval; never alter the question sent to the LLM.
CONCEPTS = (
    (r"\bttc\b|شامل.*(?:ضريب|اداء)|باحتساب.*اداء", "TTC total montant شامل الضريبة باحتساب الاداء"),
    (r"\bht\b|دون.*(?:ضريب|اداء)|خارج.*(?:ضريب|اداء)", "HT subtotal hors taxe دون الضريبة خارج الاداء"),
    (r"\btva\b|القيمة المضافة", "TVA taxe valeur ajoutee القيمة المضافة"),
    (r"fournisseur|supplier|المورد|المزود|البائع", "fournisseur supplier vendor المورد المزود البائع"),
    (r"client|customer|العميل|الحريف|المشتري", "client customer العميل الحريف المشتري"),
    (r"date|تاريخ", "date facture تاريخ الفاتورة"),
    (r"adresse|عنوان", "adresse address العنوان"),
    (r"telephone|هاتف", "telephone phone الهاتف"),
    (r"paiement|payment|دفع|خلاص", "paiement payment دفع خلاص"),
    (r"especes|نقد", "especes cash نقدا"),
    (r"matricule|fiscal|الجبائي|الضريبي", "matricule fiscal MF معرف جبائي ضريبي"),
    (r"timbre|طابع", "timbre fiscal stamp طابع جبائي"),
    (r"produit|article|designation|منتج|سلع", "produit article designation منتج سلع"),
)


def retrieval_question(question):
    normalized = normalize_text(question)
    aliases = [terms for pattern, terms in CONCEPTS if re.search(pattern, normalized)]
    return question + (" " + " ".join(aliases) if aliases else "")


def grounded_numbers(answer, sources):
    """Reject invented numbers even when the LLM supplies valid citation indexes."""
    def numbers(text):
        text = re.sub(r"\[\d+\]", "", str(text))
        text = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in text)
        text = text.replace("\u066b", ".").replace("\u066c", " ")
        return {token.replace(",", ".").replace(" ", "").replace("\u00a0", "")
                for token in re.findall(r"\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?", text)}
    return numbers(answer).issubset(set().union(*(numbers(s.get("text", "")) for s in sources)))
