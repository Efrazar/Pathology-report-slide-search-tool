"""Small, testable helpers that turn one piece of report text into one value.

Ported from my hackathon criteria engine. Each function does one job and returns None
when it cannot find an answer, rather than guessing. That matters more on real reports:
a missing value should stay missing, so the evaluation can count it as a miss.

Changes from the hackathon version are marked "PUBLIC:" with the reason.
"""
import re

_NUM = r"(\d+(?:\.\d+)?)"


def size_mm(s):
    """'2.3 cm' -> 23.0, '15 mm' -> 15.0. Takes the first size it finds.

    PUBLIC: real reports often write "2.3 x 1.8 x 1.5 cm". The first number is usually the
    greatest dimension, which is what staging uses, so we keep "first number" but allow the
    unit to come after the last dimension.
    """
    m = re.search(_NUM + r"(?:\s*x\s*\d+(?:\.\d+)?)*\s*(mm|cm)\b", s or "", re.I)
    if not m:
        return None
    v = float(m.group(1))
    return v * 10 if m.group(2).lower() == "cm" else v


def nodes_pos(s):
    """'3 of 12 positive' or '3/12' -> (3, 12). Returns (None, None) if absent.

    PUBLIC: added the 'positive/examined' slash form, which is common in real reports.
    """
    s = s or ""
    m = re.search(r"(\d+)\s+of\s+(\d+)\s+(?:lymph\s+nodes?\s+)?(?:are\s+)?positive", s, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"\((\d+)\s*/\s*(\d+)\)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


_NEG = ("not identified", "absent", "none", "negative", "no ")


def present(s):
    """'present', 'present (PL1)' -> True; 'not identified', 'absent' -> False; '' -> None.

    This is where negation lives. A keyword search for "invasion" matches "no invasion";
    this does not.
    """
    s = (s or "").lower().strip()
    if not s:
        return None
    if s.startswith(_NEG):
        return False
    if s.startswith(("present", "identified", "positive")):
        return True
    return None   # PUBLIC: unknown wording stays unknown instead of becoming False


def tnm(s):
    """Pull pathologic T and N categories out of text: 'pT2 pN1a', 'pT3N0MX', 'pT3, pNO' -> ('T3', 'N0').

    PUBLIC v2, after looking at real TCGA text:
    - T must carry the "p" prefix (pT, ypT, rpT). Without it, words like "T2-weighted" or a
      sample label "T1" would count as stages. We lose a few reports that write "T2 N0", and
      that trade is measured in script 04.
    - N is looked for just after the T (within 25 characters), so the "p" is optional there.
      A lone "pN" anywhere is the fallback.
    - OCR reads zero as the letter O ("pNO", "pTO"), so O is accepted and turned into 0.
    """
    s = s or ""
    t = re.search(r"(?<![A-Za-z])(?:y|r)?p(T(?:is|X|[0-4O][a-d]?))", s)
    n = None
    if t:
        n = re.match(r"[\s,;]*(?:c|p|y)?p?(N(?:X|[0-3O][a-c]?(?:mi)?))", s[t.end():t.end() + 25])
    if not n:
        n = re.search(r"(?<![A-Za-z])(?:y)?p(N(?:X|[0-3O][a-c]?(?:mi)?))", s)
    fix = lambda v: v[0] + v[1:].replace("O", "0") if v else None
    return (fix(t.group(1)) if t else None, fix(n.group(1)) if n else None)


def receptor(text, name):
    """Status word after a receptor name: receptor('ER positive (95%)', 'ER') -> 'positive'.

    PUBLIC: also accepts 'ER: positive' and 'Estrogen receptor ... positive'. Still the weakest
    helper: real reports describe HER2 with IHC scores (0, 1+, 2+, 3+) and FISH results, which
    this does not interpret yet. That is a known gap for the first evaluation.
    """
    aliases = {
        "ER": r"(?:ER|estrogen receptor)",
        "PR": r"(?:PR|PgR|progesterone receptor)",
        "HER2": r"(?:HER-?2(?:/neu)?|ERBB2)",
    }
    pat = aliases.get(name, re.escape(name)) + r"[^.\n]{0,40}?\b(positive|negative|equivocal)\b"
    m = re.search(pat, text or "", re.I)
    return m.group(1).lower() if m else None
