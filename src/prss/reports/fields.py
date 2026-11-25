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


_TV = r"(is|X|[0-4O][a-d]?)"                  # T value: Tis, TX, T0-T4 with a-d, O = OCR zero
_NV = r"(X|[0-3O][a-c]?(?:mi)?)"              # N value: NX, N0-N3 with a-c, N1mi
_N_AFTER_T = re.compile(r"[\s,;]*(?:[cpyP]\s?)?(?:p\s?)?N\s?" + _NV)


_STAGING_WORDS = r"(?i:tnm|stage|staging|classification|primary tumor)"


def tnm_match(s):
    """Like tnm(), but also returns where the T was found: (T, N, position or None).

    Script 04 uses the position to show the reviewer the exact words the T came from.
    """
    s = s or ""
    t = (
        # 1. a T right after a staging word, with or without "p". Tried FIRST since v4: a report
        #    can mention "(pT2)" in passing (e.g. "Visceral Pleural Involvement: Present (pT2)")
        #    before its real staging line, "Primary Tumor: pT3".
        re.search(_STAGING_WORDS + r"[^.]{0,25}?(?<![A-Za-z])(?:y|r)?(?:[pP]\s?)?T\s?" + _TV, s)
        # 2. with a "p" prefix anywhere, optionally spaced: pT2, ypT1c, PT2, "p T 3"
        or re.search(r"(?<![A-Za-z])(?:y|r)?[pP]\s?T\s?" + _TV, s)
        # 3. bare, but immediately followed by an N: T3N0MX, "T2 N2a", "T2, NO, MX"
        or re.search(r"(?<![A-Za-z0-9])T" + _TV + r"(?=" + _N_AFTER_T.pattern + r")", s)
    )
    n = _N_AFTER_T.match(s, t.end()) if t else None
    if n and n.start() + 25 < n.end():
        n = None
    if not n:
        n = re.search(r"(?<![A-Za-z])(?:y)?[pP]\s?N" + _NV, s)
    zero = lambda v: v.replace("O", "0")
    return ("T" + zero(t.group(1)) if t else None,
            "N" + zero(n.group(1)) if n else None,
            t.start() if t else None)


def tnm(s):
    """Pull pathologic T and N categories out of text -> ('T3', 'N0'), or None for each part not found.

    T is accepted in three ways, tried in order from most to least certain:
      1. shortly after a staging word:            "Primary Tumor: pT3", "pTNM Stage T1b", "TNM stage: pT2"
      2. with a "p" prefix anywhere, spaced or not: pT2, ypT1c, PT2, "p T 3"
      3. bare, but immediately followed by an N:  T3N0MX, "T2 N2a", "T2, NO, MX"
    A bare "T2" anywhere else is still ignored ("T2-weighted MRI").
    N is read right after the T; failing that, a "pN" anywhere. OCR's letter O becomes 0.
    History: v3 added rules 2-3 for bare T (15 real misses). v4 moved the staging-word rule
    first, after a real report gave "(pT2)" in a pleural-invasion line before its staging line.
    Known risk: "T2, NO EVIDENCE..." would read as N0. Rare after a T; watch the disagreements.
    """
    t, n, _ = tnm_match(s)
    return t, n


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
