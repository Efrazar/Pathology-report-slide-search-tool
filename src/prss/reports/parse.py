"""Split a pathology report into labelled fields ("Tumor Size: 2.4 cm" -> {"Tumor size": "2.4 cm"}).

Version 2, after the first look at real TCGA text (see results/baseline_coverage.json):
the TCGA-Reports corpus has NO line breaks (0 of 9,523 reports), so version 1, which looked
for "Field:" at the start of a line, found labelled fields in only 2% of reports.

How version 2 works, in two steps:
1. Find every heading-like token anywhere in the text: one to six capitalised-ish words
   followed by a colon, sitting at the start of the text or after ". ", "; " or ": ".
   The text between one heading and the next is that heading's value.
2. Map each heading to a canonical field name through SYNONYMS. Real reports say
   "Surgical Margins", "Margins" or "Margin status" for the same thing.

Unknown headings are kept too (under their own name), because script 03 counts them: the most
common unknown headings are the to-do list for SYNONYMS.

Design choice kept from v1: the parser never fails. A report with no headings gives {}.
"""
import re

# canonical field name -> lower-case headings seen in reports that mean the same thing.
# Extend this from the "most common headings" list that script 03 prints.
SYNONYMS = {
    "Specimen": ["specimen", "specimens", "specimens submitted", "specimen type", "material", "specimen(s)"],
    "Procedure": ["procedure"],
    "Histologic type": ["histologic type", "histological type", "tumor type", "tumour type", "histology"],
    "Histologic grade": ["histologic grade", "histological grade", "grade", "fuhrman nuclear grade",
                         "nuclear grade", "nottingham grade", "overall grade"],
    "Tumor size": ["tumor size", "tumour size", "size of tumor", "size"],
    "Margins": ["margins", "surgical margins", "margin", "margin status", "resection margins"],
    "Lymphovascular invasion": ["lymphovascular invasion", "lymph-vascular invasion", "angiolymphatic invasion",
                                "vascular invasion", "venous (large vessel) invasion"],
    "Perineural invasion": ["perineural invasion"],
    "Lymph nodes": ["lymph nodes", "regional lymph nodes", "number of lymph nodes examined", "lymph node status",
                    "regional lymph nodes (pn)"],
    "Pathologic stage": ["pathologic stage", "pathologic staging", "pathological staging", "staging",
                         "ajcc stage", "tnm", "pathologic staging (ptnm)", "primary tumor (pt)"],
    "Estrogen receptor": ["estrogen receptor", "er"],
    "Progesterone receptor": ["progesterone receptor", "pr"],
    "HER2": ["her2", "her2/neu", "her-2/neu"],
    "Tumor site": ["tumor site", "tumor location", "laterality", "specimen laterality"],
    "Diagnosis": ["diagnosis", "final diagnosis"]
}
_CANON = {syn.strip().lower(): canon for canon, syns in SYNONYMS.items() for syn in syns}

# A heading: starts at the beginning or after ". ", "; " or ": ", then 1-6 words (the first
# capitalised), an optional bracket, then a colon. Capping at 6 words is what stops ordinary
# sentences that happen to end in a colon from becoming headings.
_WORD = r"(?:[A-Za-z][\w/,-]*|&)"                    # any later word: "Tumor size" and "Tumor Size" both occur
_HEADING = re.compile(
    r"(?:^|(?<=[.;:]\s))"                          # where a heading may start
    r"([A-Z][\w/,-]*(?:\s" + _WORD + r"){0,5}"   # 1-6 words, the first one capitalised
    r"(?:\s?\([^)]{0,40}\))?)"                     # optional "(for renal cortical types)"
    r"\s?:\s"                                      # the colon
)


def headings(text):
    """All (heading, value) pairs in reading order. Exposed separately so script 03 can count them."""
    hits = list(_HEADING.finditer(text or ""))
    out = []
    for h, nxt in zip(hits, hits[1:] + [None]):
        value = text[h.end(): nxt.start() if nxt else None].strip(" .;")
        out.append((h.group(1).strip(), value[:300]))   # cap: a run-away value is usually a parse miss
    return out


def parse(text):
    """Return {"fields": {name: value}, "unknown": {heading: value}, "text": text}."""
    fields, unknown = {}, {}
    for head, value in headings(text):
        if not value:
            # A section title right before a sub-heading ("FINAL DIAGNOSIS: Tumor size: ...") has
            # an empty value. Storing it would block a later, real value for the same field.
            continue
        canon = _CANON.get(head.lower())
        if canon:
            fields.setdefault(canon, value)    # first occurrence wins; addenda come later
        else:
            unknown.setdefault(head, value)
    return {"fields": fields, "unknown": unknown, "text": text or ""}
