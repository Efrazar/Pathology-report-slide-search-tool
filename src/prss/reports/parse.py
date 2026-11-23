"""Split a pathology report into labelled fields ("Tumor size: 2.3 cm" -> {"Tumor size": "2.3 cm"}).

The hackathon parser learned its list of field names from the challenge's training labels.
Those are gone, so here the list is written out by hand from common synoptic-report headings
(the "Field: value" checklists pathologists fill in). You can extend FIELD_NAMES as you find
new headings in the TCGA reports; that is exactly the kind of fix Track 1 is about.

Design choice: this parser never fails. A report with no recognisable fields gives {}, and
the criteria simply do not match. Counting how many reports give {} is the first measure of
how far real reports are from the synthetic ones.
"""
import re

FIELD_NAMES = [
    "Procedure", "Specimen", "Specimen laterality", "Tumor site", "Histologic type",
    "Histologic grade", "Nottingham grade", "Gleason score", "Tumor size", "Tumor focality",
    "Depth of invasion", "Margins", "Lymphovascular invasion", "Perineural invasion",
    "Lymph nodes", "Regional lymph nodes", "Tumor deposits", "Visceral pleural invasion",
    "Extraprostatic extension", "Seminal vesicle invasion", "Pathologic stage",
    "Pathologic staging", "Estrogen receptor", "Progesterone receptor", "HER2", "Biomarkers",
]

# Longest names first, so "Pathologic staging" is tried before "Pathologic stage".
_KEYS = sorted(FIELD_NAMES, key=len, reverse=True)
_FIELD = re.compile(
    r"^\s*(?:[-*•]\s*)?(%s)\s*(?:\([^)]*\))?\s*:\s*(.+)$" % "|".join(map(re.escape, _KEYS)),
    re.I | re.M,
)


def parse(text):
    """Return {"fields": {name: value}, "text": text}. Field names are normalised to FIELD_NAMES casing."""
    canon = {k.lower(): k for k in FIELD_NAMES}
    fields = {}
    for m in _FIELD.finditer(text or ""):
        key = canon[m.group(1).lower()]
        fields.setdefault(key, m.group(2).strip())   # keep the first occurrence; addenda come later
    return {"fields": fields, "text": text or ""}
