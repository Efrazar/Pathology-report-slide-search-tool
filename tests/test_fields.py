"""Unit tests for the report helpers. Run with:  pytest -q

Why tests here: every time we change a regex to handle a new TCGA wording, these make sure
we did not break a wording that already worked. Add a test each time you fix a real case.
"""
from prss.reports.fields import nodes_pos, present, receptor, size_mm, tnm
from prss.reports.parse import parse


def test_size():
    assert size_mm("2.3 cm") == 23.0
    assert size_mm("15 mm") == 15.0
    assert size_mm("2.3 x 1.8 x 1.5 cm") == 23.0
    assert size_mm("no tumor") is None


def test_nodes():
    assert nodes_pos("3 of 12 positive") == (3, 12)
    assert nodes_pos("Metastatic carcinoma in lymph nodes (2/15)") == (2, 15)
    assert nodes_pos("") == (None, None)


def test_negation():
    assert present("Present (PL1)") is True
    assert present("Not identified") is False
    assert present("") is None
    assert present("cannot be assessed") is None


def test_tnm():
    assert tnm("pT2 pN1a") == ("T2", "N1a")
    assert tnm("pT3N0MX") == ("T3", "N0")
    assert tnm("ypT1c pN0") == ("T1c", "N0")


def test_receptor():
    assert receptor("ER positive (95%)", "ER") == "positive"
    assert receptor("Estrogen receptor: negative", "ER") == "negative"
    assert receptor("HER2/neu: equivocal (2+)", "HER2") == "equivocal"


def test_parse_fields():
    text = "FINAL DIAGNOSIS: Tumor size: 2.3 cm. Lymph nodes: 0 of 14 positive. Margins: negative"
    f = parse(text)["fields"]
    assert f == {"Tumor size": "2.3 cm", "Lymph nodes": "0 of 14 positive", "Margins": "negative"}


def test_criteria_t_number():
    from prss.reports.criteria import _t_number
    assert _t_number("T2b") == 2
    assert _t_number("Tis") == 0
    assert _t_number("TX") is None


# ---- v2: cases taken from real TCGA-Reports text ----
def test_tnm_ocr_and_false_hits():
    assert tnm("G3, pT3, pNO. Tubular adenocarcinoma") == ("T3", "N0")   # OCR letter O for zero
    assert tnm("T2-weighted MRI showed a mass") == (None, None)           # no "p": not a stage
    assert tnm("pT2 N1 MX") == ("T2", "N1")                                # N right after T, no "p"


def test_parse_one_line_report():
    text = ("Clinical Diagnosis & History: Incidental 3 cm renal mass. Tumor Type: Renal cell carcinoma. "
            "Tumor Size: Greatest diameter is 2.4 cm. Local Invasion (for renal cortical types): Not Identified. "
            "Surgical Margins: Free of tumor. Lymph Nodes: Not identified.")
    p = parse(text)
    assert p["fields"]["Histologic type"] == "Renal cell carcinoma"
    assert p["fields"]["Tumor size"] == "Greatest diameter is 2.4 cm"
    assert p["fields"]["Margins"] == "Free of tumor"
    assert "Local Invasion (for renal cortical types)" in p["unknown"]


# ---- v3: the T misses Efrain pulled from real reports (24 Nov 2026) ----
def test_tnm_bare_t_followed_by_n():
    assert tnm("No lymph node metastases. A. 5: T3N0MX. N. COMMENT.") == ("T3", "N0")
    assert tnm("PTNM CLASSIFICATION: T2, NO, MX, STAGE IIA.") == ("T2", "N0")
    assert tnm("PTNM CLASSIFICATION: T2 N2a, MX, STAGE IIIA.") == ("T2", "N2a")


def test_tnm_spaced_prefix():
    assert tnm("corresponds to tumor stage p T 3, p N 0 (0/36), MX, L1, R0.") == ("T3", "N0")


def test_tnm_after_staging_word():
    assert tnm("any of 3 lymph nodes. pTNM Stage T1b NO. Non-tumorous lung") == ("T1b", "N0")
    assert tnm("Pathologic stage T2N1MX. PART 1: LYMPH NODES") == ("T2", "N1")


def test_tnm_still_ignores_stray_t():
    assert tnm("T2-weighted MRI showed a mass; specimen T1 block") == (None, None)


# ---- v4: from the disagreement review (25 Nov 2026) ----
def test_tnm_prefers_staging_line_over_passing_mention():
    text = ("Visceral Pleural Involvement: Present (pT2). Satellite Tumor(s): Absent. "
            "Pathologic Staging: Primary Tumor: pT3. Regional Lymph Nodes: pN1.")
    assert tnm(text) == ("T3", "N1")


def test_tnm_ignores_pt_inside_words():
    assert tnm("immunohistochemical methods (DAKO HercepTest) with the following") == (None, None)


def test_tnm_match_position():
    from prss.reports.fields import tnm_match
    text = "Nuclear score:3, Pathological TNM stage: pT2. Dr."
    t, n, pos = tnm_match(text)
    assert t == "T2" and "TNM stage: pT2" in text[pos:]
