"""Trial-style criteria that can be checked against public labels.

The hackathon had 30 criteria and an answer key. TCGA has no answer key, so a criterion is
only useful here if its answer can be looked up in the structured GDC clinical data
(project, pathologic T, N and stage). These five are rewritten from hackathon criteria with
that constraint. Receptor-based criteria (ER/PR/HER2) wait until we confirm which public
labels exist for them.

Each criterion returns None (no match) or a dict of the evidence that triggered it, so a
person can check the match by reading two lines instead of the whole report.
"""
from .fields import nodes_pos, size_mm, tnm_match


def extract_tnm(parsed):
    """T and N for one report: (T, N, position of the T in the full text, or None).

    Look in the "Pathologic stage" field first, because it is the most reliable place. Then fill
    any part still missing from the whole text. Why both: a stage field often holds only the T
    ("Primary Tumor (pT): pT2"), while the N sits in its own field ("Regional Lymph Nodes (pN): pN1").
    Searching the stage field alone would lose that N.
    """
    text = parsed["text"]
    field = parsed["fields"].get("Pathologic stage")
    t = n = pos = None
    if field:
        t, n, p = tnm_match(field)
        if p is not None:
            start = text.find(field)
            pos = start + p if start >= 0 else None
    if t is None or n is None:
        t2, n2, p2 = tnm_match(text)
        if t is None:
            t, pos = t2, p2
        n = n or n2
    return t, n, pos


def row(project, parsed):
    """Flatten one parsed report into the values the criteria use."""
    f, text = parsed["fields"], parsed["text"]
    pT, pN, _ = extract_tnm(parsed)
    pos, _ = nodes_pos(f.get("Lymph nodes") or f.get("Regional lymph nodes") or text)
    return dict(project=project, pT=pT, pN=pN, nodes_pos=pos, size_mm=size_mm(f.get("Tumor size")),
                histology=(f.get("Histologic type") or "").lower(), fields=f)


def gdc_row(g):
    """The same row shape, built from a GDC clinical record instead of a report.

    Only project, T and N are filled: GDC has no node counts or tumor sizes. Running the SAME
    criterion functions on both rows is what makes the comparison fair: any difference comes
    from the inputs (our extraction vs the curated labels), not from two versions of the rules.
    """
    import re
    def main(v, letter):
        m = re.match(letter + r"(is|X|\d)", v or "")
        return letter + m.group(1) if m else None
    return dict(project=g["project"], pT=main(g.get("pT"), "T"), pN=main(g.get("pN"), "N"),
                nodes_pos=None, size_mm=None, histology=(g.get("diagnosis") or "").lower(), fields={})


def _ev(r, *keys):
    return {k: r["fields"][k] for k in keys if k in r["fields"]} or {"TNM": f"{r['pT']} {r['pN']}"}


def _t_number(pT):
    """'T2b' -> 2, 'Tis' -> 0, 'TX' or None -> None. Comparing the strings directly would be a bug:
    in Python "Tis" >= "T2" is True, because 'i' sorts after '2'."""
    if pT and len(pT) > 1 and pT[1].isdigit():
        return int(pT[1])
    return 0 if pT == "Tis" else None


def _node_positive(r):
    return (r["nodes_pos"] or 0) >= 1 or (r["pN"] or "N0").startswith(("N1", "N2", "N3"))


def breast_node_positive_over_20mm(r):
    if r["project"] == "TCGA-BRCA" and _node_positive(r) and ((r["size_mm"] or 0) > 20 or (_t_number(r["pT"]) or 0) >= 2):
        return _ev(r, "Lymph nodes", "Tumor size")


def colon_stage_iii(r):
    """Node-positive colon cancer. pN1c (tumor deposits, no positive nodes) counts, as it did in the hackathon key."""
    if r["project"] == "TCGA-COAD" and _node_positive(r):
        return _ev(r, "Lymph nodes", "Pathologic stage")


def colon_pt4(r):
    if r["project"] == "TCGA-COAD" and (r["pT"] or "").startswith("T4"):
        return _ev(r, "Depth of invasion", "Pathologic stage")


def lung_adeno_node_positive(r):
    if r["project"] == "TCGA-LUAD" and _node_positive(r):
        return _ev(r, "Histologic type", "Lymph nodes")


def lung_squamous_pt3_or_higher(r):
    if r["project"] == "TCGA-LUSC" and (_t_number(r["pT"]) or 0) >= 3:
        return _ev(r, "Pathologic stage")


CRITERIA = {f.__name__: f for f in [
    breast_node_positive_over_20mm, colon_stage_iii, colon_pt4,
    lung_adeno_node_positive, lung_squamous_pt3_or_higher,
]}
