# Pathology report and slide search tool

Find trial-eligible patients in free-text pathology reports, and find similar tissue slides, using only public data.

This project continues my module from the DNA Forge AI x Bio Hackathon (November 2026), see [Hackathon_events](https://github.com/Efrazar/Hackathon_events). There, a rule-based criteria engine scored a mean F1 of 1.000 on 30 trial criteria, but the reports were computer-generated and very regular. The question here is what it takes to make the same approach hold up on real, messy pathology reports, and then to rebuild the slide side with open pathology foundation models.

> Research project on public data. No medical claims; not for clinical decisions.

## Status

| Track | What | Status |
|---|---|---|
| 1. Reports | Read pathologic T and N stage from TCGA pathology reports, scored against GDC clinical labels | First results below. Trial criteria on top of these fields: next. |
| 2. Slides | Organ classification and similar-slide search with an open foundation model; site-bias study | Not started |
| 3. Demo | Public Streamlit app on the public results | Not started |

## Results so far (Track 1)

2,408 TCGA pathology reports from four cancer types, OCR text from the [TCGA-Reports](https://data.mendeley.com/datasets/hyg5xkznpx/1) corpus. "Labels" are the pathologic T and N categories in the NCI Genomic Data Commons (GDC). Scores compare main categories (T2a and T2b both count as T2).

| Cancer type | Reports with a GDC label | T end-to-end | N end-to-end | Accuracy when a value is found (T / N) |
|---|---|---|---|---|
| Colon (COAD) | 351 | 91% | 77% | 98% / 98% |
| Lung adenocarcinoma (LUAD) | 380 | 74% | 67% | 95% / 97% |
| Lung squamous (LUSC) | 391 | 64% | 61% | 97% / 98% |
| Breast (BRCA) | 945 | 59% | 54% | 95% / 96% |

End-to-end = correct values / all labelled reports, so a value we fail to find counts as a miss. Full numbers: [`results/tnm_scores.json`](results/tnm_scores.json).

**How we got here.** The hackathon parser, unchanged, found labelled fields in **2%** of these reports. The synthetic hackathon reports had one field per line, and this corpus has no line breaks at all. Each fix afterwards came from reading the reports the code got wrong:

1. Headings are now found anywhere in the text, with a synonym table built from the most common unmatched headings. Reports with no labelled fields fell from 98% to 19%.
2. T and N are now read when written without the "p" ("T3N0MX", "T2, NO, MX"), with spaces ("p T 3"), or with OCR's letter O for zero ("pNO").
3. A T in the staging line is now preferred over a passing mention ("Visceral pleural invasion: present (pT2)").

Accuracy when a value was found stayed at 95–98% throughout. The gains came from finding more, not from loosening the rules.

**Limitations.**
- GDC labels are not ground truth. In a manual review of 15 disagreements, about half were reports that clearly state a different stage from GDC, for example an amended report that corrects the staging. GDC may draw on sources beyond the report. 15 cases is a small sample, so this is a lead to follow up, not a measured error rate.
- Some reports contain no staging at all, so no text method can reach 100% end-to-end.
- The scores cover only four cancer types, and only T and N. The trial criteria built on these fields are not scored yet.

## Layout

```
src/prss/reports/   parser, field helpers, criteria (ported from the hackathon)
src/prss/slides/    embeddings, linear probe, similarity (Track 2)
scripts/            numbered steps, run in order
tests/              unit tests for the helpers (pytest)
results/            small committed result files
data/               downloads and intermediate files (git-ignored; scripts recreate it)
app/                Streamlit demo (Track 3)
```

## Run Track 1

```bash
conda env create -f environment.yml && conda activate prss
pytest -q                                   # helpers behave as expected
# download TCGA-Reports from Mendeley into data/raw/ (see DATA_SOURCES.md)
python scripts/01_load_reports.py
python scripts/02_get_gdc_clinical.py
python scripts/03_baseline_coverage.py      # how much does the parser find on real reports?
python scripts/04_score_tnm.py              # T and N accuracy against GDC labels; warns if worse than committed results
python scripts/05_score_criteria.py         # trial criteria: report text vs GDC labels, precision/recall/F1
```

Data and model licences: [DATA_SOURCES.md](DATA_SOURCES.md).
