# Pathology report and slide search tool

Find trial-eligible patients in free-text pathology reports, and find similar tissue slides, using only public data.

This project continues my module from the DNA Forge AI x Bio Hackathon (November 2026), see [Hackathon_events](https://github.com/Efrazar/Hackathon_events). There, a rule-based criteria engine scored a mean F1 of 1.000 on 30 trial criteria, but the reports were computer-generated and very regular. The question here is what it takes to make the same approach hold up on real, messy pathology reports, and then to rebuild the slide side with open pathology foundation models.

> Research project on public data. No medical claims; not for clinical decisions.

## Status

| Track | What | Status |
|---|---|---|
| 1. Reports | Trial criteria and T/N stage from TCGA pathology reports, scored against GDC clinical labels | Results below: mean F1 0.83 on 5 criteria, precision 94–97% |
| 2. Slides | Organ classification and similar-slide search with an open foundation model; site-bias study | Baseline scripts ready (PanNuke tiles, H-optimus-0 embeddings); first run pending |
| 3. Demo | Public Streamlit app on the public results | Not started |

## Results so far (Track 1)

### Trial criteria: does the report text pick the same patients as the curated labels?

The same criterion functions run twice per patient: once on what the parser extracted from the report, once on the GDC labels. Only patients of the criterion's cancer type whose GDC record can decide it are counted.

| Criterion | Patients judged | Qualify per GDC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Colon, pT4 | 351 | 39 | 97% | 92% | 0.95 |
| Colon, node-positive (stage III, unless metastatic) | 351 | 134 | 96% | 82% | 0.88 |
| Lung adenocarcinoma, node-positive | 379 | 121 | 94% | 69% | 0.80 |
| Lung squamous, pT3 or higher | 391 | 66 | 96% | 68% | 0.80 |
| Breast, node-positive and over 20 mm | 944 | 374 | 97% | 60% | 0.74 |
| **Mean** | | | | | **0.83** |

The hackathon scored a mean F1 of 1.00 with the same kind of rules on synthetic reports. On real reports, **precision stays at 94–97%**: when the engine flags a patient, GDC agrees about 19 times in 20. **Recall varies from 60% to 92%** and tracks how often the T or N stage can be found in the text at all (below). Full numbers: [`results/criteria_scores.json`](results/criteria_scores.json).

### T and N extraction

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
- Only four cancer types and five criteria, all built on T and N, because those are the fields GDC labels for every case. Criteria on receptors (ER/PR/HER2), margins or histology need other labels and are not scored yet.
- The 255 criteria disagreements have not yet been reviewed by hand, so the split between our misses and label problems is unknown for the criteria.

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

## Run Track 2 (slides, baseline)

Needs a CUDA GPU for step 11 only (tested target: RTX 2080 Ti, 11 GB). Steps 12 and 13 run on the CPU in seconds.

```bash
pip install -e ".[slides]"                  # torch, timm, huggingface_hub
python -c "import torch; print(torch.cuda.is_available())"   # must print True for step 11
# accept the terms at https://huggingface.co/bioptimus/H-optimus-0, then:
hf auth login                               # paste a read token (older huggingface_hub: huggingface-cli login)
python scripts/10_get_pannuke.py            # 3 folds, about 2 GB download (resumes if cut); tiles per tissue type
python scripts/11_embed_tiles.py --smoke    # 1 minute, random tiny model: checks the pipeline
python scripts/11_embed_tiles.py            # H-optimus-0, fp16; the only GPU step
python scripts/12_organ_probe.py            # linear probe: accuracy, balanced accuracy, macro F1
python scripts/13_similar_tiles.py          # precision at 1, 5, 10 against a chance line
```

Data and model licences: [DATA_SOURCES.md](DATA_SOURCES.md).
