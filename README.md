# Pathology report and slide search tool

Find trial-eligible patients in free-text pathology reports, and find similar tissue slides, using only public data.

This project continues my module from the DNA Forge AI x Bio Hackathon (November 2026), see [Hackathon_events](https://github.com/Efrazar/Hackathon_events). There, a rule-based criteria engine scored a mean F1 of 1.000 on 30 trial criteria, but the reports were computer-generated and very regular. The question here is what it takes to make the same approach hold up on real, messy pathology reports, and then to rebuild the slide side with open pathology foundation models.

> Research project on public data. No medical claims; not for clinical decisions.

## Status

| Track | What | Status |
|---|---|---|
| 1. Reports | Criteria engine on TCGA pathology reports, scored against GDC clinical labels | Scaffold and ported parser; first baseline run next |
| 2. Slides | Organ classification and similar-slide search with an open foundation model; site-bias study | Not started |
| 3. Demo | Public Streamlit app on the public results | Not started |

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
python scripts/03_baseline_coverage.py      # how much does the hackathon parser find on real reports?
```

Data and model licences: [DATA_SOURCES.md](DATA_SOURCES.md).
