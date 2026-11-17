# Silicon Town Hall

A simulated town hall for government teams to test a new policy before real outreach. The I-81 viaduct in Syracuse is the pilot.

IST 688, Building Human Centered AI Applications, Syracuse University, Fall 2026.

## In plain words

Syracuse is replacing the I-81 highway bridge through downtown. Before deciding, the state collected public comments and published them, sorted by topic: noise, pollution, traffic, people losing their homes.

A planning team uploads a draft policy. The app builds a town of a few hundred AI residents from real Census data; each reads the parts of the policy that affect its own life, talks with neighbors in simulated meetings, and writes a comment. The team gets a report of which concerns come up, from which neighborhoods and groups, and who to reach in person. For I-81 we can check that report against what real residents told the state, and we show where the AI town got people wrong.

The app is a rehearsal tool, not a replacement for real public input. Every output is labeled simulated.

## Status, Nov 11, 2026

Done:

- Proposal accepted Nov 11: [docs/proposal.md](docs/proposal.md), submitted as [docs/proposal.pdf](docs/proposal.pdf)
- Found why the Census microdata query returned nothing: the API now requires a free key (see [docs/data-sources.md](docs/data-sources.md))
- Roadmap with workstreams, phase checklists, stack and risks: [docs/roadmap.md](docs/roadmap.md)
- All 47 public source PDFs located; the downloader is tested; page and word counts are in [runs/pdf_stats.csv](runs/pdf_stats.csv), summarized in [docs/data-sources.md](docs/data-sources.md)

- Answer key: 3,009 comments split from 3,632 pages (319 OCRed), names, emails, phones and addresses removed ([runs/ingest_stats.json](runs/ingest_stats.json)). The 344 comments on the final impact statement are tagged with NYSDOT's 13 Appendix A topics by gpt-4.1, each near duplicate text once: 247 distinct texts, 92 of the comments are one form letter ([runs/ingest_tags_feis.json](runs/ingest_tags_feis.json)). Not yet checked by hand.
- Retrieval over the impact statement: 31 documents, 1,363 pages, text and page images, hybrid search with cross-encoder and vision reranking ([runs/retrieval_index.json](runs/retrieval_index.json), [runs/retrieval_eval.json](runs/retrieval_eval.json)).
- Residents, meetings with memory, the analyst and scorer, both baselines, and the Streamlit app behind a password.

20 resident pilot, Nov 11 (one Census draw, seed 0, one citywide meeting, gpt-4.1-mini residents, gpt-4.1 tagger; [runs/report_pilot_score.json](runs/report_pilot_score.json), [runs/town_pilot_real.json](runs/town_pilot_real.json)). All comments are simulated.

| Condition | Real topics raised (of 13) | Topic mix gap (0 same, 1 none shared) |
| --- | --- | --- |
| Meeting, 20 residents | 61.5% | 0.550 |
| No personas, one model, 19 comments | 76.9% | 0.466 |
| Skip the meeting, same 20 residents | 76.9% | 0.419 |

In this pilot the meeting made the comments narrower, not closer to the real ones: all 20 meeting comments raised construction, none raised air quality (165 of 344 real comments) or the viaduct alternative, and no resident opposed the project. One draw of 20 residents is a pilot, not a finding about the method.

Known problems:
- The tagger is unstable across batches. Two gpt-4.1 runs at temperature 0 gave the same codes to 184 of 344 comments ([runs/ingest_tag_stability_feis.json](runs/ingest_tag_stability_feis.json)).
- The form letter's Historic and Cultural Resources tag, which covers 92 comments, drops when the letter is sent alone ([runs/ingest_tag_form_letter_alone.json](runs/ingest_tag_form_letter_alone.json)).
- The seed 0 draw has no Hispanic resident, against 8.36% of weighted adults.

## Deadlines

| Date | What |
| --- | --- |
| Nov 8 | Proposal |
| Nov 10 | Our RAG and reranking class presentation |
| Nov 15 | Approach and milestones |
| Nov 22 | Milestones update, with a working 20 resident pilot |
| Dec 6 | App URL, project document, slides, code zip |

## Run it

```bash
pip install -r requirements.txt
python scripts/download.py    # about 610 MB of public NYSDOT PDFs into data/raw/
python scripts/pdf_stats.py   # rewrites runs/pdf_stats.csv
```

To open the app, copy `.env.example` to `.env` and set `APP_PASSWORD` (the app
stays locked without it). The OpenAI and Census keys are optional. Without the
OpenAI key every reply is FakeLLM template text, not a model, and costs nothing.
Without the Census key the app reads the public ACS PUMS CSV files instead,
downloading about 52 MB for New York into `data/raw/census/` the first time.

```bash
cp .env.example .env          # then set APP_PASSWORD
streamlit run app/streamlit_app.py
```

## Layout

| Path | What |
| --- | --- |
| `docs/` | Proposal, roadmap, data sources |
| `scripts/download.py` | Fetches every source PDF and checks each one is really a PDF |
| `scripts/pdf_stats.py` | Pages, words and OCR pages per PDF |
| `runs/` | Outputs of commands; every number in `docs/` traces to a file here |
| `data/` | Downloaded and processed data. Gitignored, because public comments carry names and email addresses |
