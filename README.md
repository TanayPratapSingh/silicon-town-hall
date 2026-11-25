# Silicon Town Hall

A simulated town hall for government teams to test a new policy before real outreach. The I-81 viaduct in Syracuse is the pilot.

IST 688, Building Human Centered AI Applications, Syracuse University, Fall 2026.

## In plain words

Syracuse is replacing the I-81 highway bridge through downtown. Before deciding, the state collected public comments and published them, sorted by topic: noise, pollution, traffic, people losing their homes.

A planning team uploads a draft policy. The app builds a town of a few hundred AI residents from real Census data; each reads the parts of the policy that affect its own life, talks with neighbors in simulated meetings, and writes a comment. The team gets a report of which concerns come up, from which neighborhoods and groups, and who to reach in person. For I-81 we can check that report against what real residents told the state, and we show where the AI town got people wrong.

The app is a rehearsal tool, not a replacement for real public input. Every output is labeled simulated.

## Status, Nov 24, 2026

Done:

- Proposal accepted Nov 11: [docs/proposal.md](docs/proposal.md), submitted as [docs/proposal.pdf](docs/proposal.pdf)
- Found why the Census microdata query returned nothing: the API now requires a free key (see [docs/data-sources.md](docs/data-sources.md))
- Roadmap with workstreams, phase checklists, stack and risks: [docs/roadmap.md](docs/roadmap.md)
- All 47 public source PDFs located; the downloader is tested; page and word counts are in [runs/pdf_stats.csv](runs/pdf_stats.csv), summarized in [docs/data-sources.md](docs/data-sources.md)

- Answer key: 3,009 comments split from 3,632 pages (319 OCRed), names, emails, phones and addresses removed ([runs/ingest_stats.json](runs/ingest_stats.json)). The 344 comments on the final impact statement are tagged with NYSDOT's 13 Appendix A topics by gpt-4.1, each near duplicate text once: 247 distinct texts, 92 of the comments are one form letter ([runs/ingest_tags_feis.json](runs/ingest_tags_feis.json)). Not yet checked by hand.
- Retrieval over the impact statement: 31 documents, 1,363 pages, text and page images, hybrid search with cross-encoder and vision reranking ([runs/retrieval_index.json](runs/retrieval_index.json), [runs/retrieval_eval.json](runs/retrieval_eval.json)).
- Residents, meetings with memory, the analyst and scorer, both baselines, and the Streamlit app behind a password.

Real model runs (gpt-4.1-mini residents, gpt-4.1 tagger), scored against the 344 real comments. All residents and comments are simulated; each pilot is one Census draw of 20 residents in one citywide meeting.

| Run | Real topics raised (of 13) | Topic mix gap (0 same, 1 none shared) | Source |
| --- | --- | --- | --- |
| Pilot draw 1 (seed 0): meeting | 61.5% | 0.550 | [runs/report_pilot_score.json](runs/report_pilot_score.json) |
| Pilot draw 1: one model, no personas | 76.9% | 0.466 | same |
| Pilot draw 1: same residents, no meeting | 76.9% | 0.419 | same |
| Pilot draw 2 (seed 1): meeting | 69.2% | 0.495 | [runs/report_pilot_score_seed1.json](runs/report_pilot_score_seed1.json) |
| Pilot draw 2: one model, no personas | 100% | 0.348 | same |
| Pilot draw 2: same residents, no meeting | 84.6% | 0.455 | same |
| Control: draw 1 residents, Census facts in the prompts | 69.2% | 0.521 | [runs/scenario_control_seed0.json](runs/scenario_control_seed0.json) |
| What if: the same, plus two plan commitments | 38.5% | 0.613 | [runs/scenario_whatif_seed0.json](runs/scenario_whatif_seed0.json) |
| Scale: 50 new residents (seed 2), 5 district meetings, 3 rounds | 92.3% | 0.426 | [runs/scenario_scale_seed2.json](runs/scenario_scale_seed2.json) |

In both pilot draws the meeting made the comments narrower, not closer to the real ones: all 20 meeting comments raised construction and none raised air quality (165 of 344 real comments), in both draws (issue #23). In the what if run all 20 residents supported the changed plan, against 9 in the control, and every comment raised the change's own topics. In the scale run support rose from 34 to 41 of 50 over three rounds. No resident opposed the plan in any meeting run, while every baseline, with no meeting, had 1 or 2 opposing comments out of 20. The scale run's 92.3% is not comparable to the pilots, because 50 comments reach more topics than 20 and it has no baseline. All scored runs are gathered in [runs/final_summary.json](runs/final_summary.json); together the recorded real runs cost $1.92 at list price.

Known problems:
- The tagger is unstable across batches. Two gpt-4.1 runs at temperature 0 gave the same codes to 184 of 344 comments ([runs/ingest_tag_stability_feis.json](runs/ingest_tag_stability_feis.json)).
- The form letter's Historic and Cultural Resources tag, which covers 92 comments, drops when the letter is sent alone ([runs/ingest_tag_form_letter_alone.json](runs/ingest_tag_form_letter_alone.json)).
- The seed 0 draw has no Hispanic resident, against 8.36% of weighted adults.
- The 100 comment hand check of the tags ([issue #3](https://github.com/TanayPratapSingh/silicon-town-hall/issues/3)) is not done, so every topic number is provisional.
- Meeting turns could state facts the resident's Census record contradicts (issue #21). Each resident's facts now go into every prompt; a phrase match check found 1 contradiction in 80 texts before and 0 in 80 after (a lower bound, [runs/final_summary.json](runs/final_summary.json)).

## Deliverables

| What | File |
| --- | --- |
| Proposal, accepted Nov 11 | [docs/proposal.pdf](docs/proposal.pdf) |
| Approach and milestones, Nov 15 | [docs/approach-and-milestones.pdf](docs/approach-and-milestones.pdf) |
| Milestone update, Nov 22 | [docs/milestone-update-nov22.pdf](docs/milestone-update-nov22.pdf) |
| Final project document | [docs/final/final-document.pdf](docs/final/final-document.pdf), Word: [final-document.docx](docs/final/final-document.docx) |
| Slides, 11 with speaker notes | [docs/slides/](docs/slides/) |
| Demo script with timings | [docs/demo-script.md](docs/demo-script.md) |
| Code | this repository; the zip is `git archive --format=zip HEAD` |

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

## Online demo (Streamlit Community Cloud)

The online build shows the committed real runs in `app/demo/` (simulated text and public Census records only): their
reports, meeting replays, comparisons and downloadable outreach briefs, and it samples residents. Search, uploads and
new meetings say they run on the team's computer, because the 724 MB index is not uploaded. It installs from
`app/requirements.txt`, which leaves out torch, sentence-transformers and spacy.

1. Sign in at share.streamlit.io with the GitHub account that owns this repository and allow access to it.
2. Create an app: this repository, branch `build`, main file `app/streamlit_app.py`; under Advanced settings pick
   Python 3.12.
3. Under Secrets, set `APP_PASSWORD` (the team password), `SILICON_DEMO = "1"` and, to sample new residents,
   `CENSUS_API_KEY`. Leave `OPENAI_API_KEY` out, so the online app can never spend money.
4. Deploy, and check that the build log installs from `app/requirements.txt`.

To refresh the demo runs after a new real run: `python -m app.core --export-demo`, then commit `app/demo/`.

## Layout

| Path | What |
| --- | --- |
| `docs/` | Proposal, roadmap, data sources |
| `townhall/ingest/` | Answer key: extraction, OCR, comment splitting, scrubbing, codebook, tagger |
| `townhall/retrieval/` | Index of text chunks and page images, fusion, cross encoder and vision reranking, evaluation |
| `townhall/town/` | Census residents, homes and districts, profiles, the LangGraph meeting loop, memory, pilots |
| `townhall/report/` | Analyst, scorer, baselines, fact check |
| `app/` | Streamlit app, charts, the online demo's requirements and committed demo runs |
| `scripts/download.py` | Fetches every source PDF and checks each one is really a PDF |
| `scripts/pdf_stats.py` | Pages, words and OCR pages per PDF |
| `scripts/run_scenario.py` | A real model run through the app's own code under call and dollar caps, scored |
| `scripts/final_summary.py` | Gathers every scored run into `runs/final_summary.json` |
| `tests/` | Offline tests: `python -m pytest` |
| `runs/` | Outputs of commands; every number in `docs/` traces to a file here |
| `data/` | Downloaded and processed data. Gitignored, because public comments carry names and email addresses |
