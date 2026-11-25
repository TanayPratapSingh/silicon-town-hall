# Silicon Town Hall: Project Document

## Overview

Silicon Town Hall lets a planning team rehearse public outreach: it tries a draft policy on simulated residents built from Census records, then reports which concerns came up, from whom, and who to reach in person. It is a rehearsal tool, not public comment, and every resident, turn and comment it shows is labeled simulated.

The I-81 viaduct in Syracuse is the pilot, because NYSDOT published 344 real comments on its final impact statement. That lets us score the simulation against what real people said.

**How it works**

1. **Community.** The app draws adults from the Census ACS 2024 5 year microdata for one Census area (Syracuse is PUMA 00701), places each in a tract and neighborhood, and has gpt-4.1-mini write a first person profile from those facts only. A keyword check and a model review flag stereotyped profiles.
2. **Policy.** The plan is indexed page by page as text and as page images. For I-81 that is 31 chapters and 1,363 pages.
3. **Meetings.** A facilitator sets an agenda, then each resident speaks in turn, citing the pages retrieved for its own situation. A running summary keeps the thread, each resident writes one comment, and long term memory carries stances into the next round.
4. **Report.** gpt-4.1 tags each comment with NYSDOT's 13 Appendix A topic codes. The report shows topics by neighborhood and group, sample comments with their cited pages, and the groups the simulation under-represents against Census shares. The team downloads it as an outreach brief, labeled simulated.
5. **What if.** The team describes a change to the plan, reruns the same residents and compares stances and topics side by side.

**Strengths**

- The who to reach list is the most dependable output, because it rests on Census counts, not on model text. It caught that the first 20 resident draw had no Hispanic resident, though Hispanic adults are 8.4% of the area.
- Residents ground what they say in cited pages, and the real comments never enter what residents read.
- It is cheap: a 20 resident meeting cost $0.05 at list price.

**Weaknesses**

- The meeting narrows what residents raise. In both pilot draws its comments covered fewer of the real topics than either baseline, and both times it missed air quality and historic resources.
- Residents echo a proposed change. In the what if run all 20 supported the plan and all 20 raised the change's own topics.
- The topic tags are not yet checked by hand. Until the 100 comment sample is labeled, every topic number is provisional.
- Each pilot draw has only 20 residents, so differences between conditions move a lot from one draw to the next.

## Technical

Two OpenAI models do the language work, ChromaDB holds the impact statement, and a LangGraph loop runs each meeting. Every paid call goes through a hard call cap and a dollar cap.

```
Census microdata --> residents and profiles --+
                                              +--> meetings (LangGraph) --> comments --> tagger --> report and outreach brief
Policy PDF --> text and page image index -----+          ^                                  |
                                                         |                                  v
                                       SQLite memory across rounds       344 real comments --> score (I-81 only)
```

The real comments enter only at the end, as the answer key the simulated comments are scored against.

**Language models**

| Model | What it does | Settings |
| --- | --- | --- |
| gpt-4.1-mini | Resident profiles and their stereotype review, the facilitator's agenda, every meeting turn, the running summary, each comment, both baselines, and the optional page image reranker | temperature 0.7 for profiles, turns and comments; 0.3 for the agenda; 0.2 for the summary |
| gpt-4.1 | Tags every real and every simulated comment with NYSDOT topic codes, with one prompt for both sides | temperature 0, JSON replies |
| FakeLLM | The app's demo mode: template text, not a model, so a demo costs nothing | no calls |

**Data and how it is handled**

- **Sources.** The final impact statement (31 chapters, 1,363 pages); 7 files of public comments on it; Appendix A of the Record of Decision, which gives 13 topic codes; the Census ACS 2024 5 year microdata (6,358 adult records in PUMA 00701); Census 2020 tract files; and the City of Syracuse neighborhood map. Details are in [data-sources.md](data-sources.md).
- **Extraction.** PyMuPDF reads the text layer, and Tesseract OCR reads the 319 pages that have none.
- **De-identification.** Names (spaCy en_core_web_sm plus name lists), emails, phone numbers and street addresses become [NAME], [EMAIL], [PHONE] and [ADDRESS] before any text reaches a model. The tagger refuses a batch that still holds an email or phone number.
- **Storage.** Raw files and comment text stay in `data/`, which is never committed. The `runs/` folder holds counts and scores only.
- **Separation.** The real comments never enter what residents read. They are used once, as the answer key.
- **Duplicates.** A form letter sent 92 times is tagged once and its tags copied, so one batch cannot tag copies of the same letter differently.

**Retrieval (RAG)**

- Each page is split into 200 word chunks with a 40 word overlap and embedded with BAAI/bge-small-en-v1.5. Each page is also embedded as a 100 dpi image with CLIP ViT-B-32.
- A query runs both searches, fuses them by reciprocal rank fusion into 50 candidate pages, and reranks them with the cross-encoder/ms-marco-MiniLM-L-6-v2 model to 20. An optional gpt-4.1-mini vision pass scores the top 10 page images. Each resident turn gets the top 5 pages.
- On a 30 question test set with one correct page each, the correct page reached the top 5 for 33% of questions with text search alone, 40% with fusion plus the cross encoder, and 47% with the vision pass added ([runs/retrieval_eval.json](../runs/retrieval_eval.json)).

**Vector database and memory.** ChromaDB, stored on disk, keeps two collections: 4,004 text chunks and 1,363 page images ([runs/retrieval_index.json](../runs/retrieval_index.json)). An uploaded policy gets its own collection, and the app can delete it. Long term memory is one SQLite file per scenario, with each resident's topic, stance, reasons and comment by round, recalled by similarity with all-MiniLM-L6-v2 embeddings.

**Tools the agents call.** The Census sampler, the tract and neighborhood placer, the retriever, the topic tagger, the scorer, the fact checker, the report builder and the plan editor (what if).

**Third party libraries.** openai, langgraph, chromadb, sentence-transformers, pymupdf, pytesseract, spacy, pandas, numpy, streamlit, altair, pydeck, pillow, python-dotenv and pytest.

## Checking it on the I-81 pilot

The tool is checked where real comments exist: on I-81, its simulated comments are scored against the 344 comments people sent NYSDOT. These are pilot numbers from 20 residents a run, all simulated, each from a file in `runs/`.

| Share of the 13 real topics raised, 20 simulated residents | Meeting | One model, no personas | Same residents, no meeting |
| --- | --- | --- | --- |
| Draw 1, seed 0 ([report_pilot_score.json](../runs/report_pilot_score.json)) | 61.5% | 76.9% | 76.9% |
| Draw 2, seed 1 ([report_pilot_score_seed1.json](../runs/report_pilot_score_seed1.json)) | 69.2% | 100% | 84.6% |
| Mean of the two draws | 65.4% | 88.5% | 80.8% |

In both draws the meeting raised fewer real topics than either baseline. A plain model knows most of the I-81 debate; the meeting pulls residents toward the same few topics.

When one change was added to the plan, all 20 residents backed it, against 9 without it, and their comments covered 38.5% of the real topics instead of 69.2% ([scenario_control_seed0.json](../runs/scenario_control_seed0.json), [scenario_whatif_seed0.json](../runs/scenario_whatif_seed0.json)). In a 50 resident run over three rounds, support rose from 34 to 41 and nobody opposed the plan, while every baseline without a meeting had 1 or 2 opposing comments ([scenario_scale_seed2.json](../runs/scenario_scale_seed2.json)).

**What this means for a team.** Trust the who to reach list, which rests on Census counts. Read the simulated opinions as objections to look for, not a forecast of what residents think.

## Ethics

The largest risk is not a privacy leak but misplaced trust: a planner who reads fluent simulated comments may skip the real outreach. Every mitigation below was measured, and two of the measurements show the tool's own failures.

| Risk | Mitigation | What we measured |
| --- | --- | --- |
| Simulated text is taken for public comment | SIMULATED on every resident, turn, comment and memory; a banner on every app page; the analyst refuses any comment not labeled simulated | Every comment in every real run carried the label |
| Real commenters' names reach a model or the repository | Names, emails, phone numbers and street addresses replaced before any model call; `data/` never committed; `runs/` holds counts only | After scrubbing: 0 emails, 0 phone numbers and 0 known full names left ([runs/ingest_stats.json](../runs/ingest_stats.json)) |
| A private draft policy lingers | Each upload gets its own index; Delete removes the PDF, its index and its page images; memories can be deleted per resident with secure delete | Covered by the offline tests; 81 pass |
| Stereotyped residents | Profiles written from Census facts only; a keyword check and a model review of every profile | 0 of 90 profiles flagged as stereotyped; 2 of 90 flagged for invented facts, both in the seed 1 draw |
| Residents contradict their own record | Each resident's Census facts go into every turn and comment prompt (added after the first pilots) | Phrase matches, a lower bound: 1 contradiction in 80 texts before the fix, 0 in 380 after it ([runs/final_summary.json](../runs/final_summary.json)) |
| Groups left out | The who to reach list compares residents with Census shares | The seed 0 draw had no Hispanic resident, though Hispanic adults are 8.4% of the area; 41 groups were flagged |
| The model already knows the I-81 debate | A baseline with no personas and no meeting | That baseline raised more real topics than the meeting in both draws (76.9% and 100%), so general knowledge, not the personas, carries most of the coverage |
| Planners over-trust a what if | Pilot scores shown beside every report | In the what if run, all 20 residents supported the changed plan and every comment raised the change's own topics |

**Bias.** Within a 20 resident draw, a group has 3 to 15 comments, too few to rank groups. In both draws every group's topic mix sat 0.47 to 0.68 from the real comments' mix, so no group was simulated well. The honest reading is that the tool shows who is missing, not what any group thinks.

**Societal impact.** Used as a rehearsal, it can show a team which groups and concerns it has not yet reached. Used as a replacement for hearings, it would put a model's guesses in the place of residents' own words. The app's labels, its pilot scores and this document all say it is the former.

## Reproduce

- **Code.** This repository, branch `main`. `git archive --format=zip HEAD` makes the code zip, without `data/`.
- **App.** Online demo at https://silicon-town-hall.streamlit.app (password protected). Locally: `streamlit run app/streamlit_app.py`; the README covers the password, the optional keys and the online demo build. The online build shows the recorded real runs, their reports and their outreach briefs; search, uploads and new meetings run locally, because the index is 724 MB.
- **Tests.** `python -m pytest` runs the offline suite: 81 tests pass.
- **Cost.** All recorded real runs together came to $1.92 at list price, under hard call and dollar caps.

| Number | Command | File in `runs/` |
| --- | --- | --- |
| Topic tags on the 344 real comments | `python -m townhall.ingest.tagger --groups feis_comments --cap 1.00` | ingest_tags_feis.json |
| Retrieval test, 30 questions | `python -m townhall.retrieval.evaluate --vision` | retrieval_eval.json |
| Pilot draw 1, both baselines, scores | `python -m townhall.town.pilot --real-profiles`, `--real-meeting`, then `townhall.report.baselines` and `townhall.report.score` | report_pilot_score.json |
| Pilot draw 2 | the same with `--seed 1` and `--suffix _seed1` | report_pilot_score_seed1.json |
| Control, what if and scale runs | `python scripts/run_scenario.py` with the arguments recorded in each file | scenario_control_seed0.json, scenario_whatif_seed0.json, scenario_scale_seed2.json |
| Everything above in one place | `python scripts/final_summary.py` | final_summary.json |

Every run file records the exact command that wrote it.
