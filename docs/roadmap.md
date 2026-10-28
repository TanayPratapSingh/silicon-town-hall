# Silicon Town Hall: Roadmap

Oct 27, 2026

We build in four phases, one per Blackboard deadline, and ship the app, document, slides and code by Dec 6, 2026. The critical path is the answer key: tagging the real I-81 comments with NYSDOT's topic codes. Nothing can be scored until it exists, so it starts first.

## Roadmap

![Roadmap: five workstreams across four Blackboard deadlines, with the Nov 22 milestones update highlighted](roadmap.png)

Every lane's work fits inside one phase, so each deadline gets something from all five people. The answer key finishes at the milestones update, which frees that person to help with evaluation in the final two weeks.

## Workstreams

Five tracks, one owner each, so all five people work in parallel from week one. Fill in the Owner column at the next group meeting.

| Workstream | What it delivers | Owner |
| --- | --- | --- |
| 1. Answer key | Every comment PDF downloaded and turned into text (OCR for scanned pages), email boilerplate and personal details stripped, each comment tagged with NYSDOT topic codes, 100 tags checked by hand |  |
| 2. Project RAG | Impact statement chunked and embedded, vector store, reranker, retrieval tests; doubles as our Nov 10 RAG and reranking presentation demo |  |
| 3. Residents | Census persona sampler, persona prompts, neighborhood meeting loop, agent memory across rounds |  |
| 4. Evaluation and ethics | Coverage and topic mix metrics, two baselines, per group divergence report, ethics rubric write up |  |
| 5. App and delivery | Streamlit app with the what if editor, deployment, slides, demo script, final document assembly |  |

## Phase checklists

The number after each task is its workstream.

### Phase 1: now to Nov 8 (proposal due)

- [ ] Trim the proposal to one page and submit it (all)
- [ ] Email the instructor the concept, subject "IST 688: Final project concept" (5)
- [x] Set up the shared repo (5); Streamlit skeleton and API spending cap still to do
- [x] Download every FEIS comment PDF, Appendix A and the impact statement chapters (1): `python scripts/download.py`
- [ ] Turn Appendix A's table of contents into a topic codebook with a one line definition per topic (1, 4)
- [ ] Get a working Census microdata query for the Syracuse area (3)

### Phase 2: Nov 9 to Nov 15 (approach and milestones due)

- [ ] Extract text, OCR scanned pages, strip the quoted NYSDOT announcement and email signatures (1)
- [ ] Remove names, emails, phone numbers and street addresses (1)
- [ ] Build the impact statement index and reranker, then present it Nov 10 (2)
- [ ] Write 20 pilot personas from Census records and review them for stereotypes (3, 4)
- [ ] Turn this roadmap into the Approach and Milestones submission (5)

### Phase 3: Nov 16 to Nov 22 (milestones update due)

- [ ] Tag all real comments with an LLM; hand check 100 and report how often the tagger agrees with us (1, 4)
- [ ] Pilot: 20 residents, one meeting, one comment each, scored end to end (3, 4)
- [ ] Run both baselines on the same pilot (4)
- [ ] Deploy the app online behind a password (5)
- [ ] Submit the milestones update with the pilot numbers, labeled as a 20 resident pilot (5)

### Phase 4: Nov 23 to Dec 6 (final deliverables due)

Thanksgiving is Nov 26, so plan on losing two or three working days.

- [ ] Scale to a few hundred residents and three meeting rounds with memory (3)
- [ ] Build the what if editor: change the plan, rerun, compare who objects (3, 5)
- [ ] Final coverage and topic mix numbers, per group divergence, missed voices list (4)
- [ ] Ethics rubric evaluation and write up (4)
- [ ] Project document, including each person's individual section (all)
- [ ] Slides, demo script and two full rehearsals under 10 minutes (5, all)
- [ ] Zip the code and submit the URL, document and slides (5)

## Stack

| Layer | Pick | Why |
| --- | --- | --- |
| LLM | A small, cheap model for residents; a stronger one for the tagger and analyst | Residents generate most of the tokens; price the 20 resident pilot before scaling |
| Orchestration | LangChain, with LangGraph for the meeting loop | Handles agent turns and memory, and counts as a second presentation topic |
| Vector store and memory | Postgres with pgvector on a hosted free tier, such as Supabase | One store for RAG and long term memory; Streamlit Cloud's disk resets on restart |
| Reranker | Open source cross encoder from sentence transformers | Free, and fast enough on CPU for a few hundred passages |
| OCR | Tesseract, with a vision model for pages it garbles | 218 of 822 FEIS comment pages have no text layer ([data sources](data-sources.md)) |
| Personas | Census ACS 5 year microdata API | Free; a key raises the rate limit |
| App | Streamlit Community Cloud behind a password | Taught in class; free hosting |

## Risks

| Risk | Why it matters | Fallback |
| --- | --- | --- |
| The model already knows the I-81 debate | Residents may repeat real arguments from training data instead of simulating; that looks like success but is not | The no persona baseline exposes it: if a generic LLM scores as well as the personas, we report that |
| Email boilerplate in the answer key | Replies quote NYSDOT's announcement underneath (seen in the A to C file), so the tagger would tag the state's words as a resident's | Strip quoted text before tagging; spot check 20 comments |
| The tagger is wrong | Every result rests on the tags | Hand check 100 tags and report agreement beside every number |
| The Census record query still returns nothing | Personas depend on it | Fix the area codes in phase 1; fall back to tract level summary tables |
| Neighborhoods are too coarse | Microdata areas cover 100,000 or more people each | Place personas into neighborhoods using tract level tables |
| API cost | Hundreds of residents, three rounds, long retrieved context | Price the pilot, set a spending cap, cache retrieved passages |
| Thanksgiving and five schedules | Phase 4 loses days | The pilot must work by Nov 22, so phase 4 is scaling, not building |
| Presentation date not posted | It may come before Dec 6 | Ask in the concept email |

## Final deliverables

Each person should keep a short weekly log starting now (what you did, what broke, how you fixed it). It becomes your individual section and makes the percentage split easy to agree on.

**Project document (10 points)**

- [ ] Overview: goal, how the system works, strengths and weaknesses (4)
- [ ] Technical: which LLMs and how they are used, data and how it is handled, RAG, the vector database, tools, third party libraries (2, 3)
- [ ] Ethics: privacy, bias, societal impact, and the mitigations from the proposal with what we measured (4)
- [ ] Individual sections, about one page each: what you did and your share of the work, challenges, how you solved them, team lessons, how you would apply them at work (all)
- [ ] Python code as a zip, plus the app URL (5)

**Presentation (5 points, 10 minutes)**

- [ ] Goal and how it works: 3 minutes (5)
- [ ] Live demo of the what if editor: 4 minutes, with a recorded backup in case the network fails (3, 5)
- [ ] Key insights, including where the simulated residents got people wrong: 2 minutes (4)
- [ ] One minute left for Q&A (all)
