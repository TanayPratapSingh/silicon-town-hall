# Silicon Town Hall: Proposal

*A simulated town hall for government teams to test new policies before real outreach*

Governments usually hear how residents feel about a new policy late, and from a self-selected few. Silicon Town Hall is a planning tool: a government team uploads a draft policy, a town of AI residents built from Census data reads it and debates it in neighborhood meetings, and the team gets a feedback report showing which concerns come up, which neighborhoods and groups raise them, and who to reach in person. It is a rehearsal before real outreach, not a replacement for it.

## Who it is for

City and state planning staff, transportation agencies, school districts and public health offices that need an early read on a new plan, zoning change, budget, service cut or public notice before they hold hearings. Any document the team uploads can be tested.

## What the app does

1. **Choose a community.** The user picks a place and the app samples a few hundred residents from Census ACS microdata (age, race, income, commute, home ownership, disability), each with a home neighborhood.
2. **Upload the policy.** The user adds any document: a plan, ordinance, budget or notice. Every page is indexed as text and as a page image in a vector database, since many pages are maps and tables.
3. **Run the meetings.** The user sets the neighborhoods and rounds. A facilitator agent runs each meeting (multi agent, LangChain); residents read the policy, hear each other and can change their minds.
4. **Read the report.** An analyst agent tags each comment by topic (NYSDOT's codes for I-81, or the team's own list) and builds a dashboard: topics by neighborhood and group, sample comments marked simulated with page citations, and a list of who to reach in person.
5. **Try a change.** The user edits the policy, for example by moving an interchange, reruns the meetings, and compares reports side by side. Scenarios are saved.

## Under the hood

- **Retrieval and reranking (RAG).** Each resident turns its profile and what it just heard into a query. Retrieval runs on text, page images, or both, merging the top 50 by reciprocal rank fusion. A text cross-encoder trims 50 to 20, then a vision model scores the top 10 page images. Pages already cited get a small boost.
- **Short term memory (one meeting):** last few turns, a running summary, and the pages already cited; it feeds the query and the reranker boost.
- **Long term memory (across rounds, SQLite + embeddings):** each resident's stance, reasons and past comments, saved after every round. The top 3 relevant memories return next round and newer opinions replace older ones, so a resident ignored in round one still remembers it in round two.

**Built with:** Census API sampler, document indexer, topic tagger, coverage scorer, plan editor, report builder, and a Streamlit app behind a password.

## Pilot: I-81 in Syracuse

We preload the plan to replace the I-81 viaduct as the demo community. NYSDOT published the public comments and answered them by topic (Appendix A), so the app can show how closely its feedback matched what real residents said: the share of real topics the residents raised, the gap in topic mix, and the concerns only real residents raised. These scores appear beside every report. The pilot uses the comments on the final impact statement (822 pages; 218 are scans needing OCR), the comments on the draft (780 pages, including 233 pages of neighborhood meeting transcripts), and the 31 impact statement chapters (1,363 pages) the residents read.

## Responsible use

The biggest risk is a government treating simulated comments as real public input. Every output and export is labeled simulated, and the app is a planning aid for deciding who to reach in person, never a record of public comment. Reports show group by group where the pilot's simulated concerns differed from real ones, including the lead and environmental justice concerns community groups raised, and name the groups missing from both sets. Real comments are stripped of names and emails before storage, and only topic counts are published. No feature aims messages at a group, and users can view and delete saved scenarios and memories.

## Sources

- NYSDOT, I-81 Viaduct Project document library (comments on the final impact statement)
- NYSDOT, Final Environmental Impact Statement, April 2022, with Appendix M (comments on the draft)
- NYSDOT, Record of Decision and Appendix A, Responses to Comments, June 2022
- U.S. Census Bureau, [ACS 5 year microdata API](https://api.census.gov/data/2023/acs/acs5/pums/variables/PUMA.json)
