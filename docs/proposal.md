# Silicon Town Hall: IST 688 Final Project Proposal

Oct 27, 2026

We propose Silicon Town Hall, a population of LLM agents built from Census data for Syracuse that reads the I-81 viaduct project documents and writes public comments. The question we test: do simulated residents raise the same concerns real residents did, and whose voices do they miss?

## Why it matters

Large public projects must collect public comment and answer it in writing. For the I-81 viaduct, NYSDOT answered comments on the final environmental impact statement in an 88 page appendix sorted by topic, from noise and air quality to displacement and the local economy ([Appendix A](https://webapps.dot.ny.gov/appendix-response-comments)).

Research is split on whether LLMs can stand in for people. Argyle et al. found that a model conditioned on real demographic backstories reproduced survey answers by subgroup ([Out of One, Many](https://arxiv.org/abs/2209.06899)). Wang et al. found that LLMs misportray and flatten identity groups and urge caution when they replace human participants ([Wang, Morgenstern and Dickerson](https://arxiv.org/abs/2402.01908)). If synthetic residents work, planners could learn who to reach before a hearing. If they fail, that failure should be measured before anyone relies on them. I-81 gives us a real decision with a public record of what people actually said.

## How it works

1. **Build the population.** Sample a few hundred residents from Census microdata for Syracuse and its suburbs (age, race, income, commute mode, homeownership, disability) and turn each into a persona with a home neighborhood.
2. **Read the project (RAG).** Each agent retrieves the parts of the environmental impact statement that touch its life (its street, commute, school, health), reranked for relevance to that persona.
3. **Deliberate (multi agent).** Agents meet in simulated neighborhood meetings run by a facilitator agent, hear each other, and can change their minds.
4. **Comment.** Each agent writes a public comment in its own words.
5. **Score.** An analyst agent tags simulated and real comments with NYSDOT's own topic codes, then compares coverage and lists the concerns real residents raised that no agent did.
6. **What if.** A user edits the plan (for example, moves an interchange) and reruns the town hall to see who objects and why.

Short term memory holds what each agent heard in the current meeting. Long term memory keeps each agent's opinions and past comments across rounds, so a resident ignored in round one carries that into round two, and saves each user's scenarios.

## How it meets the requirements

| Requirement | How Silicon Town Hall meets it |
| --- | --- |
| Short term memory | Each agent remembers the current meeting; the user's session keeps the scenario under test |
| Long term memory | Agents keep opinions and past comments across rounds; scenarios saved per user |
| RAG and reranking | Impact statement chapters in a vector database, retrieved for each persona and reranked |
| Tools and functions | Census API sampler, topic tagger, coverage scorer, plan scenario editor |
| Ethics rubric evaluation | Scored against the course ethics criteria (see Ethics) |
| Presentation topics | Multi agent systems (residents, facilitator, analyst), orchestrated with LangChain |
| Available on the web | Streamlit app deployed online behind a password |

## Data and evaluation

| Source | What we use it for |
| --- | --- |
| [Final EIS and Record of Decision](https://webapps.dot.ny.gov/record-decision-0) (NYSDOT, 2022) | The project documents agents read through RAG |
| [FEIS comment submissions](https://webapps.dot.ny.gov/i-81-fdr-feis-comment-submissions-through-c) | Real comments to score against; the A to C file alone is 118 pages, about 20,600 words of extractable text, with 25 pages needing OCR |
| [Appendix A, Responses to Comments](https://webapps.dot.ny.gov/appendix-response-comments) | The answer key: NYSDOT's topic codes, plus letters from groups such as the EPA, City of Syracuse, NYCLU and Families for Lead Freedom Now! |
| [Census ACS microdata API](https://api.census.gov/data/2023/acs/acs5/pums/variables/PUMA.json) | Resident personas |

We will report three numbers: the share of real topics that simulated residents raise, how far the simulated topic mix sits from the real one, and which concerns only real residents raised. Two baselines keep the result honest: one generic LLM with no personas, and personas with no deliberation. The page and word counts above came from our own text extraction on Oct 27, 2026.

## Ethics

The central risk is that simulated voices get treated as real ones, so the project measures its own failures rather than hiding them.

| Risk | Mitigation |
| --- | --- |
| Replacement: a city treats simulated comments as public input | Every output is labeled simulated; the app is framed as a rehearsal tool for finding who to reach, never a substitute |
| Flattening and stereotypes (Wang et al.) | Report per group where simulated concerns diverge from real ones, including the lead and environmental justice concerns raised by community groups |
| Privacy: real comments carry names and email addresses | Strip names, emails and addresses before storing; publish only aggregate topic counts |
| A biased answer key: people who comment are not a random sample | Treat real comments as what was said, not what everyone thinks; note who is missing from both sets |
| Misuse: tuning messages to wear down opposition | No feature targets persuasion at a group; outputs stay at the level of concerns |

## Sources

- NYSDOT, [I-81 Viaduct Project Record of Decision](https://webapps.dot.ny.gov/record-decision-0) and [Appendix A, Responses to Comments](https://webapps.dot.ny.gov/appendix-response-comments), June 2022
- NYSDOT, [FDR/FEIS comment submissions, A through C](https://webapps.dot.ny.gov/i-81-fdr-feis-comment-submissions-through-c)
- Argyle et al., [Out of One, Many: Using Language Models to Simulate Human Samples](https://arxiv.org/abs/2209.06899)
- Wang, Morgenstern and Dickerson, [Large language models that replace human participants can harmfully misportray and flatten identity groups](https://arxiv.org/abs/2402.01908)
- U.S. Census Bureau, [ACS 5 year microdata API](https://api.census.gov/data/2023/acs/acs5/pums/variables/PUMA.json)
