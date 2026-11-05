# Silicon Town Hall: IST 688 Final Project Proposal

We will build a town of AI residents for Syracuse, have them react to the I-81 viaduct plan, and check their comments against what real residents told the state. We want to know whether LLM agents built from Census data can predict a real community's concerns, and whose concerns they miss.

## Why I-81

Syracuse is tearing down the elevated I-81 viaduct through downtown and replacing it with a street grid. Before the decision, NYSDOT collected public comments, published them, and answered them by topic, from noise and air quality to people losing their homes ([Appendix A](https://webapps.dot.ny.gov/appendix-response-comments)). That record is our answer key.

Research disagrees on whether LLMs can stand in for people. Argyle et al. got a model to reproduce survey answers by subgroup when they gave it real demographic backstories ([Out of One, Many](https://arxiv.org/abs/2209.06899)). Wang et al. found that LLMs flatten identity groups, and they warn against using LLMs to replace human participants ([Wang, Morgenstern and Dickerson](https://arxiv.org/abs/2402.01908)). I-81 lets us test both claims on a real decision.

## How it works

1. **Make the residents.** Sample a few hundred people from Census microdata for Syracuse and its suburbs: age, race, income, how they get to work, whether they own a home, disability. Each one becomes an agent with a home neighborhood.
2. **Let them read the plan (RAG).** Each resident pulls the parts of the environmental impact statement that touch its life, such as its street, commute, school or health. A reranker puts the most relevant passages first.
3. **Hold neighborhood meetings (multi agent).** A facilitator agent runs each meeting. Residents hear each other and can change their minds.
4. **Write comments.** Each resident writes a public comment in its own words.
5. **Score them.** An analyst agent tags the simulated and the real comments with NYSDOT's topic codes, and we compare the two.
6. **Try a change.** A user edits the plan, for example by moving an interchange, and reruns the meetings to see who objects and why.

Short term memory holds what each resident heard in the current meeting. Long term memory keeps each resident's opinions and past comments between rounds, so a resident who was ignored in round one still remembers it in round two.

## Project requirements

| Requirement | Where it shows up |
| --- | --- |
| Short term memory | Each resident remembers the current meeting |
| Long term memory | Residents keep their opinions and past comments across rounds; users keep saved scenarios |
| RAG and reranking | Impact statement chapters in a vector database, retrieved for each resident and reranked |
| Tools | Census API sampler, topic tagger, coverage scorer, plan editor |
| Ethics rubric evaluation | Scored against the course ethics criteria (see Ethics) |
| Another presentation topic | Multi agent systems, built with LangChain |
| On the web | Streamlit app behind a password |

## Data and scoring

| Source | Size | Use |
| --- | --- | --- |
| [Comments on the final impact statement](https://webapps.dot.ny.gov/i-81-project-library) (7 files) | 822 pages, about 162,000 words | Main answer key |
| [Comments on the draft impact statement](https://static.parsons.com/I-81-FEIS/04-2022) (hearings, neighborhood meetings, voicemail, online and written forms) | 780 pages, about 324,000 words | Second answer key; its 233 pages of neighborhood meeting transcripts show what real versions of our meetings sounded like |
| [Appendix A, Responses to Comments](https://webapps.dot.ny.gov/appendix-response-comments) | 88 pages | NYSDOT's topic codes |
| [Final impact statement chapters](https://static.parsons.com/I-81-FEIS/04-2022) | 31 files, 1,363 pages | What the residents read |
| [Census ACS microdata API](https://api.census.gov/data/2023/acs/acs5/pums/variables/PUMA.json) | Queried as needed | Resident profiles |

We downloaded every file above and measured it with our own text extraction on Oct 27, 2026 ([runs/pdf_stats.csv](../runs/pdf_stats.csv)). About a quarter of the final comment pages (218 of 822) are scans and need OCR before we can tag them.

We will report two numbers and one list: the share of real topics the simulated residents raise, how far their topic mix is from the real one, and the concerns only real residents raised. Two baselines check whether the residents add anything: one LLM with no personas, and residents who skip the meetings. The model may also have read about the I-81 debate during training. If the LLM with no personas scores as well as our residents, the residents are probably repeating what the model already knew.

## Ethics

The biggest risk is a city treating simulated comments as real public input.

| Risk | What we do |
| --- | --- |
| Simulated comments replace real input | Label every output as simulated, and present the app as a planning aid for finding who to reach in person |
| Stereotyped residents (Wang et al.) | Report, group by group, where simulated concerns differ from real ones, including the lead and environmental justice concerns that community groups raised |
| Real comments include names and email addresses | Strip them before storing anything; publish only topic counts |
| People who comment are not a random sample of Syracuse | Describe real comments as the views of the people who commented, and name the groups missing from both sets |
| Using the app to wear down opposition | No feature aims messages at a group |

## Sources

- NYSDOT, [I-81 Viaduct Project document library](https://webapps.dot.ny.gov/i-81-project-library): the seven files of comments on the final impact statement
- NYSDOT, [Final Environmental Impact Statement, April 2022](https://static.parsons.com/I-81-FEIS/04-2022): chapters and Appendix M (comments on the draft)
- NYSDOT, [Record of Decision](https://webapps.dot.ny.gov/record-decision-0) and [Appendix A, Responses to Comments](https://webapps.dot.ny.gov/appendix-response-comments), June 2022
- Argyle et al., [Out of One, Many: Using Language Models to Simulate Human Samples](https://arxiv.org/abs/2209.06899)
- Wang, Morgenstern and Dickerson, [Large language models that replace human participants can harmfully misportray and flatten identity groups](https://arxiv.org/abs/2402.01908)
- U.S. Census Bureau, [ACS 5 year microdata API](https://api.census.gov/data/2023/acs/acs5/pums/variables/PUMA.json)
