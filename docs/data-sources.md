# Data sources

Every count here comes from `python scripts/pdf_stats.py`, run Oct 27, 2026, and is saved in [runs/pdf_stats.csv](../runs/pdf_stats.csv). Words are what pypdf extracts. A page with 20 words or fewer is counted as needing OCR.

## Summary

| Group | What it is | Files | Pages | Words extracted | Pages needing OCR |
| --- | --- | --- | --- | --- | --- |
| `feis_comments` | Comments on the final EIS, April to May 2022. The answer key. | 7 | 822 | 161,963 | 218 |
| `deis_comments` | Draft EIS stage: the state's responses (M-5) plus hearing, neighborhood meeting, voicemail, online and written comments (M-6) | 6 | 1,310 | 574,299 | 14 |
| `decision` | Record of Decision, Appendix A (responses to FEIS comments by topic), June 2022 addendum | 3 | 137 | 42,828 | 36 |
| `feis_chapters` | Final EIS chapters, the documents residents read through RAG | 31 | 1,363 | 624,107 | 49 |

`python scripts/download.py` fetches all 47 files, about 610 MB, into `data/raw/`. That folder is gitignored because the comment files carry names and email addresses.

## What we learned

- **The FEIS comments are printed Outlook emails.** Many replies quote NYSDOT's announcement underneath, so quoted text must be stripped before tagging or the tagger will tag the state's words as a resident's.
- **OCR is a workstream, not an edge case.** 218 of the 822 FEIS comment pages have no text layer. The worst files are O through R (70 of 174 pages) and NYCLU (48 of 99).
- **The Record of Decision itself is a scan** (35 pages, 0 words extracted). Appendix A, the part we need, extracts cleanly (88 pages, 36,746 words).
- **The draft EIS stage is the bigger answer key.** It holds 574,299 words, about 3.5 times the FEIS comments. It includes 233 pages of neighborhood meeting transcripts, the closest real match to our simulated neighborhood meetings.
- **pypdf warned that fontTools is missing** for some chapter fonts. Spot check chapter text before indexing it.

## Appendix A topics: seed for the codebook

From Appendix A's table of contents (Responses to Comments Received on the FDR/FEIS, June 2022):

1. Introduction
2. Project Purpose, Objectives, and Need
3. Viaduct Alternative
4. Community Grid Alternative
5. Other Potential Alternatives
6. Transportation Considerations
7. Social, Economic, and Environmental Considerations
    - 7-1 Land Acquisition, Displacement, and Relocation
    - 7-2 Local and Regional Economies
    - 7-3 Historic and Cultural Resources
    - 7-4 Air Quality, Energy, and Climate Change
    - 7-5 Noise
    - 7-6 Natural Resources
    - 7-7 Construction
8. Public Outreach
9. Letters from Organizations and Agencies: USEPA; City of Syracuse; Families for Lead Freedom Now! and Signatories; Upstate Medical University Occupational Health Clinical Center; NYCLU; form letters and a petition via NYCLU; municipality letters; Town of DeWitt Supervisor; AIA Central New York; Pyramid Management Group and Destiny USA
10. List of Commenters

## Census

- The ACS 5 year microdata variables endpoint responds: <https://api.census.gov/data/2023/acs/acs5/pums/variables/PUMA.json>
- A record query (`get=AGEP,SEX,RAC1P,PINCP,JWTRNS&for=public use microdata area:*&in=state:36&AGEP=40`) returned no rows on Oct 27. Still to fix: find the PUMA codes for Syracuse and Onondaga County and query those directly.

## Open questions

- The comment files jump from "L through Nolan" to "O through R". Check whether surnames after Nolan sit inside one of them.
- How many separate comments each file holds is unknown until the emails are split.
- Confirm how M-5 pairs each draft EIS comment with its response before using it as labels.

## Where the files come from

- [I-81 project library](https://webapps.dot.ny.gov/i-81-project-library): the seven FEIS comment files
- [Record of Decision page](https://webapps.dot.ny.gov/record-decision-0): ROD, Appendix A, June 2022 addendum
- [Final EIS index](https://static.parsons.com/I-81-FEIS/04-2022): chapters and Appendix M
