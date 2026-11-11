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

Not started: OCR, comment tagging, personas, agents, app. There are no results yet.

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

## Layout

| Path | What |
| --- | --- |
| `docs/` | Proposal, roadmap, data sources |
| `scripts/download.py` | Fetches every source PDF and checks each one is really a PDF |
| `scripts/pdf_stats.py` | Pages, words and OCR pages per PDF |
| `runs/` | Outputs of commands; every number in `docs/` traces to a file here |
| `data/` | Downloaded and processed data. Gitignored, because public comments carry names and email addresses |
