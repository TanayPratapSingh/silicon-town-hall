# Silicon Town Hall

Can AI residents built from Census data predict what Syracuse said about the I-81 viaduct project, and whose voices do they miss?

IST 688, Building Human Centered AI Applications, Syracuse University, Fall 2026.

## In plain words

Syracuse is replacing the I-81 highway bridge through downtown. Before deciding, the state collected public comments and published them, sorted by topic: noise, pollution, traffic, people losing their homes.

We build a town of a few hundred AI residents, each based on real Census data. Each one reads the parts of the project plan that affect its own life, talks with neighbors in a simulated meeting, and writes a public comment. Then we compare those comments with what real residents wrote. The interesting part is what the AI town misses: research says LLMs tend to flatten people into stereotypes, so we measure where that happens.

The app is a rehearsal tool, not a replacement for real public input. Every output is labeled simulated.

## Status, Oct 27, 2026

Done:

- Proposal drafted: [docs/proposal.md](docs/proposal.md), due Nov 8
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
