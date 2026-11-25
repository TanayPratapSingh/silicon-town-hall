# Silicon Town Hall: working context

A planning tool: a government team uploads a draft policy, AI residents built from Census data read it and debate it in neighborhood meetings, and the team gets a feedback report of concerns by neighborhood and group and who to reach in person. The I-81 viaduct is the pilot, scored against the real comments NYSDOT published. Read `README.md`, `docs/project.md` and `docs/proposal.md` before structural changes.

## House rules

1. **Never commit `data/`.** Public comments carry names and email addresses. Only aggregate counts go in `runs/`.
2. **Never invent a metric.** Every number in `docs/` or the README traces to a command and a file in `runs/`. Pilot numbers are labeled as pilot numbers in the same sentence.
3. **Report failures.** Where simulated residents miss or misportray a group, that is a result, not something to tune away.
4. **No em dashes** in any file, comment, commit message or output, and no hyphens in compound phrases like "real time" in prose.
5. **Label every simulated output as simulated** in the app and in the docs.

## Commands

```bash
python scripts/download.py    # source PDFs into data/raw/
python scripts/pdf_stats.py   # runs/pdf_stats.csv
```
