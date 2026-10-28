"""Download the public I-81 Viaduct Project documents into data/raw/.

Run from the repo root: python scripts/download.py
Raw files carry commenters' names and email addresses, so data/raw/ is gitignored.
"""
import urllib.parse
import urllib.request
from pathlib import Path

WEB = "https://webapps.dot.ny.gov/"
S3 = "https://parsonsecmpublic.s3.amazonaws.com/I-81-FEIS/04-2022/"

# Comments on the final EIS, split by commenter surname. This is the answer key.
FEIS_COMMENTS = [
    "i-81-fdr-feis-comment-submissions-through-c",
    "i-81-fdr-feis-comment-submissions-d-through-k",
    "i81-fdr-feis-comment-submissions-l-through-nolan",
    "i81-fdr-feis-comment-submissions-o-through-r",
    "i81-fdr-feis-comment-submissions-s-and-t",
    "i-81-fdr-feis-comment-submissions-u-through-z",
    "i-81-fdr-feis-comment-submissions-nyclu",
]

# Record of Decision, its Appendix A (responses to FEIS comments by topic) and the June 2022 addendum.
DECISION = ["record-decision", "appendix-response-comments", "addendum-appendix-rod-2022-june"]

# Comments on the draft EIS and the state's responses, published as FEIS Appendix M.
DEIS_COMMENTS = [
    "Appendix M-5_Responses to Comments Received__April 2022.pdf",
    "Appendix M-6_Part 1_Public Hearing Transcripts__April 2022.pdf",
    "Appendix M-6_Part 2_Neighborhood Meeting Transcripts__April 2022.pdf",
    "Appendix M-6_Part 3_Voicemail Transcripts__April 2022.pdf",
    "Appendix M-6_Part 4_Online Comment Form__April 2022.pdf",
    "Appendix M-6_Part 5_Written Comment Forms__April 2022.pdf",
]

# FEIS chapters, the documents the simulated residents read through RAG.
FEIS_CHAPTERS = [
    "00 Summary_April 2022.pdf",
    "01 Introduction_April 2022.pdf",
    "02 Project Setting_April 2022.pdf",
    "03 Alternatives_April 2022.pdf",
    "04 Construction Means and Methods_April 2022.pdf",
    "05 Transportation and Engineering Considerations_April 2022.pdf",
    "06 Social Economic and Environmental Considerations_April 2022.pdf",
    "06-1 Introduction_April 2022.pdf",
    "06-2-1 Neighborhood Character_April 2022.pdf",
    "06-2-2 Social Groups Benefitted or Harmed_April 2022.pdf",
    "06-2-3 Environmental Justice_April 2022.pdf",
    "06-3-1 LandAcqDisplRelocated_April 2022.pdf",
    "06-3-2 EC-Local and Regional Economy_April 2022.pdf",
    "06-4-1 Historic and Cultural Resources_April 2022.pdf",
    "06-4-2 Parklands and Recreational Resources_April 2022.pdf",
    "06-4-3 Visual_April 2022.pdf",
    "06-4-4 Air Quality_April 2022.pdf",
    "06-4-5 Energy and Climate Change_April 2022.pdf",
    "06-4-6 Noise_April 2022.pdf",
    "06-4-7 Water Resources_April 2022.pdf",
    "06-4-8 General Ecology_April 2022.pdf",
    "06-4-9 Asbestos and Lead_April 2022.pdf",
    "06-4-10 Contaminated Materials_April 2022.pdf",
    "06-4-11 Farmlands_April 2022.pdf",
    "06-5 ShortTermvsLongTerm_April 2022.pdf",
    "06-6 Commitment of Resources_April 2022.pdf",
    "06-7 Unavoidable Impacts_April 2022.pdf",
    "07 Final Section 4f Evaluation_April 2022.pdf",
    "08 Summary of Alternatives_April 2022.pdf",
    "09 Agency Coordination and Public Outreach_April 2022.pdf",
    "10 Financing and Implementation_April 2022.pdf",
]


def sources():
    for slug in FEIS_COMMENTS:
        yield "feis_comments", slug + ".pdf", WEB + slug
    for slug in DECISION:
        yield "decision", slug + ".pdf", WEB + slug
    for name in DEIS_COMMENTS:
        yield "deis_comments", name, S3 + urllib.parse.quote(name)
    for name in FEIS_CHAPTERS:
        yield "feis_chapters", name, S3 + urllib.parse.quote(name)


def main(out=Path("data/raw")):
    for folder, name, url in sources():
        dest = out / folder / name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "silicon-town-hall/0.1"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(part, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        # The state's slugs serve the PDF directly; anything else means a moved page.
        with open(part, "rb") as f:
            if f.read(5) != b"%PDF-":
                part.unlink()
                raise SystemExit(f"not a PDF, check the link: {url}")
        part.rename(dest)
        print(f"{dest.stat().st_size:>11,}  {dest}")


if __name__ == "__main__":
    main()
