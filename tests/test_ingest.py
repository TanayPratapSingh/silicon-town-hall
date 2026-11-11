"""Offline tests for the answer key pipeline: synthetic text only, no PDFs, no network."""

import csv
import json

import pytest

from townhall.ingest import codebook, scrub, split, tagger
from townhall.llm import FakeLLM

ANNOUNCEMENT = """The New York State Department of Transportation (NYSDOT)
and the Federal Highway Administration (FHWA) have published
a Final Environmental Impact Statement for the I-81 Viaduct Project."""

WEB_EMAIL = f"""Say NO to the Grid!
Jane Roe <jroe@example.com>
Sat 4/16/2022 11:42 AM
To: dot.sm.mo.I81Opportunities <dot.sm.mo.I81Opportunities@dot.ny.gov>
ATTENTION: This email came from an external source. Do not open attachments or click on links
from unknown senders or unexpected
emails.
Keep the viaduct. Traffic on 481 will be awful.
Thank you,
Jane Roe
12 Maple Street
Syracuse, NY 13210
Sent from my iPhone
On Apr 15, 2022, at 9:24 AM, dot.sm.mo.I81Opportunities
<dot.sm.mo.I81Opportunities@dot.ny.gov> wrote:
{ANNOUNCEMENT}
1 of 1"""

FORWARD = """From:
Quill, Pat (DOT)
To:
Stone, Lee A (DOT)
Subject:
FW: We need more time!
Date:
Tuesday, May 3, 2022 7:29:45 AM
-----Original Message-----
From: Paul Doe <pdoe@example.com>
Sent: Tuesday, May 3, 2022 7:09 AM
Subject: We need more time!
Please extend the review period."""


def page(n, text, file="feis_comments/test.pdf"):
    return {"file": file, "page": n, "text": text, "words": len(text.split())}


def test_clean_email_strips_header_banner_quote_and_signature():
    text, names, quoted = split.clean_email(WEB_EMAIL)
    assert quoted
    assert text == "Say NO to the Grid!\nKeep the viaduct. Traffic on 481 will be awful."
    assert "Jane Roe" in names


def test_forward_keeps_original_message_and_drops_staff_header():
    text, names, _ = split.clean_email(FORWARD)
    assert "Please extend the review period." in text
    assert "Stone" not in text and "Original Message" not in text
    assert {"Pat Quill", "Paul Doe"} <= set(names)


def test_split_emails_by_page_headers_and_attachments():
    pages = [page(1, WEB_EMAIL), page(2, "Firefox\nabout:blank\nattached scan text\n1 of 1"),
             page(3, FORWARD), page(4, "Petition to the NYSDOT\nWe demand a health needs assessment.\n"
                                       "Name Signature Address\nA Person 1 Elm St")]
    comments = list(split.split_emails("feis_comments/test.pdf", pages))
    assert [c["pages"] for c in comments] == [[1, 2], [3, 3], [4, 4]]
    assert [c["kind"] for c in comments] == ["email", "email", "petition"]
    assert "A Person" not in comments[2]["text"]
    assert comments[0]["zip"] == "13210"


def test_transcript_one_comment_per_speaker_without_staff():
    text = """DATE: August 17, 2021
MR. SMITH: Our next speaker is Ann Lee.
MS. LEE: My name is Ann Lee. The grid will
help the 15th Ward.
MR. SMITH: Thirty seconds.
MS. LEE: Thank you.
MR. DOE: Keep the bridge.
(The proceeding concluded.)
STATE OF NEW YORK
I, A REPORTER, do hereby certify"""
    comments = list(split.split_transcript("deis_comments/Part 1_x.pdf", [page(1, text)], "hearing"))
    assert [c["names"] for c in comments] == [["Lee"], ["Doe"]]
    assert comments[0]["text"] == "My name is Ann Lee. The grid will\nhelp the 15th Ward.\nThank you."
    assert "certify" not in comments[1]["text"]


def test_voicemail_and_online_form_splits():
    vm = "Roe, Jane_2021-08-19.wav\nKeep 81.\n3154463531_2021-10-14.mp3\nGrid only."
    comments = list(split.split_voicemail("Part 3_x.pdf", [page(1, vm)]))
    assert [c["lines"] for c in comments] == [["Keep 81."], ["Grid only."]]
    assert comments[0]["names"] == ["Jane Roe"]
    form = "Date\nName\nComment\nOctober 13, 2021\nRoe, Jane\nBuild the grid.\nJuly 20, 2021\nDoe, John\nNo."
    comments = list(split.split_online("Part 4_x.pdf", [page(1, form)]))
    assert [(c["names"], c["lines"]) for c in comments] == [(["Jane Roe"], ["Build the grid."]),
                                                          (["John Doe"], ["No."])]


def test_scrub_emails_phones_addresses():
    text = "Reach me at jroe@example.com or (315) 555-0142 or 315‐555‐0199. I live at 12 Maple Street, Syracuse."
    out, counts = scrub.scrub(text, use_ner=False)
    assert out == "Reach me at [EMAIL] or [PHONE] or [PHONE]. I live at [ADDRESS], Syracuse."
    assert counts == {"EMAIL": 1, "PHONE": 2, "ADDRESS": 1}
    assert scrub.scrub("Keep 81 as a raised highway.", use_ner=False)[0] == "Keep 81 as a raised highway."


def test_scrub_names_from_every_layer():
    lex = scrub.Lexicon(["Jane Roe", "Paul Doe", "Ann Lee"], place_text="Almond Street. Almond Street. Almond Street.")
    text = ("Dear Mr. Quill, I am Ann and Paul Doe agrees. Lee, Ann signed too. My name is Ada Byron, A-D-\nA. "
            "Jane wrote this. Fix Almond Street near Dr. King Elementary on Martin Luther King East.")
    out, counts = scrub.scrub(text, own=["Jane Roe"], lex=lex, use_ner=False)
    assert out == ("Dear Mr. [NAME], I am [NAME] and [NAME] agrees. [NAME] signed too. My name is [NAME], [NAME]. "
                   "[NAME] wrote this. Fix Almond Street near Dr. King Elementary on Martin Luther King East.")
    assert counts["NAME"] == 7


def test_scrub_with_spacy_ner():
    pytest.importorskip("spacy")
    lex = scrub.Lexicon(["Mary Smith"])
    out, _ = scrub.scrub("I spoke with Mary Johnson about the grid, and with P. Smith on S. Smith Street.", lex=lex)
    assert out == "I spoke with [NAME] about the grid, and with [NAME] on S. Smith Street."


def test_find_zip_skips_nysdot_office():
    raw = "Mail to 333 E. Washington Street\nSyracuse, NY 13202\nJane, Liverpool, N.Y. 13088"
    assert scrub.find_zip(raw) == "13088"
    assert scrub.find_zip("no zip here") == ""


CODEBOOK = [{"code": "4", "topic": "Community Grid Alternative", "definition": "Support for the grid."},
            {"code": "7-5", "topic": "Noise", "definition": "Noise mitigation."}]


def test_tagger_parses_json_and_drops_unknown_codes():
    assert tagger.parse('{"codes": ["7-5", "4", "99"]}', CODEBOOK) == ["4", "7-5"]
    assert tagger.parse('["7-5"]', CODEBOOK) == ["7-5"]
    assert tagger.parse("not json", CODEBOOK) == []
    assert tagger.parse('{"codes": "4"}', CODEBOOK) == []


def test_tag_with_fake_llm():
    llm = FakeLLM(responder=lambda messages: {"codes": [7.5, "7-5"]})
    assert tagger.tag("The noise on 481 is unbearable.", CODEBOOK, llm) == ["7-5"]
    user = llm.calls[0][1]["content"]
    assert "7-5: Noise" in user and "unbearable" in user


def test_handcheck_sample_is_blind_and_seeded(tmp_path):
    comments = [{"id": f"c{i:05d}", "file": "f.pdf", "pages": [i, i], "kind": "email", "text": f"t{i}"}
                for i in range(300)]
    path = tmp_path / "handcheck.csv"
    first = tagger.write_handcheck(comments, path)
    assert first == tagger.write_handcheck(comments, path) and len(first) == 100
    rows = list(csv.DictReader(path.read_text().splitlines()))
    assert len(rows) == 100 and all(r["human_codes"] == "" for r in rows)
    assert list(rows[0]) == ["id", "file", "pages", "kind", "text", "human_codes"]


def test_codebook_first_line_prefers_group_summary():
    section = ("C 3-1: Keep it. R 3-1: See response to Comment 4-2. Commenters expressed the opinion that the viaduct "
               "should be rebuilt. More text.")
    assert codebook.first_line(section) == "Commenters expressed the opinion that the viaduct should be rebuilt."
    assert codebook.first_line("R 5-1: See response to Comment 5-10.") == ""
    toc = "2 \nProject Purpose, Objectives, and Need ....... 1 \n7-5 Noise ........ 20 "
    assert [(c, t) for c, t, _ in codebook.TOC.findall(toc)] == [("2", "Project Purpose, Objectives, and Need"),
                                                                ("7-5", "Noise")]


def test_tagger_cli_exits_cleanly_without_key(monkeypatch, capsys):
    monkeypatch.setattr(tagger.config, "OPENAI_API_KEY", "")
    assert tagger.main([]) == 0
    assert "not set" in capsys.readouterr().out


def test_committed_codebook_is_valid_json():
    topics = json.loads(codebook.OUT.read_text())
    assert topics and all({"code", "topic", "definition"} <= set(t) for t in topics)
