"""Step 2: split the comment files into individual comments.

Every splitter yields dicts with file, pages [first, last], kind, text and names. `names`
holds the commenter's own name strings found in headers, speaker labels and signatures;
scrub.py uses them and they are never written to disk.

Appendix M-5 is skipped: it is NYSDOT's grouped summaries and responses, not resident text.
"""

import re

from townhall.ingest.scrub import find_zip

WEB_DATE = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} [AP]M\s*$",
                      re.MULTILINE)
COUNTER = re.compile(r"^\s*(\d+) ?(?:of|/) ?(\d+)\s*$", re.MULTILINE)
# Browser and Outlook print furniture, page numbers, stray OCR characters, phone sign offs.
CHROME = re.compile(r"""^\s*(?:
    \d{1,2}/\d{1,2}/\d{2,4},\ \d{1,2}:\d{2}\ [AP]M
  | .*\ -\ dot\.sm\.mo\.I81Opportunities\ -\ Outlook
  | https?://outlook\.office365\.com\S*
  | \d+\ ?(?:of|/)\ ?\d+ | \#\ of\ \#
  | Firefox | about:blank | Set\ Up\ Voice\ Mail
  | \d{1,3} | \S | [_\-=*]{5,}
  | -+\ ?Original\ Message\ ?-+
  | Sent\ from\ (?:my|Mail|Yahoo|AOL|Outlook)\b[^.]{0,40}\.? | Get\ Outlook\ for\ \S+ | Begin\ forwarded\ message:
  | \d+\ attachments?\ \(.*\)
  | [^.!?]{0,80}\((?:DOT|FHWA)\)(?:;[^.!?]{0,80})?       # agency staff in wrapped To and Cc lines
)\s*$""", re.VERBOSE | re.IGNORECASE)
BANNER = re.compile(r"ATTENTION: This email came from an external source\..*?unexpected\s+emails\.|"
                    r"CAUTION: This email originated from outside.*?content\s+is\s+safe\.", re.DOTALL)
TRANSCRIPTION = re.compile(r"Thank you for using Transcription!.*?clear enough to transcribe\.", re.DOTALL)
PRINTER = re.compile(r"\A\s*\d{1,3}\s*\n[^\n]{2,40}\n(?=\s*From:)")  # desktop print: page no. and printing user
KEY = re.compile(r"^\s*(From|To|Cc|Bcc|Sent|Date|Attachments|Importance|Subject):\s*(.*)$", re.IGNORECASE)
ADDR = re.compile(r"<[^<>\s]+@[^<>\s]+>")
ATTACHMENT = re.compile(r"^\s*\S.*\.(?:pdf|docx?|mp3|wav|jpe?g|png|xlsx?|pptx?|msg)[;,]?\s*$", re.IGNORECASE)
ANNOUNCE = re.compile(r"The\s+New\s+York\s+State\s+Department\s+of\s+Transportation\s+\(NYSDOT\)\s+and\s+the\s+"
                      r"Federal\s+Highway\s+Administration\s+\(FHWA\)\s+have\s+published")
REPLY = re.compile(r"^(?:On [^\n]*\n?[^\n]*wrote:|From:|-+ ?Original Message|_{5,})", re.MULTILINE)
WROTE = re.compile(r"^On [^\n]{0,200}\n?[^\n]{0,200}wrote:\s*$", re.MULTILINE)
CLOSING = re.compile(r"^\s*(?:sincerely|respectfully(?: submitted)?|regards|best regards|kind regards|"
                     r"warm regards|best|thanks|thank you|thank you very much|cordially|yours truly|"
                     r"a concerned (?:resident|citizen))"
                     r"[,.!]*\s*$", re.IGNORECASE)
PERSONISH = re.compile(r"^[A-Z][A-Za-z'\-]+(?: [A-Z]\.?)?(?: [A-Z][A-Za-z'\-]+){1,2}$")
ORG_WORDS = {"chapter", "association", "council", "committee", "department", "group", "inc", "llc", "union", "project",
             "office", "center", "coalition", "partnership", "society", "board", "authority", "company", "club",
             "church", "foundation", "network", "alliance", "team", "task", "force", "grid", "save", "now", "syracuse",
             "york", "county", "city", "town", "state", "transportation", "opportunities", "subject", "no"}


def clean_name(raw):
    """'Doe, Jane (DOT)' -> 'Jane Doe'; returns '' for things that are not names."""
    raw = re.sub(r"\(.*?\)|\"|'(?=\s|$)", "", ADDR.sub("", raw)).strip(" ,;")
    if "," in raw:
        last, _, first = raw.partition(",")
        raw = f"{first.strip()} {last.strip()}"
    raw = re.sub(r"\s+", " ", raw).strip()
    if any(w.lower() in ORG_WORDS or (w.isupper() and len(w) <= 3) for w in raw.split()):
        return ""
    return raw if PERSONISH.match(raw) or (PERSONISH.match(raw.title()) and raw.isupper()) else ""


def strip_signature(lines):
    """Drop everything from the last closing line (within the last 15 lines); return kept lines, signer names."""
    for i in range(len(lines) - 1, max(-1, len(lines) - 16), -1):
        if CLOSING.match(lines[i]):
            names = [n for n in map(clean_name, lines[i + 1:i + 4]) if n]
            return lines[:i], names
    return lines, []


def cut_quote(text):
    """Cut the quoted NYSDOT announcement and anything quoted below a reply marker. Returns (text, cut?)."""
    m = ANNOUNCE.search(text)
    if m:
        markers = list(REPLY.finditer(text, 0, m.start()))
        return text[:markers[-1].start() if markers else m.start()], True
    w = WROTE.search(text)
    return (text[:w.start()], True) if w else (text, False)


def clean_email(text):
    """Headers, print furniture, quoted announcement and signature out; returns (text, names, quote_cut)."""
    text = PRINTER.sub("", text)
    text = TRANSCRIPTION.sub("", BANNER.sub("", text))
    text, quoted = cut_quote(text)
    out, names, pending = [], [], None
    for line in text.splitlines():
        s = line.strip()
        key = KEY.match(s)
        if pending and not key:  # value line under a bare "From:" style key
            if pending == "from":
                names.append(clean_name(s))
            elif pending == "subject":
                out.append(s)
            pending = None
            continue
        pending = None
        if key:
            field, value = key.group(1).lower(), key.group(2).strip()
            if not value:
                pending = field
            elif field == "from":
                names.append(clean_name(value))
            elif field == "subject":
                out.append(value)
            continue
        if ";" in s and re.search(r"\S@\S|\((?:DOT|FHWA|[A-Z]{3,})\)", s):  # wrapped recipient lists
            continue
        if ADDR.search(s):  # "Name <a@b>" sender and recipient lines
            names.append(clean_name(s.split("<")[0]))
            continue
        if not s or CHROME.match(s) or WEB_DATE.match(s) or ATTACHMENT.match(s) or s.lower() == "on behalf of" \
                or re.fullmatch(r"\S+@\S+", s):
            continue
        out.append(s)
    out, signers = strip_signature(out)
    return "\n".join(out).strip(), [n for n in names + signers if n], quoted


def email_kind(text, has_header):
    if "Petition to the NYSDOT" in text:
        return "petition"
    if "Tear Down the Structural Racism of I-81" in text:
        return "form_email"
    if "Voice Mail (" in text:
        return "voicemail"
    return "email" if has_header else "letter"


def page_start(text, prev_closed):
    """Does this FEIS page begin a new submission? Returns 'header', 'letter', 'petition' or None."""
    if "Petition to the NYSDOT" in text:
        return "petition"
    if "about:blank" in text:  # Firefox printout of an attachment: belongs to the email before it
        return None
    body = [s for s in (l.strip() for l in PRINTER.sub("", text).splitlines()) if s and not CHROME.match(s)][:3]
    if any(s.startswith("From:") for s in body):
        return "header"
    counter = COUNTER.search(text)
    if WEB_DATE.search(text) and (not counter or counter.group(1) == "1"):
        return "header"
    return "letter" if prev_closed else None


def split_emails(file, pages):
    """FEIS comment files: printed Outlook emails, forwarded voicemails, letters, petition sheets."""
    groups, prev_closed = [], True
    for page in pages:
        start = page_start(page["text"], prev_closed)
        if start or not groups or groups[-1][0] == "petition":
            groups.append([start or "letter", []])
        groups[-1][1].append(page)
        counters = COUNTER.findall(page["text"])
        prev_closed = bool(counters) and counters[-1][0] == counters[-1][1]
    for start, group in groups:
        raw = "\n".join(p["text"] for p in group)
        if start == "petition":  # keep the printed demands, drop the handwritten signature table
            raw = re.split(r"^\s*Name\b.{0,20}Signature", raw, maxsplit=1, flags=re.MULTILINE | re.IGNORECASE)[0]
        text, names, quoted = clean_email(raw)
        yield {"file": file, "pages": [group[0]["page"], group[-1]["page"]], "text": text, "names": names,
               "kind": email_kind(raw, start == "header"), "quote_cut": quoted, "zip": find_zip(raw)}


TRANSCRIPT_NOISE = re.compile(r"""^\s*(?:\d{1,2} | Page\ \d+ | 800\.523\.7887 | .*Associated\ Reporters.*
  | ARII@courtsteno\.com | www\.courtsteno\.com | .*I.?81\ Viaduct\ (?:Project|Voicemails).*
  | \((?:Off|On)\ the\ record\) | Voicemail\ Comments\ 2021.* | Transcribed\ with\ Audio.*)\s*$""", re.VERBOSE | re.IGNORECASE)
LABEL = re.compile(r"^\s*((?:MR|MS|MRS|DR|MISS)\.? [A-Z][A-Z'\-. ]*|THE REPORTER|COURT REPORTER|THE COURT REPORTER|"
                   r"MALE SPEAKER[A-Z ]*|FEMALE SPEAKER[A-Z ]*|UNIDENTIFIED SPEAKER[A-Z ]*|DATE|TIME|LOCATION|"
                   r"APPEARANCES):\s*(.*)$")
# hearing moderator, virtual hearing host, stenographers
STAFF = {"MR. SMITH", "MS. MAGUIRE", "THE REPORTER", "COURT REPORTER", "THE COURT REPORTER"}
SESSION_HEAD = {"DATE", "TIME", "LOCATION", "APPEARANCES"}
END = re.compile(r"proceeding concluded|^\s*STATE OF NEW YORK|WITNESS WHEREOF|hereby certify", re.IGNORECASE)
INDEX_LINE = re.compile(r"\b\d{1,3}:\d{1,2}\b")


def lines_of(pages):
    """(page, line) pairs with transcript furniture removed and word index pages skipped."""
    for page in pages:
        lines = [l for l in page["text"].splitlines() if l.strip()]
        if lines and sum(bool(INDEX_LINE.search(l)) for l in lines) > 0.4 * len(lines):
            continue  # concordance index at the back of each transcript: word page:line
        for line in lines:
            if not TRANSCRIPT_NOISE.match(line):
                yield page["page"], line.strip()


def split_transcript(file, pages, kind):
    """Hearing and neighborhood meeting transcripts: one comment per speaker per session, staff excluded."""
    session, speaker, turns, order = 0, None, {}, []
    for page, line in lines_of(pages):
        label = LABEL.match(line)
        if label:
            speaker = label.group(1).strip()
            if speaker in SESSION_HEAD:
                session, speaker = (session + 1 if speaker == "DATE" else session), None
                continue
            line = label.group(2)
            if speaker in STAFF:
                continue
            key = (session, speaker)
            if key not in turns:
                turns[key] = {"pages": [page, page], "lines": []}
                order.append(key)
        elif END.search(line):
            speaker = None
        if speaker and speaker not in STAFF and line:
            turn = turns[(session, speaker)]
            turn["pages"][1] = page
            turn["lines"].append(line)
    for key in order:
        surname = re.sub(r"^(?:MR|MS|MRS|DR|MISS)\.? ", "", key[1])
        named = surname != key[1]
        yield {"file": file, "pages": turns[key]["pages"], "kind": kind, "text": "\n".join(turns[key]["lines"]),
               "names": [surname.title()] if named else [], "quote_cut": False}


VM_START = re.compile(r"^(?:(.*?)_\s*\d{4}-\d{2}-\d{2}.*|Caller:.*)$")
VM_FRAGMENT = re.compile(r"^(?:\d{2}-\d{2}-\d{2}|[ap]m\.?|\.?wav|\.?mp3)$", re.IGNORECASE)


def split_voicemail(file, pages):
    """Voicemail transcripts: a new comment at every recording file name line (Name_2021-08-19.wav)."""
    current = None
    for page, line in lines_of(pages):
        start = VM_START.match(line)
        if start:
            if current:
                yield current
            current = {"file": file, "pages": [page, page], "kind": "voicemail", "lines": [],
                       "names": [n for n in [clean_name(start.group(1) or "")] if n], "quote_cut": False}
            continue
        if END.search(line):
            if current:
                yield current
            current = None
        elif current and not VM_FRAGMENT.match(line):
            current["pages"][1] = page
            current["lines"].append(line)
    if current:
        yield current


FORM_DATE = re.compile(r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)"
                       r" \d{1,2}, \d{4}$")
FORM_HEAD = {"I-81 Viaduct Project Comments from On-line Comment Form", "Date", "Name", "Comment"}


def split_online(file, pages):
    """Online comment form export: repeating Date / 'Last, First' / comment blocks."""
    lines = [(p, l) for p, l in ((p["page"], l.strip()) for p in pages for l in p["text"].splitlines())
             if l and l not in FORM_HEAD and not l.isdigit()]
    current = None
    for i, (page, line) in enumerate(lines):
        if FORM_DATE.match(line) and i + 1 < len(lines):
            if current:
                yield current
            current = {"file": file, "pages": [page, page], "kind": "online_form", "lines": [],
                       "names": [n for n in [clean_name(lines[i + 1][1])] if n], "name_line": i + 1,
                       "quote_cut": False}
        elif current and i != current["name_line"]:
            current["pages"][1] = page
            current["lines"].append(line)
    if current:
        yield current


FORM_FIELD = re.compile(r"^\W*(?:Name|Nombre|Address|Direcci.n|City|Ciudad|State|Estado|Zip|C.digo|Phone|N.mero|Email|"
                        r"Correo)\b", re.IGNORECASE)
FORM_END = re.compile(r"Mail comments to|Envie sus comentarios", re.IGNORECASE)


def form_comment(text):
    """Comment sheets: keep what follows 'Comments:' (all of it when OCR lost that label), minus personal fields."""
    parts = re.split(r"Comments?:|Comentarios:", text, maxsplit=1)
    rest = FORM_END.split(parts[-1])[0]
    return "\n".join(l for l in rest.splitlines() if l.strip() and not FORM_FIELD.match(l))


def split_written(file, pages):
    """Scanned written comment sheets and flyers: one comment per page, cover and blank pages skipped.
    shortcut: handwriting defeats both the PDF text layer and Tesseract; a vision model would read these."""
    for page in pages[1:]:
        if page["words"] >= 5:
            kind = "flyer" if "Speak Your Truth" in page["text"] else "written_form"
            yield {"file": file, "pages": [page["page"]] * 2, "kind": kind, "text": form_comment(page["text"]).strip(),
                   "names": [], "quote_cut": False, "zip": find_zip(page["text"])}


def split_file(file, pages):
    if file.startswith("feis_comments/"):
        items = split_emails(file, pages)
    elif "Part 1_" in file:
        items = split_transcript(file, pages, "hearing")
    elif "Part 2_" in file:
        items = split_transcript(file, pages, "neighborhood_meeting")
    elif "Part 3_" in file:
        items = split_voicemail(file, pages)
    elif "Part 4_" in file:
        items = split_online(file, pages)
    elif "Part 5_" in file:
        items = split_written(file, pages)
    else:
        return []
    out = []
    for item in items:
        if "lines" in item:
            lines = item.pop("lines")
            item["zip"] = find_zip("\n".join(lines))
            lines, signers = strip_signature(lines)
            item["text"], item["names"] = "\n".join(lines), item["names"] + signers
        item.pop("name_line", None)
        item.setdefault("zip", find_zip(item["text"]))
        if item["text"].strip():
            out.append(item)
    return out


def split_all(pages):
    by_file = {}
    for page in pages:
        by_file.setdefault(page["file"], []).append(page)
    return [c for file, file_pages in by_file.items() for c in split_file(file, file_pages)]
