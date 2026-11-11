"""Step 3: replace emails, phones, street addresses and personal names with [EMAIL], [PHONE], [ADDRESS], [NAME].

Names come from five layers, all collected as character spans on the original text and then applied once:
  A. full names known anywhere in the corpus (headers, signatures, speaker labels, NYSDOT's lists of commenters),
     and a known first name followed by a capitalized word that is not a place word
  B. tokens of this comment's own names (sender, signer, speaker)
  C. honorifics: Mr., Ms., Dr. plus a capitalized word, and names spelled out letter by letter
  D. self introductions: "my name is ...", "this is <first name> ..."
  E. spaCy PERSON entities backed by a known first name, and known surnames that never appear in lower case
Place words (capitalized tokens frequent in the impact statement chapters, such as Almond, Oakwood, Destiny)
do not count as name evidence on their own, a candidate followed by a place type (Street, Square, USA) is left
alone, and "Martin Luther King" and "Dr. King" (street, school) are protected.
"""

import re
from collections import Counter

EMAIL = re.compile(r"[\w.+%=-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<![\w-])(?:\+?1[\s.\-‐-―]?)?\(?\d{3}\)?[\s.\-‐-―]?\d{3}[\s.\-‐-―]?\d{4}(?![\w-])")
SUFFIX = (r"(?i:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Place|Pl|Way|Terrace|Ter|"
          r"Circle|Cir|Parkway|Pkwy|Square|Sq|Trail)")
# house number, then capitalized street words on one line, then a suffix; or a PO box
ADDRESS = re.compile(rf"\b\d{{1,6}}(?:-\d+)?\s+(?:[NSEW]\.?[ \t]+)?(?:[A-Z][\w'.\-]*,?[ \t]+){{1,4}}{SUFFIX}\b\.?"
                     rf"(?:,?[ \t]+(?i:Apt|Suite|Unit|Ste|#)\.?[ \t]*\w+)?|\b(?i:P\.?[ ]?O\.?[ ]+Box)[ ]+\d+")
ZIP = re.compile(r"\b(?:NY|N\.Y\.|New York)[\s,.]+(1[0-4]\d{3})(?:-\d{4})?\b|\bZip:?\s*(1[0-4]\d{3})\b")
WORD = r"[A-Z][\w'’\-]+"
FULL = re.compile(rf"(?=\b({WORD})\s+(?:[A-Z]\.?\s+)?({WORD})\b)")
LAST_FIRST = re.compile(rf"(?=\b({WORD}),\s+({WORD})\b)")
HONORIFIC = re.compile(rf"\b(?:Mr|Mrs|Ms|Miss|Dr|Mx)\.?\s+({WORD}(?:\s+{WORD})?)")
INTRO = re.compile(rf"\b((?i:my name is|my name's|this is|i am|i'm))\s+"
                   rf"({WORD}(?:\s+[A-Z](?:\.|\b))?(?:\s+{WORD}){{0,2}})")
INITIAL = re.compile(r"\b[A-Z]\.\s?(?:[A-Z]\.\s?)?([A-Z][a-z][\w'’\-]*)\b")  # "P. Smith", "R. M. Walker"
SPELLED = re.compile(r"\b[A-Z](?:\s*-\s*[A-Z]){2,}\b")  # hearing speakers spell their names: S-M-I-T-H
PROTECT = re.compile(r"(?:Dr\.?\s+)?Martin\s+Luther\s+King(?:,?\s+Jr\.?)?|Dr\.?\s+King|\bMLK\b")
TOKEN = re.compile(r"[A-Za-z][\w'’\-]+")
PLACE_TYPES = {"street", "st", "avenue", "ave", "road", "rd", "drive", "dr", "lane", "ln", "boulevard", "blvd",
               "square", "sq", "park", "cemetery", "school", "elementary", "center", "building", "mall", "usa", "creek",
               "lake", "hall", "airport", "tower", "towers", "homes", "plaza", "circle", "county", "city", "town",
               "village", "bridge", "corners", "reservation", "armory", "university", "college", "hospital", "memorial",
               "federal"}
_NLP = None


def nlp():
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "tagger", "attribute_ruler"])
        _NLP.max_length = 3_000_000
    return _NLP


def find_zip(raw):
    """Last ZIP in the raw comment, skipping the NYSDOT and City Hall address on Washington Street."""
    zips = [m.group(1) or m.group(2) for m in ZIP.finditer(raw)
            if "Washington St" not in raw[max(0, m.start() - 50):m.start()]]
    return zips[-1] if zips else ""


class Lexicon:
    """Known names, built from every name string found in the corpus. Lives in memory only."""

    def __init__(self, names, place_text="", corpus_text=""):
        self.full, self.first, self.last = set(), set(), set()
        for name in names:
            parts = [p.lower() for p in name.split() if len(p.rstrip(".")) > 1]
            if len(parts) >= 2:
                self.full.add((parts[0], parts[-1]))
                self.first.add(parts[0])
                self.last.add(parts[-1])
        caps = Counter(m.group(0).lower() for m in re.finditer(r"\b[A-Z][a-z]+\b", place_text))
        named = {w for m in FULL.finditer(place_text) if (m.group(1).lower(), m.group(2).lower()) in self.full
                 for w in (m.group(1).lower(), m.group(2).lower())}  # a first name seen in a known full name is no place
        self.places = {w for w, n in caps.items() if n >= 3} - named
        self.lower = set(re.findall(r"\b[a-z][a-z'\-]+\b", corpus_text + "\n" + place_text))
        # a surname that is never written in lower case is not also an everyday word ("Young", "Hill" are)
        self.rare_last = {w for w in self.last - self.places - self.lower if len(w) >= 4}

    def is_first(self, word):
        word = word.lower()
        return word in self.first and word not in self.places


def before_place(text, end):
    """True when the next word is a place type: Harrison Street, Clinton Square, Destiny USA."""
    nxt = re.match(r"[ \t]*([A-Za-z]+)", text[end:end + 30])  # same line, no comma: "Jack Doe, Town Councilor"
    return bool(nxt) and nxt.group(1).lower() in PLACE_TYPES


def name_spans(text, own, lex, use_ner=True):
    spans = []
    soft = []  # weaker evidence: skipped when a place type follows
    for pattern, order in ((FULL, (1, 2)), (LAST_FIRST, (2, 1))):  # A
        for m in pattern.finditer(text):
            first, last = m.group(order[0]).lower(), m.group(order[1]).lower()
            span = (min(m.start(1), m.start(2)), max(m.end(1), m.end(2)))
            if (first, last) in lex.full:
                spans.append(span)
            elif order == (1, 2) and first in lex.first and first not in lex.lower and last not in PLACE_TYPES \
                    and (last not in lex.places or first not in lex.places) and "\n" not in text[span[0]:span[1]]:
                soft.append(span)  # a known first name before an unknown capitalized word
    tokens = {t for name in own for t in TOKEN.findall(name) if len(t) > 1 and t.lower() not in lex.places}
    if tokens:  # B
        for m in re.finditer(r"\b(?:" + "|".join(map(re.escape, sorted(tokens))) + r")\b", text, re.IGNORECASE):
            if m.group(0)[0].isupper():
                soft.append(m.span())
    spans += [m.span(1) for m in HONORIFIC.finditer(text) if m.group(1).split()[0].lower() not in lex.places]  # C
    spans += [m.span() for m in SPELLED.finditer(text)]
    soft += [m.span() for m in INITIAL.finditer(text) if m.group(1).lower() in lex.last]
    for m in INTRO.finditer(text):  # D
        if m.group(1).lower().startswith("my name") or lex.is_first(m.group(2).split()[0]):
            soft.append(m.span(2))
    for m in TOKEN.finditer(text):  # E: rare surnames
        if m.group(0)[0].isupper() and re.sub(r"['’]s$", "", m.group(0).lower()) in lex.rare_last:
            soft.append(m.span())
    for ent in nlp()(text).ents if use_ner else ():  # E: spaCy, backed by a known first name
        line = ent.text.split("\n")[0].rstrip()
        words = [w.strip(".,").lower() for w in line.split()]
        if ent.label_ != "PERSON" or not words or words[-1] in PLACE_TYPES:
            continue
        last = words[-1]
        unusual_last = len(words) > 1 and last.isalpha() and last not in lex.lower and last not in lex.places
        if any(w in lex.first for w in words[:-1]) or lex.is_first(words[0]) or words[-1] in lex.rare_last \
                or unusual_last:  # last word never seen in lower case and not a place
            soft.append((ent.start_char, ent.start_char + len(line)))
    return spans + [(s, e) for s, e in soft if not before_place(text, e)]


def scrub(text, own=(), lex=None, use_ner=True):
    """Returns (scrubbed text, Counter of replacements by type)."""
    lex = lex or Lexicon(own)
    protected = [m.span() for m in PROTECT.finditer(text)]
    spans = [(s, e, "EMAIL") for s, e in (m.span() for m in EMAIL.finditer(text))]
    spans += [(s, e, "PHONE") for s, e in (m.span() for m in PHONE.finditer(text))]
    spans += [(s, e, "ADDRESS") for s, e in (m.span() for m in ADDRESS.finditer(text))]
    spans += [(s, e, "NAME") for s, e in name_spans(text, own, lex, use_ner)
              if not any(s < pe and ps < e for ps, pe in protected)]
    out, counts, cursor = [], Counter(), 0
    for s, e, label in sorted(spans, key=lambda x: (x[0], -x[1])):  # earliest first, longest wins on ties
        if s < cursor:
            continue
        out += [text[cursor:s], f"[{label}]"]
        counts[label] += 1
        cursor = e
    out.append(text[cursor:])
    return "".join(out), counts


def commenter_lists(pages):
    """(file, name, comment type) rows from NYSDOT's lists of commenters (Appendix A and M-5, section 10).
    A row with one name cell (an organization, or a first name under the last name above it) gets name ''."""
    rows, files = [], ("decision/appendix-response-comments.pdf",
                       "deis_comments/Appendix M-5_Responses to Comments Received__April 2022.pdf")
    for file in files:
        lines = [l.strip() for p in pages if p["file"] == file and "First Name" in p["text"]  # list pages only
                 for l in p["text"].splitlines() if l.strip()]
        for i in range(2, len(lines) - 1):
            if re.fullmatch(r"\d{2}/\d{2}/\d{2}|Unknown", lines[i + 1]):
                two_cells = not re.search(r"\d", lines[i - 2])
                rows.append((file, f"{lines[i - 1]} {lines[i - 2].rstrip(',')}" if two_cells else "", lines[i]))
    return rows
