"""Paths and settings shared by every part of the app."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"          # gitignored: raw PDFs, extracted text, indexes, databases
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RUNS = ROOT / "runs"          # committed: aggregate outputs that every number in docs/ traces to

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
RESIDENT_MODEL = os.getenv("RESIDENT_MODEL", "gpt-4.1-mini")
ANALYST_MODEL = os.getenv("ANALYST_MODEL", "gpt-4.1")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4.1-mini")
