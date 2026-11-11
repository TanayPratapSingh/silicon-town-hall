"""Census sampler: adult residents of one PUMA from ACS PUMS microdata, decoded to readable labels.

With CENSUS_API_KEY set it queries the ACS 2024 5 year PUMS API (2020 to 2024, 2020 PUMA
definitions). Without a key it reads the ACS 2024 1 year PUMS CSV files from www2.census.gov,
downloading them into data/raw/census/ the first time (about 52 MB for New York; the 5 year
CSV files are 189 MB). Every record carries the source it came from.
"""

import functools
import json
import urllib.parse
import urllib.request
import zipfile

import pandas as pd

from townhall import config

YEAR = 2024
API_SURVEY = "acs5"  # for PUMA 00701 on Nov 11, 2026: 7,537 person records in the 5 year file, 1,602 in the 1 year
CSV_SURVEY = "acs1"
CENSUS_RAW = config.RAW / "census"
PUMS_CSV_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_{kind}{abbr}.zip"
PUMS_API_URL = "https://api.census.gov/data/{year}/acs/{survey}/pums"
STATE_ABBR = {"36": "ny"}  # shortcut: only New York is mapped for the CSV path; add a state here to sample it

PERSON_VARS = ["SERIALNO", "SPORDER", "AGEP", "SEX", "RAC1P", "HISP", "PINCP", "ADJINC", "JWTRNS", "DIS", "PWGTP"]

# Labels from the 2024 PUMS data dictionary (data/raw/census/PUMS_Data_Dictionary_2024.csv), hyphens dropped.
SEX = {1: "Male", 2: "Female"}
RACE = {
    1: "White alone",
    2: "Black or African American alone",
    3: "American Indian alone",
    4: "Alaska Native alone",
    5: "American Indian and Alaska Native tribes specified, or not specified and no other races",
    6: "Asian alone",
    7: "Native Hawaiian and Other Pacific Islander alone",
    8: "Some Other Race alone",
    9: "Two or More Races",
}
HISPANIC = {
    1: "Not Spanish/Hispanic/Latino", 2: "Mexican", 3: "Puerto Rican", 4: "Cuban", 5: "Dominican",
    6: "Costa Rican", 7: "Guatemalan", 8: "Honduran", 9: "Nicaraguan", 10: "Panamanian", 11: "Salvadoran",
    12: "Other Central American", 13: "Argentinean", 14: "Bolivian", 15: "Chilean", 16: "Colombian",
    17: "Ecuadorian", 18: "Paraguayan", 19: "Peruvian", 20: "Uruguayan", 21: "Venezuelan",
    22: "Other South American", 23: "Spaniard", 24: "All Other Spanish/Hispanic/Latino",
}
COMMUTE = {
    1: "Car, truck, or van", 2: "Bus", 3: "Subway or elevated rail", 4: "Long distance train or commuter rail",
    5: "Light rail, streetcar, or trolley", 6: "Ferryboat", 7: "Taxi or ride hailing services", 8: "Motorcycle",
    9: "Bicycle", 10: "Walked", 11: "Worked from home", 12: "Other method",
}
TENURE = {
    1: "Owned with mortgage or loan",
    2: "Owned free and clear",
    3: "Rented",
    4: "Occupied without payment of rent",
}
DISABILITY = {1: "With a disability", 2: "Without a disability"}
NO_COMMUTE = "N/A (not a worker, or not at work last week)"
NO_TENURE = "N/A (lives in group quarters such as a dorm or nursing home)"


def fetch(url, path):
    """Download url to path once; later calls reuse the file."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as response:
            data = response.read()
        part = path.with_name(path.name + ".part")  # never leave a half written file under the real name
        part.write_bytes(data)
        part.replace(path)
    return path


def census_api(url, params):
    """One Census API table as a DataFrame of strings. Needs CENSUS_API_KEY."""
    query = urllib.parse.urlencode({**params, "key": config.CENSUS_API_KEY})
    with urllib.request.urlopen(f"{url}?{query}", timeout=120) as response:
        body, final_url = response.read(), response.geturl()
    if "missing_key" in final_url or not body.lstrip().startswith(b"["):
        raise RuntimeError(f"Census API returned no rows from {url}; check CENSUS_API_KEY")  # url has no key in it
    rows = json.loads(body)
    return pd.DataFrame(rows[1:], columns=rows[0])


@functools.lru_cache(maxsize=4)
def load_people(state="36", puma="00701"):
    """Every person record in the PUMA with tenure joined from the housing record. Returns (DataFrame, source)."""
    if config.CENSUS_API_KEY:
        url = PUMS_API_URL.format(year=YEAR, survey=API_SURVEY)
        # The PUMS API attaches housing variables such as TEN to each person row, so no join is needed.
        people = census_api(url, {"get": ",".join(PERSON_VARS + ["TEN"]),
                                  "for": f"public use microdata area:{puma}", "in": f"state:{state}"})
        return people, f"ACS {YEAR} 5 year PUMS (2020 to 2024), Census API ({url})"

    def read(kind, columns):
        name = f"csv_{kind}{STATE_ABBR[state]}.zip"
        path = fetch(PUMS_CSV_URL.format(year=YEAR, kind=kind, abbr=STATE_ABBR[state]),
                     CENSUS_RAW / f"{CSV_SURVEY}_{YEAR}" / name)
        with zipfile.ZipFile(path) as z:
            table = pd.read_csv(z.open(f"psam_{kind}{state}.csv"), usecols=columns + ["PUMA"], dtype=str)
        return table[table.PUMA == puma]

    people = read("p", PERSON_VARS)
    homes = read("h", ["SERIALNO", "TEN"])
    people = people.merge(homes[["SERIALNO", "TEN"]], on="SERIALNO", how="left")
    abbr = STATE_ABBR[state]
    return people, f"ACS {YEAR} 1 year PUMS, public CSV files csv_p{abbr}.zip and csv_h{abbr}.zip from www2.census.gov"


def _label(table, value, na=None):
    code = 0 if pd.isna(value) else int(value)  # the CSV leaves N/A blank; the API sends 0
    if code == 0 and na:
        return na
    return table[code]  # an unknown code raises on purpose: a silent fallback would hide a decoding bug


def _adjustment(adjinc):
    """ADJINC as a factor: the CSV writes 1015250 (6 implied decimals), the API writes 1.015250."""
    return float(adjinc) if "." in str(adjinc) else int(adjinc) / 1_000_000


def decode(row):
    """One PUMS person row (codes as strings) to a readable record."""
    race = _label(RACE, row["RAC1P"])
    hispanic = _label(HISPANIC, row["HISP"])
    return {
        "serialno": row["SERIALNO"],
        "sporder": int(row["SPORDER"]),
        "age": int(row["AGEP"]),
        "sex": _label(SEX, row["SEX"]),
        "race": race,
        "hispanic_origin": hispanic,
        "race_ethnicity": "Hispanic or Latino" if hispanic != HISPANIC[1] else f"{race}, not Hispanic or Latino",
        "income": round(int(row["PINCP"]) * _adjustment(row["ADJINC"])),  # 2024 dollars
        "commute": _label(COMMUTE, row["JWTRNS"], NO_COMMUTE),
        "tenure": _label(TENURE, row["TEN"], NO_TENURE),
        "disability": _label(DISABILITY, row["DIS"]),
        "person_weight": int(row["PWGTP"]),
    }


def adults(people):
    return people[people["AGEP"].astype(int) >= 18]


def sample_residents(state="36", puma="00701", n=20, seed=0, people=None, source="records passed in by the caller"):
    """n adults drawn without replacement with probability proportional to PWGTP, decoded to labels."""
    if people is None:
        people, source = load_people(state, puma)
    pool = adults(people).sort_values(["SERIALNO", "SPORDER"])  # the API returns rows in a new order each request
    picked = pool.sample(n=n, weights=pool["PWGTP"].astype(int), random_state=seed)
    return [{"resident_id": f"r{i:03d}", **decode(row), "state": state, "puma": puma, "census_source": source}
            for i, row in enumerate(picked.to_dict("records"))]
