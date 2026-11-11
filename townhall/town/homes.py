"""Home neighborhood. PUMS has no tract, so each resident gets a census tract from its PUMA and the
City of Syracuse neighborhood that contains the tract's population center.

With CENSUS_API_KEY set, the tract is drawn from ACS 2024 5 year tract tables in proportion to how
many people in each tract share the resident's race and Hispanic origin (B03002) and tenure
(B25008 owner or renter, B26001 group quarters). Without a key, the draw is by total population
only (Census 2020 counts), independent of who the resident is.

Keyless public files saved under data/raw/census/:
  tracts in the PUMA: Census 2020 tract to 2020 PUMA relationship file
  tract population and population center: Census 2020 centers of population by tract
  neighborhoods: City of Syracuse open data, Syracuse Neighborhoods (boundaries updated 2026-06-04)
"""

import json

import numpy as np
import pandas as pd

from townhall import config
from townhall.town import census
from townhall.town.census import CENSUS_RAW, YEAR, census_api, fetch

REL_URL = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/2020_Census_Tract_to_2020_PUMA.txt"
CENPOP_URL = "https://www2.census.gov/geo/docs/reference/cenpop2020/tract/CenPop2020_Mean_TR{state}.txt"
NEIGHBORHOODS_URL = ("https://services6.arcgis.com/bdPqSfflsdgFRVVM/arcgis/rest/services/"
                     "Syracuse_Neighborhoods_20260604_view/FeatureServer/0/query"
                     "?where=1%3D1&outFields=NAME&outSR=4326&f=geojson")
OUTSIDE = "Outside the Syracuse neighborhood map"
UNASSIGNED = "unassigned"

NOT_HISPANIC = ", not Hispanic or Latino"
RACE_VARS = {"Hispanic or Latino": "B03002_012E"}
for code, var in [(1, "003"), (2, "004"), (3, "005"), (4, "005"), (5, "005"), (6, "006"), (7, "007"), (8, "008"), (9, "009")]:
    RACE_VARS[census.RACE[code] + NOT_HISPANIC] = f"B03002_{var}E"
TENURE_VARS = {census.TENURE[1]: "B25008_002E", census.TENURE[2]: "B25008_002E", census.TENURE[3]: "B25008_003E",
               census.TENURE[4]: "B25008_003E", census.NO_TENURE: "B26001_001E"}  # occupied without rent counts as renter
ACS_VARS = sorted({"B01003_001E", *RACE_VARS.values(), *TENURE_VARS.values()})

METHOD_POPULATION = ("tract drawn with probability proportional to tract population, independent of the resident's "
                     "age, race, income or tenure; neighborhood is the one containing the tract's population center")
METHOD_CONDITIONED = ("tract drawn with probability proportional to people in the tract with the resident's race and "
                      "Hispanic origin times the tract's share with the resident's tenure (treated as independent within "
                      "a tract; all ages); neighborhood is the one containing the tract's population center")


def _inside(x, y, ring):
    """Ray casting point in polygon test for one ring of [lon, lat] points."""
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def neighborhood_of(lon, lat, features):
    """Name of the GeoJSON feature containing the point, or OUTSIDE."""
    for feature in features:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for rings in polygons:
            if sum(_inside(lon, lat, ring) for ring in rings) % 2:  # odd count: inside the outer ring, not in a hole
                return feature["properties"]["NAME"]
    return OUTSIDE


def tract_name(tractce):
    """'000501' -> 'Census Tract 5.01', the way the Census names tracts."""
    suffix = tractce[4:]
    return f"Census Tract {int(tractce[:4])}" + (f".{suffix}" if suffix != "00" else "")


def load_tracts(state="36", puma="00701"):
    """Tracts in the PUMA with population, neighborhood and, with a key, ACS counts. Returns (DataFrame, source)."""
    rel = pd.read_csv(fetch(REL_URL, CENSUS_RAW / "2020_Census_Tract_to_2020_PUMA.txt"), dtype=str, encoding="utf-8-sig")
    rel = rel[(rel.STATEFP == state) & (rel.PUMA5CE == puma)]
    centers = pd.read_csv(fetch(CENPOP_URL.format(state=state), CENSUS_RAW / f"CenPop2020_Mean_TR{state}.txt"),
                          dtype=str, encoding="utf-8-sig")
    tracts = rel.merge(centers, on=["STATEFP", "COUNTYFP", "TRACTCE"], how="left")
    tracts["population"] = tracts.POPULATION.astype(int)
    source = "tract population: Census 2020 centers of population by tract"
    if config.CENSUS_API_KEY:
        counties = ",".join(sorted(tracts.COUNTYFP.unique()))
        acs = census_api(f"https://api.census.gov/data/{YEAR}/acs/acs5",
                         {"get": ",".join(ACS_VARS), "for": "tract:*", "in": f"state:{state} county:{counties}"})
        acs[ACS_VARS] = acs[ACS_VARS].astype(int).clip(lower=0)  # ACS marks missing estimates with large negatives
        tracts = tracts.merge(acs.rename(columns={"county": "COUNTYFP", "tract": "TRACTCE"}),
                              on=["COUNTYFP", "TRACTCE"], how="left", validate="one_to_one")
        tracts["population"] = tracts.B01003_001E
        source = f"tract counts: ACS {YEAR} 5 year B01003, B03002, B25008, B26001, Census API"
    features = json.loads(fetch(NEIGHBORHOODS_URL, CENSUS_RAW / "syracuse_neighborhoods.geojson").read_text())["features"]
    tracts["neighborhood"] = [neighborhood_of(float(lon), float(lat), features)
                              for lon, lat in zip(tracts.LONGITUDE, tracts.LATITUDE)]
    tracts["tract"] = tracts.STATEFP + tracts.COUNTYFP + tracts.TRACTCE
    tracts["tract_name"] = tracts.TRACTCE.map(tract_name)
    source += ("; tracts in the PUMA: Census 2020 tract to PUMA relationship file; population centers: Census 2020; "
               "neighborhoods: City of Syracuse open data")
    keep = ["tract", "tract_name", "population", "neighborhood"] + [v for v in ACS_VARS if v in tracts]
    return tracts[keep].reset_index(drop=True), source


def assign_homes(residents, tracts, seed=0):
    """Give each resident a tract and neighborhood. Conditioned on race, ethnicity and tenure when the tract
    counts are present, otherwise by population. Mutates and returns residents."""
    rng = np.random.default_rng(seed)
    population = tracts.population.to_numpy(dtype=float)
    conditioned = all(v in tracts for v in ACS_VARS)
    for resident in residents:
        weights, draw = population, "population"
        if conditioned:
            same_race = tracts[RACE_VARS[resident["race_ethnicity"]]].to_numpy(dtype=float)
            same_tenure = tracts[TENURE_VARS[resident["tenure"]]].to_numpy(dtype=float)
            weights = same_race * same_tenure / np.where(population > 0, population, 1)
            draw = "race, ethnicity and tenure"
            if weights.sum() == 0:  # no tract reports anyone like this resident: fall back, and say so
                weights, draw = population, "population (no tract had a matching count)"
        row = tracts.iloc[rng.choice(len(tracts), p=weights / weights.sum())]
        resident.update(tract=row.tract, tract_name=row.tract_name, neighborhood=row.neighborhood, home_draw=draw)
    return residents


def home_neighborhoods(residents, state="36", puma="00701", seed=0):
    """Load tracts and assign homes; if the tract files cannot be loaded, mark every resident unassigned."""
    try:
        tracts, source = load_tracts(state, puma)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        for resident in residents:
            resident.update(tract=UNASSIGNED, tract_name=UNASSIGNED, neighborhood=UNASSIGNED, home_draw=UNASSIGNED)
        return residents, {"source": f"UNASSIGNED: tract files could not be loaded ({type(error).__name__}: {error})",
                           "method": None, "tracts_in_puma": 0}
    assign_homes(residents, tracts, seed)
    method = METHOD_CONDITIONED if all(v in tracts for v in ACS_VARS) else METHOD_POPULATION
    mapped = set(tracts.neighborhood) - {OUTSIDE}
    names = {f["properties"]["NAME"] for f in json.loads((CENSUS_RAW / "syracuse_neighborhoods.geojson").read_text())["features"]}
    return residents, {"source": source, "method": method, "tracts_in_puma": len(tracts),
                       "tracts_outside_neighborhood_map": int((tracts.neighborhood == OUTSIDE).sum()),
                       "neighborhoods_with_a_tract_center": len(mapped),
                       "neighborhoods_without_a_tract_center": sorted(names - mapped),
                       "tract_population_total": int(tracts.population.sum())}
