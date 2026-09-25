"""Type / city / occupancy normalisation.

Raw labels are always kept in the DB (`property_type_raw`); these functions only derive
the grouped columns used for analytics.
"""
import re

TYPE_TO_ASSET = {"Retail": "Commercial", "Office": "Commercial", "Residential": "Residential"}

# raw label (lower-case) -> type_group. "Commercial Office" and "Office" merge.
_EXACT_TYPES = {
    "retail": "Retail",
    "commercial office": "Office",
    "office": "Office",
    "residential": "Residential",
}
# keyword fallback for free text typed by users ("a shop", "3bhk flat", "IT park")
_TYPE_KEYWORDS = [
    ("Retail", ["retail", "shop", "showroom", "high street", "high-street", "mall", "store", "storefront"]),
    ("Office", ["office", "it park", "coworking", "co-working", "business park"]),
    ("Residential", ["residential", "flat", "apartment", "villa", "house", "bungalow", "penthouse",
                     "condo", "bhk", "home"]),
]

# Tail of "Locality, X" that is a state, not a city -> the locality becomes the city.
STATES = {"maharashtra", "karnataka", "haryana", "uttar pradesh", "tamil nadu", "telangana", "gujarat",
          "rajasthan", "west bengal", "kerala", "goa", "punjab", "madhya pradesh"}

CITY_ALIASES = {
    "gurgaon": "Gurugram", "gurugram": "Gurugram",
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "blr": "Bengaluru",
    "bombay": "Mumbai", "mumbai": "Mumbai",
    "new delhi": "Delhi", "delhi": "Delhi",
    "delhi ncr": "Delhi NCR", "delhi-ncr": "Delhi NCR", "ncr": "Delhi NCR",
    "noida": "Noida", "greater noida": "Greater Noida",
}

# city -> metro (used for geographic concentration). Unknown cities are their own metro.
CITY_TO_METRO = {
    "Mumbai": "Mumbai", "Navi Mumbai": "Mumbai", "Thane": "Mumbai",
    "Gurugram": "Delhi NCR", "Noida": "Delhi NCR", "Greater Noida": "Delhi NCR",
    "Ghaziabad": "Delhi NCR", "Faridabad": "Delhi NCR", "Delhi": "Delhi NCR", "Delhi NCR": "Delhi NCR",
    "Bengaluru": "Bengaluru",
}

_OCCUPANCY_KEYWORDS = [
    ("Self-occupied", ["self-occupied", "self occupied", "selfoccupied", "owner occupied", "own use", "i live", "live in"]),
    ("Vacant", ["vacant", "empty", "unoccupied", "unrented", "not rented", "no tenant"]),
    ("Tenanted", ["tenanted", "rented", "leased", "let out", "on rent", "has a tenant"]),
]


def _has_word(text: str, kw: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", text) is not None


def normalize_type(text: str | None) -> dict | None:
    """-> {"type_group", "asset_class"} or None if it can't be classified."""
    if not text:
        return None
    t = text.strip().lower()
    group = _EXACT_TYPES.get(t)
    if not group:
        for g, kws in _TYPE_KEYWORDS:
            if any(_has_word(t, k) for k in kws):
                group = g
                break
    if not group:
        return None
    return {"type_group": group, "asset_class": TYPE_TO_ASSET[group]}


def normalize_city(text: str | None) -> str | None:
    if not text or not text.strip():
        return None
    t = text.strip()
    return CITY_ALIASES.get(t.lower()) or (t.title() if t.islower() or t.isupper() else t)


def metro_for(city: str | None) -> str | None:
    if not city:
        return None
    return CITY_TO_METRO.get(city, city)


def normalize_occupancy(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip().lower()
    for canon, kws in _OCCUPANCY_KEYWORDS:
        if any(kw in t for kw in kws):
            return canon
    return None


def tidy_locality(text: str | None) -> str | None:
    if not text or not text.strip():
        return None
    t = " ".join(text.split())
    return t.title() if t.islower() else t


def split_location(location: str) -> tuple[str | None, str | None]:
    """'Bandra West, Mumbai' -> ('Bandra West', 'Mumbai'); 'Alibaug, Maharashtra' -> ('Alibaug', 'Alibaug')."""
    parts = [p.strip() for p in (location or "").split(",") if p.strip()]
    if not parts:
        return None, None
    if len(parts) == 1:
        city = normalize_city(parts[0])
        return (None, city) if parts[0].lower() in CITY_ALIASES else (tidy_locality(parts[0]), None)
    tail = parts[-1]
    locality = ", ".join(parts[:-1])
    if tail.lower() in STATES:
        return tidy_locality(locality), normalize_city(locality)
    return tidy_locality(locality), normalize_city(tail)
