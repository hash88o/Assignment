"""Fuzzy reference resolution: 'the Bandra one', 'my Delhi office', 'P007' -> property rows."""
import re

from .normalize import CITY_ALIASES, normalize_occupancy, normalize_type

_STOP = {"my", "the", "a", "an", "one", "ones", "property", "properties", "in", "at", "of", "on", "near", "that",
         "this", "it", "to", "me", "i", "own", "unit", "place", "asset", "assets", "and", "or", "for", "one's"}

# Small static locality -> city gazetteer (from the seed data) used when a user omits the city.
LOCALITY_CITY = {
    "bandra": "Mumbai", "andheri": "Mumbai", "worli": "Mumbai", "lower parel": "Mumbai", "powai": "Mumbai",
    "juhu": "Mumbai", "colaba": "Mumbai", "bkc": "Mumbai", "malabar hill": "Mumbai",
    "whitefield": "Bengaluru", "koramangala": "Bengaluru", "indiranagar": "Bengaluru", "hsr layout": "Bengaluru",
    "jayanagar": "Bengaluru", "electronic city": "Bengaluru",
    "golf course road": "Gurugram", "dlf phase": "Gurugram", "cyber city": "Gurugram",
    "connaught place": "Delhi", "saket": "Delhi",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _haystack(p: dict) -> list[str]:
    parts = [p.get("locality"), p.get("city"), p.get("metro"), p.get("type_group"), p.get("property_type_raw"),
             p.get("sub_type"), p.get("asset_class"), p.get("occupancy_status"), p["property_id"]]
    return _words(" ".join(x for x in parts if x))


def _token_matches(tok: str, p: dict, hay: list[str]) -> bool:
    if tok == p["property_id"].lower():
        return True
    if any(w == tok or (len(tok) >= 3 and w.startswith(tok)) for w in hay):
        return True
    canon = CITY_ALIASES.get(tok)  # gurgaon -> Gurugram, bangalore -> Bengaluru
    if canon and canon.lower() in " ".join(hay):
        return True
    t = normalize_type(tok)
    if t and (t["type_group"] == p["type_group"]):
        return True
    if tok == "commercial" and p["asset_class"] == "Commercial":
        return True
    occ = normalize_occupancy(tok)
    return bool(occ and occ == p["occupancy_status"])


def find_matches(props: list[dict], query: str) -> dict:
    """-> {"status": found|ambiguous|none, "matches": [...], "closest": [...]}; props are the user's own rows."""
    q = (query or "").strip()
    ids = re.findall(r"\bp\d{3,}\b", q.lower())
    if ids:
        hit = [p for p in props if p["property_id"].lower() in ids]
        return {"status": "found" if len(hit) == 1 else ("ambiguous" if hit else "none"), "matches": hit, "closest": []}
    toks = [t for t in _words(q) if t not in _STOP]
    if not toks:
        return {"status": "ambiguous" if len(props) > 1 else ("found" if props else "none"),
                "matches": props, "closest": []}
    scored = []
    for p in props:
        hay = _haystack(p)
        n = sum(1 for t in toks if _token_matches(t, p, hay))
        scored.append((n, p))
    full = [p for n, p in scored if n == len(toks)]
    if full:
        return {"status": "found" if len(full) == 1 else "ambiguous", "matches": full, "closest": []}
    partial = [p for n, p in sorted(scored, key=lambda x: -x[0]) if n > 0]
    return {"status": "none", "matches": [], "closest": partial[:3]}
