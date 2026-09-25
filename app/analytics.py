"""Pure, deterministic portfolio maths. No DB, no LLM.

Every number the agent reports comes out of here, pre-formatted next to the raw value.
Property dicts use the DB column names (see db.py).
"""
from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation

from .normalize import (metro_for, normalize_city, normalize_occupancy, normalize_type, TYPE_TO_ASSET)

CR = 10_000_000
LAKH = 100_000
CONCENTRATION_THRESHOLD = 50.0

_UNITS = {"cr": CR, "crore": CR, "crores": CR, "l": LAKH, "lac": LAKH, "lacs": LAKH, "lakh": LAKH,
          "lakhs": LAKH, "k": 1_000, "thousand": 1_000}
_AMOUNT_RE = re.compile(r"^(\d[\d,]*(?:\.\d+)?|\.\d+)\s*([a-z]*)\.?$")

GROUP_DIMENSIONS = {
    "type_group": "type_group", "type": "type_group", "property_type": "type_group",
    "asset_class": "asset_class", "class": "asset_class",
    "city": "city", "metro": "metro", "region": "metro",
    "occupancy_status": "occupancy_status", "occupancy": "occupancy_status",
    "locality": "locality", "sub_type": "sub_type",
}


def parse_inr(value) -> int:
    """'₹12 Cr' | '1.2 crore' | '50 lakh' | '50L' | '5,00,000' | 42000000 -> rupees (int)."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"cannot parse amount: {value!r}")
    if isinstance(value, (int, float)):
        return int(round(value))
    s = str(value).strip().lower()
    s = re.sub(r"₹|\brs\.?|\binr\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    m = _AMOUNT_RE.match(s)
    if not m:
        raise ValueError(f"cannot parse amount: {value!r}")
    num, unit = m.groups()
    if unit and unit not in _UNITS:
        raise ValueError(f"unknown unit {unit!r} in amount {value!r}")
    try:
        amount = Decimal(num.replace(",", "")) * (_UNITS[unit] if unit else 1)
    except InvalidOperation:
        raise ValueError(f"cannot parse amount: {value!r}")
    return int(amount.to_integral_value())


def _indian_group(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def format_inr(amount) -> str:
    """12_00_00_000 -> '₹12.00 Cr'; 45_00_000 -> '₹45.0 L'; 12_345 -> '₹12,345'."""
    n = int(round(amount))
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a >= CR or round(a / LAKH, 1) >= 100:
        return f"{sign}₹{a / CR:,.2f} Cr"
    if a >= LAKH:
        return f"{sign}₹{a / LAKH:.1f} L"
    return f"{sign}₹{_indian_group(a)}"


def format_area(sqft) -> str:
    """16100 -> '16,100 sq ft' (so the model never has to group digits)."""
    return f"{_indian_group(int(round(sqft)))} sq ft"


def _pct(part: float, whole: float, nd: int = 1) -> float:
    return round(part / whole * 100, nd) if whole else 0.0


def _rec(p: dict) -> dict:
    """Internal record: value/rent scaled by ownership_percent; derived fields added."""
    own = (p.get("ownership_percent") if p.get("ownership_percent") is not None else 100) / 100
    value = int(round(p["current_estimated_value_inr"] * own))
    rent = int(round((p.get("annual_rent_inr") or 0) * own))
    area = p["area_sqft"]
    city = p.get("city")
    return {
        "property_id": p["property_id"],
        "type_group": p["type_group"],
        "asset_class": p.get("asset_class") or TYPE_TO_ASSET.get(p["type_group"]),
        "sub_type": p.get("sub_type"),
        "locality": p.get("locality"),
        "city": city,
        "metro": p.get("metro") or metro_for(city),
        "area_sqft": area,
        "value": value,
        "rent": rent,
        "occupancy_status": p.get("occupancy_status") or "Vacant",
        "ownership_percent": p.get("ownership_percent") if p.get("ownership_percent") is not None else 100,
        "hypothetical": bool(p.get("hypothetical")),
    }


def _income_producing(r: dict) -> bool:
    return r["occupancy_status"] == "Tenanted" and r["rent"] > 0


def _label(r: dict) -> str:
    where = r["locality"] if r["locality"] == r["city"] else ", ".join(x for x in (r["locality"], r["city"]) if x)
    return f"{r['type_group']} – {where}" + (" (hypothetical)" if r["hypothetical"] else "")


def _brief(r: dict) -> dict:
    """Compact form used in highlights (raw numbers live in the `properties` list)."""
    return {"property_id": r["property_id"], "label": _label(r), "value_fmt": format_inr(r["value"]),
            "rent_fmt": format_inr(r["rent"]), "gross_yield_pct": _pct(r["rent"], r["value"], 2)}


def _full(r: dict) -> dict:
    d = _brief(r)
    d.update({"value": r["value"], "annual_rent": r["rent"], "area_fmt": format_area(r["area_sqft"]),
        "type_group": r["type_group"], "asset_class": r["asset_class"], "sub_type": r["sub_type"],
        "locality": r["locality"], "city": r["city"], "metro": r["metro"],
        "area_sqft": r["area_sqft"], "occupancy_status": r["occupancy_status"],
        "income_producing": _income_producing(r),
        "value_per_sqft": round(r["value"] / r["area_sqft"]) if r["area_sqft"] else None,
    })
    d["value_per_sqft_fmt"] = format_inr(d["value_per_sqft"]) if d["value_per_sqft"] else None
    if r["ownership_percent"] != 100:
        d["ownership_percent"] = r["ownership_percent"]
    if r["hypothetical"]:
        d["hypothetical"] = True
    return d


def _loc_match(r: dict, text: str) -> bool:
    t = text.strip().lower()
    canon = (normalize_city(text) or "").lower()
    city, metro, loc = (r["city"] or "").lower(), (r["metro"] or "").lower(), (r["locality"] or "").lower()
    return t in (city, metro) or canon in (city, metro) or t in loc


def apply_filters(recs: list[dict], filters: dict | None) -> list[dict]:
    if not filters:
        return recs
    f = {k: v for k, v in filters.items() if v not in (None, "")}
    out = []
    for r in recs:
        if "property_ids" in f and r["property_id"] not in f["property_ids"]:
            continue
        if "type_group" in f and r["type_group"].lower() != str(f["type_group"]).lower():
            continue
        if "asset_class" in f and r["asset_class"].lower() != str(f["asset_class"]).lower():
            continue
        if "location" in f and not _loc_match(r, f["location"]):
            continue
        if "city" in f and (r["city"] or "").lower() != (normalize_city(f["city"]) or "").lower():
            continue
        if "metro" in f and (r["metro"] or "").lower() != str(f["metro"]).lower():
            continue
        if "locality" in f and str(f["locality"]).lower() not in (r["locality"] or "").lower():
            continue
        if "sub_type" in f and str(f["sub_type"]).lower() not in (r["sub_type"] or "").lower():
            continue
        if "occupancy_status" in f and r["occupancy_status"].lower() != str(f["occupancy_status"]).lower():
            continue
        if "min_value" in f and r["value"] < f["min_value"]:
            continue
        if "max_value" in f and r["value"] > f["max_value"]:
            continue
        if "min_rent" in f and r["rent"] < f["min_rent"]:
            continue
        if "max_rent" in f and r["rent"] > f["max_rent"]:
            continue
        if "min_area" in f and r["area_sqft"] < f["min_area"]:
            continue
        if "max_area" in f and r["area_sqft"] > f["max_area"]:
            continue
        out.append(r)
    return out


def _group_row(key, recs, total_value) -> dict:
    value = sum(r["value"] for r in recs)
    rent = sum(r["rent"] for r in recs)
    inc_value = sum(r["value"] for r in recs if _income_producing(r))
    return {
        "key": key, "count": len(recs),
        "value": value, "value_fmt": format_inr(value),
        "share_pct": _pct(value, total_value),
        "annual_rent": rent, "rent_fmt": format_inr(rent),
        "area_sqft": sum(r["area_sqft"] for r in recs),
        "yield_on_all_pct": _pct(rent, value, 2),
        "yield_on_income_producing_pct": _pct(rent, inc_value, 2) if inc_value else None,
    }


def _breakdown(recs, dim, total_value) -> list[dict]:
    groups = defaultdict(list)
    for r in recs:
        groups[r[dim] or "Unknown"].append(r)
    rows = [_group_row(k, v, total_value) for k, v in groups.items()]
    return sorted(rows, key=lambda x: -x["value"])


def _concentration(recs) -> list[dict]:
    """Flags on the WHOLE (scenario) portfolio: one type / city / metro > 50% of value."""
    total = sum(r["value"] for r in recs)
    if len(recs) < 2 or not total:
        return []
    flags = []
    for dim, name in (("type_group", "property type"), ("metro", "region"), ("city", "city")):
        for row in _breakdown(recs, dim, total):
            if row["share_pct"] > CONCENTRATION_THRESHOLD:
                flags.append({"dimension": dim, "key": row["key"], "share_pct": row["share_pct"],
                              "message": f"{row['share_pct']}% of portfolio value is in {name} '{row['key']}'"})
    # a metro flag and a city flag for the same place is noise: keep the metro one
    seen = {(f["dimension"], f["key"]) for f in flags}
    return [f for f in flags if not (f["dimension"] == "city" and ("metro", f["key"]) in seen)]


def _summary(recs: list[dict], full_total_value: int) -> dict:
    total_value = sum(r["value"] for r in recs)
    total_rent = sum(r["rent"] for r in recs)
    total_area = sum(r["area_sqft"] for r in recs)
    inc = [r for r in recs if _income_producing(r)]
    inc_value = sum(r["value"] for r in inc)
    vacant = [r for r in recs if r["occupancy_status"] == "Vacant"]
    selfocc = [r for r in recs if r["occupancy_status"] == "Self-occupied"]
    vac_value = sum(r["value"] for r in vacant)
    n = len(recs)
    out = {
        "totals": {
            "count": n, "value": total_value, "value_fmt": format_inr(total_value),
            "area_sqft": total_area, "area_fmt": format_area(total_area), "annual_rent": total_rent, "annual_rent_fmt": format_inr(total_rent),
            "value_per_sqft": round(total_value / total_area) if total_area else None,
        },
        "share_of_full_portfolio_pct": _pct(total_value, full_total_value),
        "yield": {
            "on_all_assets_pct": _pct(total_rent, total_value, 2),
            "on_income_producing_assets_pct": _pct(total_rent, inc_value, 2) if inc_value else None,
            "income_producing_count": len(inc),
        },
        "vacancy": {
            "vacant_count": len(vacant), "vacant_value": vac_value, "vacant_value_fmt": format_inr(vac_value),
            "vacant_pct_by_value": _pct(vac_value, total_value),
            "vacant_pct_by_count": _pct(len(vacant), n),
            "self_occupied_count": len(selfocc),
            "self_occupied_pct_by_value": _pct(sum(r["value"] for r in selfocc), total_value),
            "note": "Vacancy counts Vacant only; Self-occupied is not vacancy.",
        },
    }
    out["totals"]["value_per_sqft_fmt"] = format_inr(out["totals"]["value_per_sqft"]) if out["totals"]["value_per_sqft"] else None
    if recs:
        by_val = sorted(recs, key=lambda r: -r["value"])
        by_rent = sorted([r for r in recs if r["rent"] > 0], key=lambda r: -r["rent"])
        by_yield = sorted(inc, key=lambda r: -(r["rent"] / r["value"]) if r["value"] else 0)
        out["highlights"] = {
            "highest_value": _brief(by_val[0]),
            "lowest_value": _brief(by_val[-1]),
            "highest_rent": _brief(by_rent[0]) if by_rent else None,
            "best_yield": _brief(by_yield[0]) if by_yield else None,
            "lowest_yield_income_producing": _brief(by_yield[-1]) if by_yield else None,
            "vacant_properties": [_brief(r) for r in vacant],
        }
        out["top_by_value"] = [r["property_id"] for r in by_val[:3]]
        out["bottom_by_value"] = [r["property_id"] for r in by_val[::-1][:3]]
        out["top_rent"] = [r["property_id"] for r in by_rent[:3]]
    return out


def _split_prefs(text: str | None) -> list[str]:
    return [t.strip() for t in re.split(r"[/,]", text or "") if t.strip()]


def preference_mismatches(properties: list[dict], user: dict | None) -> list[dict]:
    """Deterministic: assets outside the user's stated type / location preferences."""
    if not user:
        return []
    recs = [_rec(p) for p in properties if p.get("status", "Active") == "Active"]
    types, classes = set(), set()
    for tok in _split_prefs(user.get("preferences")):
        t = tok.lower()
        if t == "commercial":
            classes.add("Commercial")
        elif (n := normalize_type(tok)):
            types.add(n["type_group"])
    locs = [t.lower() for t in _split_prefs(user.get("preferred_locations"))]
    out = []
    for r in recs:
        reasons = []
        if (types or classes) and r["type_group"] not in types and r["asset_class"] not in classes:
            reasons.append(f"{r['type_group']} is outside stated preference ({user.get('preferences')})")
        if locs:
            hay_loc = (r["locality"] or "").lower()
            ok = any(t in (r["city"] or "").lower() or t in (r["metro"] or "").lower()
                     or t in hay_loc or (hay_loc and hay_loc in t) or normalize_city(t) == r["city"]
                     for t in locs)
            if not ok:
                reasons.append(f"{r['locality']} is outside preferred locations ({user.get('preferred_locations')})")
        if reasons:
            out.append({"property_id": r["property_id"], "label": _label(r), "reasons": reasons})
    return out


def _insights(scenario_all: list[dict], flags: list[dict], mismatches: list[dict]) -> list[dict]:
    """Candidate proactive insights, in priority order. The agent mentions at most one."""
    out = []
    if flags:
        f = max(flags, key=lambda x: x["share_pct"])
        out.append({"type": "concentration", "text": f["message"]})
    vac = [r for r in scenario_all if r["occupancy_status"] == "Vacant"]
    if vac:
        v = sum(r["value"] for r in vac)
        total = sum(r["value"] for r in scenario_all)
        out.append({"type": "vacancy", "text": f"{len(vac)} vacant propert{'y' if len(vac) == 1 else 'ies'} "
                    f"({format_inr(v)}, {_pct(v, total)}% of value earning no rent): "
                    + "; ".join(f"{r['property_id']} {_label(r)}" for r in vac)})
    if mismatches:
        m = mismatches[0]
        out.append({"type": "preference_mismatch", "text": f"{m['property_id']} {m['label']}: {m['reasons'][0]}"})
    inc = sorted([r for r in scenario_all if _income_producing(r)], key=lambda r: r["rent"] / r["value"])
    if len(inc) >= 2:
        lo, hi = inc[0], inc[-1]
        if (hi["rent"] / hi["value"] - lo["rent"] / lo["value"]) * 100 >= 1.0:
            out.append({"type": "low_yield", "text": f"Lowest-yield rented asset: {lo['property_id']} {_label(lo)} "
                        f"at {_pct(lo['rent'], lo['value'], 2)}% vs best {_pct(hi['rent'], hi['value'], 2)}%"})
    return out


_OVERRIDE_FIELDS = {"current_estimated_value_inr": "value", "annual_rent_inr": "rent", "area_sqft": "area_sqft",
                    "occupancy_status": "occupancy_status"}


def _apply_scenario(recs, exclude_ids, overrides, add_hyp):
    by_id = {r["property_id"]: r for r in recs}
    desc = []
    scen = [dict(r) for r in recs]
    if exclude_ids:
        for pid in exclude_ids:
            if pid not in by_id:
                raise ValueError(f"unknown property id {pid!r}")
        ex = set(exclude_ids)
        desc.append("excluding " + ", ".join(f"{pid} ({_label(by_id[pid])})" for pid in exclude_ids))
        scen = [r for r in scen if r["property_id"] not in ex]
    if overrides:
        for pid, changes in overrides.items():
            if pid not in by_id:
                raise ValueError(f"unknown property id {pid!r}")
            target = next((r for r in scen if r["property_id"] == pid), None)
            if target is None:
                continue  # excluded and overridden: exclusion wins
            parts = []
            for k, v in changes.items():
                internal = _OVERRIDE_FIELDS.get(k)
                if not internal:
                    raise ValueError(f"cannot override field {k!r}")
                if internal in ("value", "rent"):
                    own = target["ownership_percent"] / 100
                    v = int(round(v * own))
                    parts.append(f"{internal} → {format_inr(v)}")
                else:
                    parts.append(f"{internal} → {v}")
                target[internal] = v
            desc.append(f"{pid} ({_label(by_id[pid])}) with " + ", ".join(parts))
    if add_hyp:
        h = dict(add_hyp)
        h.setdefault("property_id", "HYPO-1")
        h["hypothetical"] = True
        h.setdefault("annual_rent_inr", 0)
        h.setdefault("occupancy_status", "Tenanted" if h["annual_rent_inr"] else "Vacant")
        h.setdefault("ownership_percent", 100)
        hr = _rec(h)
        scen.append(hr)
        desc.append(f"adding a hypothetical {hr['type_group']} property in {hr['locality'] or hr['city']} "
                    f"worth {format_inr(hr['value'])}")
    return scen, ("Scenario: " + "; ".join(desc)) if desc else None


def _delta(actual: dict, scenario: dict, act_recs, scen_recs, actual_flags) -> dict:
    def metric(_name, a, s, fmt=None):
        d = {"actual": a, "scenario": s, "change": None if a is None or s is None else round(s - a, 2)}
        if a not in (None, 0) and s is not None:
            d["change_pct"] = round((s - a) / a * 100, 1)
        if fmt:
            d["actual_fmt"], d["scenario_fmt"] = fmt(a), fmt(s)
            d["change_fmt"] = fmt(s - a) if a is not None and s is not None else None
        return d

    def yield_pt(x):
        return None if x is None else f"{x:.2f}%"

    out = {
        "count": metric("count", actual["totals"]["count"], scenario["totals"]["count"]),
        "value": metric("value", actual["totals"]["value"], scenario["totals"]["value"], format_inr),
        "annual_rent": metric("rent", actual["totals"]["annual_rent"], scenario["totals"]["annual_rent"], format_inr),
        "area_sqft": metric("area", actual["totals"]["area_sqft"], scenario["totals"]["area_sqft"]),
        "yield_on_all_assets_pct": metric("y", actual["yield"]["on_all_assets_pct"], scenario["yield"]["on_all_assets_pct"], yield_pt),
        "yield_on_income_producing_pct": metric("y", actual["yield"]["on_income_producing_assets_pct"],
                                                scenario["yield"]["on_income_producing_assets_pct"], yield_pt),
        "vacant_pct_by_value": metric("v", actual["vacancy"]["vacant_pct_by_value"], scenario["vacancy"]["vacant_pct_by_value"]),
    }
    for k in ("yield_on_all_assets_pct", "yield_on_income_producing_pct"):
        if out[k]["change"] is not None:
            out[k]["change_fmt"] = f"{out[k]['change']:+.2f} pts"
    shifts = {}
    a_total = actual["totals"]["value"]
    s_total = scenario["totals"]["value"]
    for dim in ("type_group", "metro"):
        a = {r["key"]: r["share_pct"] for r in _breakdown(act_recs, dim, a_total)}
        s = {r["key"]: r["share_pct"] for r in _breakdown(scen_recs, dim, s_total)}
        shifts[dim] = [{"key": k, "actual_share_pct": a.get(k, 0.0), "scenario_share_pct": s.get(k, 0.0),
                        "change_pts": round(s.get(k, 0.0) - a.get(k, 0.0), 1)} for k in sorted(set(a) | set(s))]
    return {"metrics": out, "share_shifts": shifts, "actual_concentration_flags": actual_flags}


def appreciation(properties: list[dict]) -> dict:
    """No dates anywhere -> no annualised return. Absolute gain only where a purchase price exists."""
    priced = [p for p in properties if p.get("purchase_price_inr") and p.get("status", "Active") == "Active"]
    if not priced:
        return {"available": False,
                "reason": "purchase_price_inr and purchase dates are not recorded, so appreciation and "
                          "returns since purchase cannot be calculated. Provide the purchase price (and date) per property."}
    rows, tg, tp = [], 0, 0
    for p in priced:
        own = p["ownership_percent"] / 100 if p.get("ownership_percent") is not None else 1
        cur = int(round(p["current_estimated_value_inr"] * own))
        cost = int(round(p["purchase_price_inr"] * own))
        tg += cur - cost
        tp += cost
        rows.append({"property_id": p["property_id"], "purchase_price": cost, "purchase_price_fmt": format_inr(cost),
                     "current_value_fmt": format_inr(cur), "absolute_gain": cur - cost,
                     "absolute_gain_fmt": format_inr(cur - cost), "absolute_gain_pct": _pct(cur - cost, cost)})
    return {"available": True, "properties": rows, "properties_without_purchase_price": len(
        [p for p in properties if p.get("status", "Active") == "Active"]) - len(priced),
        "total_absolute_gain": tg, "total_absolute_gain_fmt": format_inr(tg),
        "annualised_return": {"available": False,
                              "reason": "No purchase dates are recorded, so an annualised return / CAGR is impossible."}}


def portfolio_analysis(properties: list[dict], *, filters: dict | None = None, exclude_ids=None,
                       overrides: dict | None = None, add_hypothetical: dict | None = None,
                       group_by: str | None = None, user: dict | None = None) -> dict:
    """The core function. Scenarios (exclude / overrides / add) are in-memory only."""
    actual_all = [_rec(p) for p in properties if p.get("status", "Active") == "Active"]
    is_hyp = bool(exclude_ids or overrides or add_hypothetical)
    scen_all, desc = (_apply_scenario(actual_all, exclude_ids, overrides, add_hypothetical)
                      if is_hyp else (actual_all, None))

    view = apply_filters(scen_all, filters)
    scen_total = sum(r["value"] for r in scen_all)
    out = {"is_hypothetical": is_hyp, "scenario_description": desc, "filters_applied": filters or {}}
    if not scen_all:
        out.update(empty=True, message="No active properties in this portfolio.")
        return out
    out.update(_summary(view, scen_total))
    view_total = out["totals"]["value"]
    # Lean by default to save LLM input tokens; other breakdowns come from group_by or compare.
    out["breakdowns"] = {"type_group": _breakdown(view, "type_group", view_total),
                         "city": _breakdown(view, "city", view_total)}
    if group_by:
        dim = GROUP_DIMENSIONS.get(str(group_by).lower())
        if not dim:
            raise ValueError(f"cannot group by {group_by!r}; use one of {sorted(set(GROUP_DIMENSIONS.values()))}")
        out["group_by"] = {"dimension": dim, "rows": _breakdown(view, dim, view_total)}
    out["properties"] = [_full(r) for r in sorted(view, key=lambda r: -r["value"])]
    out["concentration_flags"] = _concentration(scen_all)
    mism = preference_mismatches(_to_props(scen_all), user) if user else []
    out["insight_candidates"] = _insights(scen_all, out["concentration_flags"], mism)
    if is_hyp:
        actual_view = apply_filters(actual_all, filters)
        actual_sum = _summary(actual_view, sum(r["value"] for r in actual_all))
        actual_flags = _concentration(actual_all)
        if actual_view:
            out["actual"] = {k: actual_sum[k] for k in ("totals", "yield", "vacancy")}
            out["delta_vs_actual"] = _delta(actual_sum, out, actual_view, view, actual_flags)
    return out


def _to_props(recs: list[dict]) -> list[dict]:
    """Back to property-shaped dicts (value already ownership-scaled -> ownership 100)."""
    return [{"property_id": r["property_id"], "type_group": r["type_group"], "asset_class": r["asset_class"],
             "sub_type": r["sub_type"], "locality": r["locality"], "city": r["city"], "metro": r["metro"],
             "area_sqft": r["area_sqft"], "current_estimated_value_inr": r["value"],
             "annual_rent_inr": r["rent"], "occupancy_status": r["occupancy_status"],
             "ownership_percent": 100, "status": "Active"} for r in recs]


def _side_filter(dimension: str, value: str) -> tuple[str, str]:
    dim = GROUP_DIMENSIONS.get((dimension or "").lower())
    if dim not in ("type_group", "asset_class", "city", "metro", "occupancy_status"):
        raise ValueError(f"cannot compare on {dimension!r}")
    if dim == "type_group":
        n = normalize_type(value)
        if n:
            return dim, n["type_group"]
    elif dim == "asset_class":
        return dim, value.strip().title()
    elif dim == "city":
        return dim, normalize_city(value) or value
    elif dim == "occupancy_status":
        return dim, normalize_occupancy(value) or value.strip().title()
    return dim, value.strip()


def compare(properties: list[dict], dimension: str, a: str, b: str) -> dict:
    """e.g. compare(props, 'asset_class', 'Residential', 'Commercial')."""
    recs = [_rec(p) for p in properties if p.get("status", "Active") == "Active"]
    total = sum(r["value"] for r in recs)
    sides = {}
    for tag, val in (("a", a), ("b", b)):
        dim, canon = _side_filter(dimension, val)
        sub = [r for r in recs if (r[dim] or "").lower() == canon.lower()]
        row = _group_row(canon, sub, total)
        row["value_per_sqft"] = round(row["value"] / row["area_sqft"]) if row["area_sqft"] else None
        row["value_per_sqft_fmt"] = format_inr(row["value_per_sqft"]) if row["value_per_sqft"] else None
        row["vacant_count"] = len([r for r in sub if r["occupancy_status"] == "Vacant"])
        sides[tag] = row
    A, B = sides["a"], sides["b"]

    def higher(field):
        if A[field] is None or B[field] is None:
            return "n/a"
        if A[field] == B[field]:
            return "equal"
        return A["key"] if A[field] > B[field] else B["key"]

    return {"dimension": GROUP_DIMENSIONS[dimension.lower()], "a": A, "b": B,
            "higher_value": higher("value"), "higher_rent": higher("annual_rent"),
            "higher_yield_on_all_assets": higher("yield_on_all_pct"),
            "value_difference_fmt": format_inr(abs(A["value"] - B["value"])),
            "share_gap_pts": round(abs(A["share_pct"] - B["share_pct"]), 1)}
