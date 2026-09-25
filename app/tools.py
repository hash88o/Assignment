"""Agent tools, built per request so user_id / conversation_id are closed over, never LLM arguments.

Tools return compact JSON strings. Domain problems (unknown property, ambiguous reference, bad amount)
come back as {"status": "error"|"ambiguous"|"missing_fields"|...} so the model can recover in-conversation;
only unexpected exceptions propagate (and are counted as tool errors in the trace).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from . import config, db
from .analytics import (appreciation, compare as compare_fn, format_inr, parse_inr, portfolio_analysis)
from .normalize import (metro_for, normalize_city, normalize_occupancy, normalize_type, tidy_locality)
from .resolve import LOCALITY_CITY, find_matches

OCCUPANCIES = ("Tenanted", "Vacant", "Self-occupied")
_PENDING_REMOVALS: dict[tuple[str, str], float] = {}  # (conversation_id, property_id) -> time of the pending confirmation
_PENDING_TTL_S = 30 * 60
_lock = threading.Lock()


@dataclass
class TurnContext:
    user_id: str
    conversation_id: str
    handoff_reasons: list[str] = field(default_factory=list)
    large_changes: list[str] = field(default_factory=list)


def _d(x):
    """Nested pydantic args arrive as model instances; the tool bodies work with plain dicts."""
    if isinstance(x, BaseModel):
        return x.model_dump()
    if isinstance(x, list):
        return [_d(i) for i in x]
    return x


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def _err(msg: str, **extra) -> str:
    return _j({"status": "error", "error": msg, **extra})


def _label(p: dict) -> str:
    where = p["locality"] if p["locality"] == p["city"] else ", ".join(x for x in (p["locality"], p["city"]) if x)
    return f"{p['type_group']} – {where}"


def _card(p: dict) -> dict:
    return {"property_id": p["property_id"], "label": _label(p), "sub_type": p["sub_type"],
            "area_sqft": p["area_sqft"], "value_fmt": format_inr(p["current_estimated_value_inr"]),
            "rent_fmt": format_inr(p["annual_rent_inr"]), "occupancy_status": p["occupancy_status"]}


class Filters(BaseModel):
    property_type: str | None = Field(None, description="Retail, Office or Residential (or a synonym like shop/flat/villa)")
    asset_class: str | None = Field(None, description="Commercial or Residential")
    location: str | None = Field(None, description="City, region (e.g. Delhi NCR) or locality, e.g. Mumbai, Gurgaon, Bandra")
    occupancy_status: str | None = Field(None, description="Tenanted, Vacant or Self-occupied")
    sub_type: str | None = Field(None, description="e.g. Apartment, Villa, Mall retail")
    min_value: str | None = Field(None, description="Inclusive lower bound on value, amount exactly as the user wrote it, e.g. '10 crore'")
    max_value: str | None = Field(None, description="Inclusive upper bound on value, as the user wrote it")
    min_rent: str | None = Field(None, description="Inclusive lower bound on ANNUAL rent, as the user wrote it")
    max_rent: str | None = Field(None, description="Inclusive upper bound on ANNUAL rent, as the user wrote it")
    min_area_sqft: float | None = None
    max_area_sqft: float | None = None


class ValueOverride(BaseModel):
    property: str = Field(description="Property ID (P001) or short reference such as 'Bandra' or 'Delhi office'")
    new_value: str | None = Field(None, description="Hypothetical value, as the user wrote it, e.g. '14 Cr'")
    new_annual_rent: str | None = Field(None, description="Hypothetical ANNUAL rent, as the user wrote it")


class HypotheticalProperty(BaseModel):
    property_type: str = Field(description="Retail, Office or Residential")
    locality: str | None = None
    city: str | None = None
    area_sqft: float
    value: str = Field(description="As the user wrote it, e.g. '3 Cr'")
    annual_rent: str | None = Field(None, description="ANNUAL rent as the user wrote it")
    occupancy_status: str | None = None


class CompareSpec(BaseModel):
    dimension: Literal["asset_class", "type_group", "city", "metro", "occupancy_status"]
    a: str = Field(description="e.g. Residential")
    b: str = Field(description="e.g. Commercial")


class AnalysisArgs(BaseModel):
    filters: Filters | None = Field(None, description="Restrict the analysed set (retail only, Mumbai only, above ₹10 Cr...)")
    group_by: Literal["type_group", "asset_class", "city", "metro", "occupancy_status", "locality", "sub_type"] | None = None
    exclude_properties: list[str] | None = Field(
        None, description="HYPOTHETICAL: property IDs or short references to leave out, e.g. ['Bandra']")
    value_overrides: list[ValueOverride] | None = Field(None, description="HYPOTHETICAL: 'what if X were worth Y'")
    add_hypothetical_property: HypotheticalProperty | None = Field(None, description="HYPOTHETICAL: 'what if I bought...'")
    compare: CompareSpec | None = Field(None, description="Side-by-side comparison, e.g. Residential vs Commercial")


class FindArgs(BaseModel):
    query: str = Field(description="What the user called it, e.g. 'the Bandra one', 'my Delhi office', 'P007'")


class AddArgs(BaseModel):
    property_type: str | None = Field(None, description="Retail, Office or Residential (synonyms ok)")
    locality: str | None = Field(None, description="Locality/area, e.g. Indiranagar")
    city: str | None = Field(None, description="City if the user said it; otherwise omit (the tool infers and tells you)")
    area_sqft: float | None = None
    value: str | None = Field(None, description="Current value exactly as the user wrote it, e.g. '4.2 crore'")
    annual_rent: str | None = Field(None, description="ANNUAL rent as the user wrote it")
    occupancy_status: str | None = Field(None, description="Tenanted, Vacant or Self-occupied")
    sub_type: str | None = None
    optional_details_asked: bool = Field(
        False, description="Set true only after you already asked the user about rent/occupancy once")
    allow_duplicate: bool = False


class UpdateArgs(BaseModel):
    property_ref: str = Field(description="Property ID or short reference, e.g. 'Bandra retail'")
    field: str = Field(description="value | annual_rent | monthly_rent | area_sqft | occupancy_status | sub_type | "
                                   "property_type | locality | city | ownership_percent | purchase_price")
    new_value: str = Field(description="New value exactly as the user wrote it, e.g. '12.5 crore'")


class RemoveArgs(BaseModel):
    property_ref: str
    confirmed: bool = Field(False, description="True only if the user explicitly confirmed the removal in their last message")


class FlagArgs(BaseModel):
    reason: str = Field(description="One short sentence for the human team")


def _infer_city(user: dict, own: list[dict], locality: str | None, given: str | None):
    if given:
        return normalize_city(given), "provided by user"
    if locality:
        low = locality.lower()
        for p in own:
            if (p["locality"] or "").lower() == low:
                return p["city"], f"your existing property in {p['locality']}"
        for k, city in LOCALITY_CITY.items():
            if low.startswith(k) or k in low:
                return city, f"{locality} is in {city}"
    if user.get("city"):
        return normalize_city(user["city"]), f"your profile city ({user['city']})"
    return None, None


def _resolve_one(own: list[dict], ref: str):
    """-> (property, None) or (None, error_payload_json)."""
    res = find_matches(own, ref)
    if res["status"] == "found":
        return res["matches"][0], None
    if res["status"] == "ambiguous":
        return None, _j({"status": "ambiguous", "reference": ref, "message": "Several properties match; ask the user which one.",
                         "candidates": [_card(p) for p in res["matches"]]})
    return None, _j({"status": "error", "error": f"no property matches '{ref}'",
                     "closest": [_card(p) for p in res["closest"]] or None,
                     "your_properties": [_card(p) for p in own]})


_UPDATE_FIELDS = {
    "value": "current_estimated_value_inr", "current_value": "current_estimated_value_inr",
    "estimated_value": "current_estimated_value_inr", "current_estimated_value_inr": "current_estimated_value_inr",
    "annual_rent": "annual_rent_inr", "rent": "annual_rent_inr", "annual_rent_inr": "annual_rent_inr",
    "monthly_rent": "monthly_rent", "area": "area_sqft", "area_sqft": "area_sqft",
    "occupancy": "occupancy_status", "occupancy_status": "occupancy_status", "sub_type": "sub_type",
    "property_type": "property_type", "type": "property_type", "locality": "locality", "city": "city",
    "ownership_percent": "ownership_percent", "purchase_price": "purchase_price_inr",
    "purchase_price_inr": "purchase_price_inr",
}
_MONEY = {"current_estimated_value_inr", "annual_rent_inr", "purchase_price_inr"}
_FIELD_LABELS = {"current_estimated_value_inr": "value", "annual_rent_inr": "annual rent", "purchase_price_inr": "purchase price",
                 "area_sqft": "area (sq ft)", "occupancy_status": "occupancy", "ownership_percent": "ownership %"}


def build_tools(ctx: TurnContext) -> list[StructuredTool]:
    uid, conv = ctx.user_id, ctx.conversation_id

    def _portfolio_now() -> dict:
        """Post-write totals computed here, so the model never has to add or subtract anything itself."""
        r = portfolio_analysis(db.list_properties(uid))
        if r.get("empty"):
            return {"count": 0, "total_value_fmt": format_inr(0), "annual_rent_fmt": format_inr(0)}
        return {"count": r["totals"]["count"], "total_value_fmt": r["totals"]["value_fmt"],
                "annual_rent_fmt": r["totals"]["annual_rent_fmt"], "yield_on_all_assets_pct": r["yield"]["on_all_assets_pct"],
                "yield_on_income_producing_assets_pct": r["yield"]["on_income_producing_assets_pct"]}

    def get_portfolio_analysis(filters=None, group_by=None, exclude_properties=None, value_overrides=None,
                               add_hypothetical_property=None, compare=None) -> str:
        own = db.list_properties(uid)
        user = db.get_user(uid)
        filters, value_overrides, add_hypothetical_property, compare = (
            _d(filters), _d(value_overrides), _d(add_hypothetical_property), _d(compare))
        try:
            f = _build_filters(filters)
            exclude_ids = []
            for ref in exclude_properties or []:
                p, bad = _resolve_one(own, ref)
                if bad:
                    return bad
                exclude_ids.append(p["property_id"])
            overrides = {}
            for o in value_overrides or []:
                p, bad = _resolve_one(own, o["property"])
                if bad:
                    return bad
                ch = {}
                if o.get("new_value"):
                    ch["current_estimated_value_inr"] = parse_inr(o["new_value"])
                if o.get("new_annual_rent"):
                    ch["annual_rent_inr"] = parse_inr(o["new_annual_rent"])
                if not ch:
                    return _err("value override needs new_value or new_annual_rent")
                overrides[p["property_id"]] = ch
            hyp = None
            if add_hypothetical_property:
                hyp, bad = _build_hypothetical(user, own, add_hypothetical_property)
                if bad:
                    return bad
            res = portfolio_analysis(own, filters=f, exclude_ids=exclude_ids or None, overrides=overrides or None,
                                     add_hypothetical=hyp, group_by=group_by, user=user)
            out = {"status": "ok", "user_context": _user_context(user), **res, "appreciation": appreciation(own)}
            if compare:
                out["comparison"] = compare_fn(own, compare["dimension"], compare["a"], compare["b"])
            return _j(out)
        except ValueError as e:
            return _err(str(e))

    def _build_filters(flt) -> dict | None:
        if not flt:
            return None
        f = {}
        if flt.get("property_type"):
            t = normalize_type(flt["property_type"])
            if t:
                f["type_group"] = t["type_group"]
            elif flt["property_type"].strip().lower() == "commercial":
                f["asset_class"] = "Commercial"
            else:
                raise ValueError(f"unknown property type {flt['property_type']!r}; use Retail, Office or Residential")
        if flt.get("asset_class"):
            if flt["asset_class"].strip().title() not in ("Commercial", "Residential"):
                raise ValueError("asset_class must be Commercial or Residential")
            f["asset_class"] = flt["asset_class"].strip().title()
        for k in ("location", "sub_type"):
            if flt.get(k):
                f[k] = flt[k]
        if flt.get("occupancy_status"):
            occ = normalize_occupancy(flt["occupancy_status"])
            if not occ:
                raise ValueError("occupancy_status must be Tenanted, Vacant or Self-occupied")
            f["occupancy_status"] = occ
        for k in ("min_value", "max_value", "min_rent", "max_rent"):
            if flt.get(k):
                f[k] = parse_inr(flt[k])
        if flt.get("min_area_sqft") is not None:
            f["min_area"] = flt["min_area_sqft"]
        if flt.get("max_area_sqft") is not None:
            f["max_area"] = flt["max_area_sqft"]
        return f or None

    def _build_hypothetical(user, own, h):
        missing = [k for k in ("property_type", "area_sqft", "value") if not h.get(k)]
        if not (h.get("locality") or h.get("city")):
            missing.append("location")
        t = normalize_type(h.get("property_type") or "")
        if h.get("property_type") and not t:
            missing.append("property_type (Retail, Office or Residential)")
        if missing:
            return None, _j({"status": "missing_fields", "missing": missing})
        city, _src = _infer_city(user, own, h.get("locality"), h.get("city"))
        rent = parse_inr(h["annual_rent"]) if h.get("annual_rent") else 0
        occ = normalize_occupancy(h.get("occupancy_status") or "") or ("Tenanted" if rent else "Vacant")
        return {"type_group": t["type_group"], "asset_class": t["asset_class"],
                "locality": tidy_locality(h.get("locality")) or city, "city": city, "metro": metro_for(city),
                "area_sqft": h["area_sqft"], "current_estimated_value_inr": parse_inr(h["value"]),
                "annual_rent_inr": rent, "occupancy_status": occ}, None

    def _user_context(u: dict) -> dict:
        lo, hi = u.get("portfolio_value_min"), u.get("portfolio_value_max")
        pref = None if lo is None else (format_inr(lo) if lo == hi else f"{format_inr(lo)} – {format_inr(hi)}")
        return {"name": u["name"], "city": u["city"], "preferences": u["preferences"],
                "preferred_locations": u["preferred_locations"], "stated_value_preference": pref}

    def find_properties(query: str) -> str:
        res = find_matches(db.list_properties(uid), query)
        return _j({"status": res["status"], "matches": [_card(p) for p in res["matches"]],
                   "closest": [_card(p) for p in res["closest"]] or None})

    def add_property(property_type=None, locality=None, city=None, area_sqft=None, value=None, annual_rent=None,
                     occupancy_status=None, sub_type=None, optional_details_asked=False, allow_duplicate=False) -> str:
        user, own = db.get_user(uid), db.list_properties(uid)
        missing = []
        t = normalize_type(property_type) if property_type else None
        if not t:
            missing.append("property_type (Retail, Office or Residential)")
        if not (locality or city):
            missing.append("location (locality and city)")
        if not area_sqft or area_sqft <= 0:
            missing.append("area_sqft")
        if not value:
            missing.append("value")
        if missing:
            return _j({"status": "missing_fields", "missing": missing})
        try:
            val = parse_inr(value)
            rent = parse_inr(annual_rent) if annual_rent else None
        except ValueError as e:
            return _err(str(e))
        if val <= 0:
            return _err("value must be greater than zero")
        loc = tidy_locality(locality)
        c, src = _infer_city(user, own, loc, city)
        if not c:
            return _j({"status": "missing_fields", "missing": ["city"]})
        occ = normalize_occupancy(occupancy_status) if occupancy_status else None
        if occupancy_status and not occ:
            return _err("occupancy_status must be Tenanted, Vacant or Self-occupied")
        assumptions = []
        if not city:
            assumptions.append(f"City assumed to be {c} ({src}).")
        if occ is None and rent:
            occ = "Tenanted"
        if occ is None or (occ == "Tenanted" and rent is None):
            if not optional_details_asked:
                return _j({"status": "ask_optional", "message": "Nothing is saved yet. Ask the user ONCE for annual rent and whether it is "
                           "tenanted / vacant / self-occupied (they may say they don't know), then call add_property again "
                           "with optional_details_asked=true and the same fields.", "so_far": {
                               "type_group": t["type_group"], "locality": loc, "city": c, "area_sqft": area_sqft,
                               "value_fmt": format_inr(val)}})
            occ = occ or "Vacant"
            rent = rent or 0
            assumptions.append(f"Recorded as {occ} with annual rent ₹0 because it was not provided; can be updated any time.")
        if occ in ("Vacant", "Self-occupied") and rent:
            assumptions.append(f"Rent ignored: a {occ} property earns ₹0.")
            rent = 0
        if not allow_duplicate:
            for p in own:
                if (p["locality"] or "").lower() == (loc or "").lower() and p["type_group"] == t["type_group"] \
                        and p["area_sqft"] == area_sqft and p["current_estimated_value_inr"] == val:
                    return _j({"status": "possible_duplicate", "existing": _card(p),
                               "message": "An identical property already exists. Nothing added; ask if they really want a second one."})
        raw = property_type.strip().title()
        sub = sub_type or (raw if raw.lower() not in ("retail", "office", "residential", "commercial office") else None)
        row = db.insert_property(uid, {
            "property_type_raw": raw, "type_group": t["type_group"], "asset_class": t["asset_class"], "sub_type": sub,
            "locality": loc or c, "city": c, "metro": metro_for(c), "area_sqft": area_sqft,
            "current_estimated_value_inr": val, "annual_rent_inr": rent or 0, "occupancy_status": occ})
        return _j({"status": "added", "property": _card(row), "assumptions": assumptions, "portfolio_now": _portfolio_now()})

    def update_property(property_ref: str, field: str, new_value: str) -> str:
        own = db.list_properties(uid)
        p, bad = _resolve_one(own, property_ref)
        if bad:
            return bad
        col = _UPDATE_FIELDS.get(field.strip().lower().replace(" ", "_"))
        if not col:
            return _err(f"cannot update field {field!r}", updatable=sorted(set(_UPDATE_FIELDS) - {"current_value", "estimated_value", "rent", "area", "occupancy", "type"}))
        updates, effects = {}, []
        try:
            if col in _MONEY or col == "monthly_rent":
                n = parse_inr(new_value)
                if col == "monthly_rent":
                    col, n = "annual_rent_inr", n * 12
                if n < 0 or (n == 0 and col == "current_estimated_value_inr"):
                    return _err("amount must be positive")
                updates[col] = n
            elif col == "area_sqft":
                n = float(str(new_value).replace(",", "").split()[0])
                if n <= 0:
                    return _err("area must be positive")
                updates[col] = n
            elif col == "ownership_percent":
                n = float(str(new_value).replace("%", "").strip())
                if not 0 < n <= 100:
                    return _err("ownership_percent must be between 0 and 100")
                updates[col] = n
            elif col == "occupancy_status":
                occ = normalize_occupancy(new_value)
                if not occ:
                    return _err("occupancy_status must be Tenanted, Vacant or Self-occupied")
                updates[col] = occ
            elif col == "property_type":
                t = normalize_type(new_value)
                if not t:
                    return _err("property_type must be Retail, Office or Residential")
                updates.update(property_type_raw=new_value.strip().title(), type_group=t["type_group"], asset_class=t["asset_class"])
            elif col == "city":
                c = normalize_city(new_value)
                updates.update(city=c, metro=metro_for(c))
            elif col == "locality":
                updates[col] = tidy_locality(new_value)
            else:
                updates[col] = new_value.strip()
        except (ValueError, IndexError) as e:
            return _err(f"could not read the new value: {e}")

        # keep occupancy and rent consistent
        if "occupancy_status" in updates and updates["occupancy_status"] != "Tenanted" and p["annual_rent_inr"]:
            updates["annual_rent_inr"] = 0
            effects.append(f"Annual rent reset to ₹0 because the property is now {updates['occupancy_status']}.")
        if "occupancy_status" in updates and updates["occupancy_status"] == "Tenanted" and not p["annual_rent_inr"]:
            effects.append("Now Tenanted but annual rent is still ₹0 — ask the user for the rent.")
        if updates.get("annual_rent_inr") and p["occupancy_status"] != "Tenanted" and "occupancy_status" not in updates:
            updates["occupancy_status"] = "Tenanted"
            effects.append("Occupancy set to Tenanted because a rent was given.")

        old = {k: p[k] for k in updates}
        if all(old[k] == v for k, v in updates.items()):
            return _j({"status": "unchanged", "property_id": p["property_id"], "message": "That is already the current value."})
        newp = db.update_property_fields(uid, p["property_id"], updates)
        shown = "annual_rent_inr" if col == "annual_rent_inr" else col
        fmt = (lambda v: format_inr(v)) if shown in _MONEY else (lambda v: v)
        out = {"status": "updated", "property_id": p["property_id"], "label": _label(newp), "field": _FIELD_LABELS.get(shown, shown),
               "old": fmt(p[shown]), "new": fmt(newp[shown]), "side_effects": effects, "portfolio_now": _portfolio_now()}
        if shown == "current_estimated_value_inr" and p[shown]:
            pct = (newp[shown] - p[shown]) / p[shown] * 100
            out["change_pct"] = round(pct, 1)
            if abs(pct) > config.LARGE_CHANGE_PCT:
                out["large_change"] = True
                out["warning"] = f"Large change ({pct:+.1f}%). Echo it and ask the user to double-check."
                ctx.large_changes.append(f"{p['property_id']} value {pct:+.0f}%")
        return _j(out)

    def remove_property(property_ref: str, confirmed: bool = False) -> str:
        own = db.list_properties(uid)
        p, bad = _resolve_one(own, property_ref)
        if bad:
            return bad
        key = (conv, p["property_id"])
        with _lock:
            asked = _PENDING_REMOVALS.get(key)
            live = asked is not None and time.time() - asked < _PENDING_TTL_S
            if not (confirmed and live):
                _PENDING_REMOVALS[key] = time.time()
                return _j({"status": "confirmation_required", "property": _card(p),
                           "message": "Nothing removed. Ask the user to confirm removing this property (yes/no)."})
            _PENDING_REMOVALS.pop(key, None)
        db.update_property_fields(uid, p["property_id"], {"status": "Inactive"}, action="remove")
        return _j({"status": "removed", "property": _card(p), "note": "Soft-deleted (status Inactive); support can restore it.",
                   "portfolio_now": _portfolio_now()})

    def flag_for_human(reason: str) -> str:
        db.add_attention(conv, [f"handoff: {reason}"])
        ctx.handoff_reasons.append(reason)
        return _j({"status": "flagged", "message": "The team has been notified. Tell the user a teammate will follow up."})

    def mk(fn, name, desc, schema):
        return StructuredTool.from_function(func=fn, name=name, description=desc, args_schema=schema)

    return [
        mk(get_portfolio_analysis, "get_portfolio_analysis",
           "The ONE tool for reading the portfolio: lists, totals, rent, yield, vacancy, breakdowns, highest/lowest, "
           "concentration, comparisons and what-if scenarios. Use filters to narrow (retail only, Mumbai, above 10 crore). "
           "Scenario arguments (exclude_properties, value_overrides, add_hypothetical_property) never change stored data. "
           "All numbers, incl. formatted ₹ strings, come pre-computed; never do maths yourself.", AnalysisArgs),
        mk(find_properties, "find_properties",
           "Resolve a vague reference ('the Bandra one') to concrete properties. Returns found / ambiguous / none. "
           "Only needed when you must be sure which property is meant; update/remove/exclude accept references directly.", FindArgs),
        mk(add_property, "add_property",
           "Add a property to the portfolio. Required: type, location, area, value. Returns missing_fields (nothing written) "
           "if any are absent, or ask_optional once for rent/occupancy.", AddArgs),
        mk(update_property, "update_property",
           "Change one field of one property. Executes immediately and returns old -> new. Pass amounts as the user wrote them; "
           "use monthly_rent if the user gave a monthly figure.", UpdateArgs),
        mk(remove_property, "remove_property",
           "Remove a property (soft delete). First call always returns confirmation_required; call again with confirmed=true only "
           "after the user says yes.", RemoveArgs),
        mk(flag_for_human, "flag_for_human",
           "Hand the conversation to a human on the team (legal/tax advice, buy/sell requests, user asks for a person, "
           "repeated confusion, unrecoverable tool error).", FlagArgs),
    ]
