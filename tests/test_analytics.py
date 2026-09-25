import pytest

from app import db
from app.analytics import (appreciation, compare, format_inr, parse_inr, portfolio_analysis,
                           preference_mismatches)
from app.normalize import metro_for, normalize_city, normalize_type, split_location

CR = 10_000_000


@pytest.mark.parametrize("text,expected", [
    ("₹12 Cr", 12 * CR), ("1.2 crore", 12_000_000), ("50 lakh", 5_000_000), ("50L", 5_000_000),
    ("5,00,000", 500_000), ("Rs. 4.2 crores", 42_000_000), ("12.5cr", 125_000_000), (42_000_000, 42_000_000),
    ("₹ 3 lakhs", 300_000), ("10 Cr.", 100_000_000), ("0", 0), ("1.05 cr", 10_500_000),
])
def test_parse_inr(text, expected):
    assert parse_inr(text) == expected


@pytest.mark.parametrize("bad", ["", "abc", "12 dollars", None, "₹", "1.2.3 cr"])
def test_parse_inr_rejects(bad):
    with pytest.raises(ValueError):
        parse_inr(bad)


@pytest.mark.parametrize("n,expected", [
    (120_000_000, "₹12.00 Cr"), (4_500_000, "₹45.0 L"), (12_345, "₹12,345"), (1_234_567, "₹12.3 L"),
    (9_996_000, "₹1.00 Cr"),  # rounds up across the L/Cr boundary instead of "₹100.0 L"
    (0, "₹0"), (-4_500_000, "-₹45.0 L"), (1_234_500_000_000, "₹123,450.00 Cr"), (123_456, "₹1.2 L"),
    (99_999, "₹99,999"),
])
def test_format_inr(n, expected):
    assert format_inr(n) == expected


def test_roundtrip():
    for s in ("₹12.50 Cr", "₹45.0 L"):
        assert format_inr(parse_inr(s)) == s


def test_type_map_merges_office():
    assert normalize_type("Commercial Office")["type_group"] == "Office"
    assert normalize_type("Office")["type_group"] == "Office"
    assert normalize_type("Retail") == {"type_group": "Retail", "asset_class": "Commercial"}
    assert normalize_type("Residential")["asset_class"] == "Residential"
    assert normalize_type("a 3bhk flat")["type_group"] == "Residential"
    assert normalize_type("shop")["type_group"] == "Retail"
    assert normalize_type("commercial") is None  # ambiguous: retail or office?


def test_city_normalisation():
    assert normalize_city("gurgaon") == "Gurugram"
    assert normalize_city("Bangalore") == "Bengaluru"
    assert metro_for("Noida") == "Delhi NCR" and metro_for("Gurugram") == "Delhi NCR"
    assert metro_for("Alibaug") == "Alibaug"


def test_split_location_state_tail():
    assert split_location("Bandra West, Mumbai") == ("Bandra West", "Mumbai")
    assert split_location("Alibaug, Maharashtra") == ("Alibaug", "Alibaug")  # not a city called Maharashtra
    assert split_location("Noida Sector 62, Noida") == ("Noida Sector 62", "Noida")


@pytest.mark.parametrize("uid,value,rent,count", [
    ("U001", 297_000_000, 13_200_000, 3), ("U002", 193_000_000, 2_400_000, 3),
    ("U003", 487_000_000, 26_400_000, 3), ("U004", 138_000_000, 7_800_000, 3),
])
def test_totals(props, uid, value, rent, count):
    r = portfolio_analysis(props[uid])
    assert r["totals"]["value"] == value and r["totals"]["annual_rent"] == rent and r["totals"]["count"] == count
    assert r["is_hypothetical"] is False and r["scenario_description"] is None and "delta_vs_actual" not in r


def test_yield_both_ways(props):
    y = portfolio_analysis(props["U002"])["yield"]
    assert y["on_all_assets_pct"] == 1.24          # 2.4M / 193M
    assert y["on_income_producing_assets_pct"] == 4.71  # 2.4M / 51M (only P005 is rented)


def test_vacancy_excludes_self_occupied(props):
    v = portfolio_analysis(props["U002"])["vacancy"]
    assert v["vacant_count"] == 1 and v["self_occupied_count"] == 1
    assert v["vacant_pct_by_value"] == 40.4  # P006 ₹7.8 Cr / ₹19.3 Cr, P004 self-occupied not counted


def test_breakdown_and_concentration(props):
    r = portfolio_analysis(props["U001"])
    tg = {row["key"]: row for row in r["breakdowns"]["type_group"]}
    assert tg["Retail"]["share_pct"] == 71.4 and tg["Office"]["share_pct"] == 28.6
    dims = {(f["dimension"], f["key"]) for f in r["concentration_flags"]}
    assert ("type_group", "Retail") in dims and ("metro", "Mumbai") in dims
    assert ("city", "Mumbai") not in dims  # metro flag subsumes the identical city flag


def test_alibaug_not_in_mumbai(props):
    r = portfolio_analysis(props["U002"], filters={"location": "Mumbai"})
    assert {p["property_id"] for p in r["properties"]} == {"P004", "P005"}
    assert r["share_of_full_portfolio_pct"] == 59.6


def test_gurgaon_alias_filter(props):
    r = portfolio_analysis(props["U003"], filters={"location": "Gurgaon"})
    assert {p["property_id"] for p in r["properties"]} == {"P007", "P009"}
    assert len(portfolio_analysis(props["U003"], filters={"location": "Delhi NCR"})["properties"]) == 3


def test_r001_retail(props):
    r = portfolio_analysis(props["U001"], filters={"type_group": "Retail"})
    assert [p["property_id"] for p in r["properties"]] == ["P001", "P003"]


def test_r002_above_10cr(props):
    r = portfolio_analysis(props["U001"], filters={"min_value": parse_inr("10 crore")})
    assert [p["property_id"] for p in r["properties"]] == ["P001"]


def test_r004_highest_rent(props):
    r = portfolio_analysis(props["U003"])
    assert r["highlights"]["highest_rent"]["property_id"] == "P007"
    assert r["highlights"]["highest_rent"]["rent_fmt"] == "₹1.68 Cr"


def test_which_performs_better_retail(props):
    r = portfolio_analysis(props["U001"], filters={"type_group": "Retail"})
    assert r["highlights"]["best_yield"]["property_id"] == "P003"        # 6.52% vs 6.00%
    assert r["highlights"]["highest_rent"]["property_id"] == "P001"      # but P001 has the higher rent


def test_group_by(props):
    r = portfolio_analysis(props["U004"], group_by="asset_class")
    rows = {x["key"]: x for x in r["group_by"]["rows"]}
    assert rows["Commercial"]["share_pct"] == 75.4 and rows["Residential"]["share_pct"] == 24.6
    with pytest.raises(ValueError):
        portfolio_analysis(props["U004"], group_by="colour")


def test_exclude_scenario(props):
    before = db.list_properties("U001")
    r = portfolio_analysis(props["U001"], exclude_ids=["P001"])
    assert r["is_hypothetical"] is True
    assert "excluding P001" in r["scenario_description"] and "Bandra West" in r["scenario_description"]
    assert r["totals"]["value"] == 177_000_000 and r["totals"]["count"] == 2
    d = r["delta_vs_actual"]["metrics"]
    assert d["value"]["actual"] == 297_000_000 and d["value"]["change"] == -120_000_000
    assert d["value"]["change_fmt"] == "-₹12.00 Cr"
    assert r["actual"]["totals"]["value"] == 297_000_000
    assert db.list_properties("U001") == before  # DB untouched


def test_override_scenario(props):
    r = portfolio_analysis(props["U001"], overrides={"P001": {"current_estimated_value_inr": 140_000_000}})
    assert r["is_hypothetical"] and r["totals"]["value"] == 317_000_000
    assert r["delta_vs_actual"]["metrics"]["value"]["change"] == 20_000_000
    assert "P001" in r["scenario_description"] and "₹14.00 Cr" in r["scenario_description"]


def test_add_hypothetical(props):
    r = portfolio_analysis(props["U004"], add_hypothetical={
        "type_group": "Retail", "asset_class": "Commercial", "locality": "Indiranagar", "city": "Bengaluru",
        "area_sqft": 3000, "current_estimated_value_inr": 42_000_000, "annual_rent_inr": 3_000_000})
    assert r["totals"]["count"] == 4 and r["totals"]["value"] == 180_000_000
    assert any(p.get("hypothetical") for p in r["properties"])
    assert "hypothetical" in r["scenario_description"]
    assert db.list_properties("U004").__len__() == 3


def test_scenario_unknown_id(props):
    with pytest.raises(ValueError):
        portfolio_analysis(props["U001"], exclude_ids=["P999"])


def test_scenario_concentration_changes(props):
    # U003 is 72.9% Office. Excluding the ₹24 Cr Gurugram office drops Office to 46.6% -> flag disappears.
    r = portfolio_analysis(props["U003"], exclude_ids=["P007"])
    actual = {(f["dimension"], f["key"]) for f in r["delta_vs_actual"]["actual_concentration_flags"]}
    scenario = {(f["dimension"], f["key"]) for f in r["concentration_flags"]}
    assert ("type_group", "Office") in actual and ("type_group", "Office") not in scenario
    assert ("metro", "Delhi NCR") in scenario
    shift = {s["key"]: s for s in r["delta_vs_actual"]["share_shifts"]["type_group"]}
    assert shift["Office"]["actual_share_pct"] == 72.9 and shift["Office"]["scenario_share_pct"] == 46.6


def test_scenario_single_property_left_has_no_flags(props):
    assert portfolio_analysis(props["U003"], exclude_ids=["P007", "P008"])["concentration_flags"] == []


def test_empty_portfolio():
    r = portfolio_analysis([])
    assert r["empty"] is True and "totals" not in r


def test_single_property_no_concentration(props):
    r = portfolio_analysis(props["U001"], filters={"location": "Andheri"})
    assert r["totals"]["count"] == 1 and r["yield"]["on_income_producing_assets_pct"] is None
    assert portfolio_analysis([props["U001"][0]])["concentration_flags"] == []


def test_inactive_excluded(props):
    p = [dict(x) for x in props["U001"]]
    p[0]["status"] = "Inactive"
    assert portfolio_analysis(p)["totals"]["count"] == 2


def test_ownership_percent_multiplier(props):
    p = [dict(x) for x in props["U001"]]
    p[0]["ownership_percent"] = 50
    r = portfolio_analysis(p)
    assert r["totals"]["value"] == 297_000_000 - 60_000_000 and r["totals"]["annual_rent"] == 13_200_000 - 3_600_000


def test_compare_asset_class(props):
    c = compare(props["U004"], "asset_class", "Residential", "Commercial")
    assert c["a"]["value"] == 34_000_000 and c["b"]["value"] == 104_000_000
    assert c["higher_value"] == "Commercial" and c["higher_yield_on_all_assets"] == "Commercial"
    assert c["a"]["share_pct"] == 24.6


def test_compare_bad_dimension(props):
    with pytest.raises(ValueError):
        compare(props["U004"], "colour", "a", "b")


def test_appreciation_unavailable_without_purchase_price(props):
    a = appreciation(props["U001"])
    assert a["available"] is False and "purchase" in a["reason"]


def test_appreciation_gain_but_no_annualised(props):
    p = [dict(x) for x in props["U001"]]
    p[0]["purchase_price_inr"] = 90_000_000
    a = appreciation(p)
    assert a["available"] and a["total_absolute_gain"] == 30_000_000
    assert a["annualised_return"]["available"] is False and "date" in a["annualised_return"]["reason"]


def test_preference_mismatch(props):
    u2 = db.get_user("U002")
    m = preference_mismatches(props["U002"], u2)
    assert [x["property_id"] for x in m] == ["P006"]  # Alibaug is outside Bandra / Worli / Lower Parel
    assert preference_mismatches(props["U003"], db.get_user("U003")) == []
    assert [x["property_id"] for x in preference_mismatches(props["U004"], db.get_user("U004"))] == ["P012"]


def test_insight_candidates_present(props):
    r = portfolio_analysis(props["U002"], user=db.get_user("U002"))
    types = [i["type"] for i in r["insight_candidates"]]
    assert "vacancy" in types and "preference_mismatch" in types


def test_area_is_preformatted(props):
    r = portfolio_analysis(props["U001"])
    assert r["totals"]["area_fmt"] == "16,100 sq ft" and r["properties"][0]["area_fmt"] == "5,200 sq ft"
