"""
HDB rental - build the modelling file
    original cleaned columns (42)
  + refined versions of earlier features (15)
  + new features (24)
  = 81 columns, saved as hdb_rentals_with_features.csv

See feature_dictionary.csv (English + Chinese) and FEATURES.md.
Only pandas and numpy are needed. Run in VS Code / Jupyter, or: python create_features.py

RULES
1. Rent-based features only look at months STRICTLY BEFORE the row's month (no leakage).
   For a time-based train/test split set TEST_START (e.g. "2025-09"): test rows then only
   use history from before that month.
2. Columns with too many blanks were not created (ft_rent_yoy_change_lag 20%,
   exits_per_station_800m 19%).
3. The only blanks left (1.54% of rows: Jan 2021 + 14 one-room rows) are filled with the
   nearest later month's market level and flagged in market_level_filled.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------------
INPUT_PATH = Path("data") / "hdb_rentals_clean.csv"
OUTPUT_PATH = INPUT_PATH.with_name("hdb_rentals_with_features.csv")
TEST_START = None          # e.g. "2025-09" for a time-based split; None = use all earlier months

# NOTE: dist_nearest_exit_km / dist_nearest_mrt_km are actually in METRES (median ~460).
# The code below treats them as metres.

MATURE_TOWNS = {"ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CENTRAL",
                "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS",
                "QUEENSTOWN", "SERANGOON", "TAMPINES", "TOA PAYOH"}
REGION = {
    "CENTRAL": ["BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CENTRAL", "GEYLANG", "KALLANG/WHAMPOA",
                "MARINE PARADE", "QUEENSTOWN", "TOA PAYOH"],
    "EAST": ["BEDOK", "PASIR RIS", "TAMPINES"],
    "NORTH": ["SEMBAWANG", "WOODLANDS", "YISHUN"],
    "NORTH-EAST": ["ANG MO KIO", "HOUGANG", "PUNGGOL", "SENGKANG", "SERANGOON"],
    "WEST": ["BUKIT BATOK", "BUKIT PANJANG", "CHOA CHU KANG", "CLEMENTI", "JURONG EAST",
             "JURONG WEST", "TENGAH"],
}
TOWN_TO_REGION = {t: r for r, towns in REGION.items() for t in towns}

# URA postal district by postal sector (first 2 digits of postal code)
DISTRICT_SECTORS = {
    1: [1, 2, 3, 4, 5, 6], 2: [7, 8], 3: [14, 15, 16], 4: [9, 10], 5: [11, 12, 13], 6: [17],
    7: [18, 19], 8: [20, 21], 9: [22, 23], 10: [24, 25, 26, 27], 11: [28, 29, 30],
    12: [31, 32, 33], 13: [34, 35, 36, 37], 14: [38, 39, 40, 41], 15: [42, 43, 44, 45],
    16: [46, 47, 48], 17: [49, 50, 81], 18: [51, 52], 19: [53, 54, 55, 82], 20: [56, 57],
    21: [58, 59], 22: [60, 61, 62, 63, 64], 23: [65, 66, 67, 68], 24: [69, 70, 71],
    25: [72, 73], 26: [77, 78], 27: [75, 76], 28: [79, 80],
}
SECTOR_TO_DISTRICT = {s: d for d, secs in DISTRICT_SECTORS.items() for s in secs}

# Unresolved station labels in the data (station codes) -> station names
STATION_ALIAS = {"CC30": "KEPPEL", "CC31": "CANTONMENT", "CC9": "PAYA LEBAR",
                 "DT4": "HUME", "NE18": "PUNGGOL COAST"}

FLAT_TYPE_ORD = {"1-ROOM": 1, "2-ROOM": 2, "3-ROOM": 3, "4-ROOM": 4, "5-ROOM": 5, "EXECUTIVE": 6}


# ----------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------
def lookup_month_table(table, key_series, t_series):
    """table: DataFrame indexed by month t (rows) with one column per key.
    Returns table[t, key] for each row (NaN where missing)."""
    try:
        stacked = table.stack(future_stack=True)      # pandas >= 2.1
    except TypeError:
        stacked = table.stack(dropna=False)           # older pandas
    idx = pd.MultiIndex.from_arrays([t_series.to_numpy(), key_series.astype(str).to_numpy()])
    return stacked.reindex(idx).to_numpy()


def make_key(df, cols):
    key = df[cols[0]].astype(str)
    for c in cols[1:]:
        key = key + "|" + df[c].astype(str)
    return key


def rolling_median_table(df, key, window, min_rows, n_months):
    """Median monthly_rent per key over months [t-window, t-1], for every t in 0..n_months.
    Returns (median table, count table), rows = t, columns = key values."""
    keys = df[key].astype(str)
    med, cnt = {}, {}
    for t in range(n_months + 1):
        in_window = (df["_m"] >= t - window) & (df["_m"] < t) & (df["_m"] < df["_cut"].iloc[0])
        g = df.loc[in_window, "monthly_rent"].groupby(keys[in_window])
        med[t], cnt[t] = g.median(), g.size()
    med = pd.DataFrame(med).T
    cnt = pd.DataFrame(cnt).T.reindex_like(med).fillna(0)
    return med.where(cnt >= min_rows), cnt


def past_premium(df, key_cols, shrink):
    """Shrunken mean of the month-adjusted residual (_resid) for the same key over all
    months before the row's month. Returns (premium, n_past)."""
    key = make_key(df, key_cols)
    hist = df["_m"] < df["_cut"]
    agg = (df.loc[hist].groupby([key[hist], df.loc[hist, "_m"]])["_resid"]
           .agg(["sum", "count"]))
    sums = agg["sum"].unstack(0).reindex(range(df["_n_months"].iloc[0] + 1)).fillna(0)
    cnts = agg["count"].unstack(0).reindex(range(df["_n_months"].iloc[0] + 1)).fillna(0)
    # cumulative up to t-1 (shift by one month so the current month is excluded)
    csum = sums.cumsum().shift(1).fillna(0)
    ccnt = cnts.cumsum().shift(1).fillna(0)
    s = lookup_month_table(csum, key, df["_t"])
    n = lookup_month_table(ccnt, key, df["_t"])
    s = np.nan_to_num(s); n = np.nan_to_num(n)
    return s / (n + shrink), n


def rolling_count(df, key_cols, window):
    """Number of rental rows with the same key in months [t-window, t-1]."""
    key = make_key(df, key_cols)
    hist = df["_m"] < df["_cut"]
    cnt = (df.loc[hist].groupby([df.loc[hist, "_m"], key[hist]]).size().unstack(1)
           .reindex(range(df["_n_months"].iloc[0] + 1)).fillna(0))
    rolled = cnt.rolling(window, min_periods=1).sum().shift(1).fillna(0)
    return np.nan_to_num(lookup_month_table(rolled, key, df["_t"]))


def pct_rank_unique_blocks(df, value_col, group_col="town"):
    """Percentile rank of value_col among UNIQUE blocks (address_key) in the same group."""
    blocks = df.drop_duplicates("address_key")[["address_key", group_col, value_col]].copy()
    blocks["_rank"] = blocks.groupby(group_col)[value_col].rank(pct=True, method="average")
    return df["address_key"].map(blocks.set_index("address_key")["_rank"])


def add_refined_existing_features(df):
    """15 refined versions of earlier candidate features.
    Dropped candidates: hdb_rental_stock_share, small_flat_stock_share (94% zeros),
    matches_dominant_block_type (5% blank, repeats block_share_this_flat_type),
    rent_month_sin / rent_month_cos (no real seasonality), multiple_mrt_stations_within_800m
    (repeats n_mrt_stations_within_800m), mrt_km_x_1_room (only 61 one-room rows)."""
    df = df.copy()
    tot = df["block_total_units"].replace(0, np.nan)
    mrt_m = df["dist_nearest_mrt_km"]          # stored in metres despite the name
    exit_m = df["dist_nearest_exit_km"]

    # Block composition
    df["large_flat_stock_share"] = (df["5room_sold"] + df["exec_sold"] + df["multigen_sold"]) / tot
    by_type = pd.DataFrame({
        "1-ROOM": df["1room_sold"] + df["1room_rental"],
        "2-ROOM": df["2room_sold"] + df["2room_rental"] + df["studio_apartment_sold"],
        "3-ROOM": df["3room_sold"] + df["3room_rental"],
        "4-ROOM": df["4room_sold"],
        "5-ROOM": df["5room_sold"],
        "EXECUTIVE": df["exec_sold"] + df["multigen_sold"],
    })
    top = by_type.max(axis=1)
    tied = by_type.eq(top, axis=0).sum(axis=1) > 1
    df["dominant_block_flat_type"] = by_type.idxmax(axis=1).where(~tied, "TIED")
    df["dominant_flat_type_share"] = top / tot

    # Block density
    df["approx_units_per_floor"] = df["block_total_units"] / df["block_max_floor"].replace(0, np.nan)
    df["commercial_x_units_per_floor"] = (df["flag_commercial"] == "Y") * df["approx_units_per_floor"]

    # Time (use INSTEAD of rent_year + rent_month in the model, not together)
    df["months_since_2021_01"] = (df["rent_year"] - 2021) * 12 + df["rent_month"] - 1

    # Rail
    df["extra_distance_to_mrt_m"] = mrt_m - exit_m
    df["rail_access_category"] = np.select(
        [mrt_m <= 400, mrt_m <= 800, exit_m <= 400], ["MRT_400", "MRT_800", "RAIL_400"], "OTHER_ACCESS")
    df["mrt_stations_400_to_800m"] = df["n_mrt_stations_within_800m"] - df["n_mrt_stations_within_400m"]

    # Interactions (mainly useful for linear models)
    df["flat_type_town"] = df["flat_type"].astype(str) + " | " + df["town"].astype(str)
    for ftype in ["2-ROOM", "3-ROOM", "4-ROOM", "5-ROOM", "EXECUTIVE"]:
        name = "mrt_km_x_" + ftype.lower().replace("-", "_")
        df[name] = (df["flat_type"] == ftype) * mrt_m / 1000
    return df


REFINED_COLUMNS = [
    "large_flat_stock_share", "dominant_block_flat_type", "dominant_flat_type_share",
    "approx_units_per_floor", "commercial_x_units_per_floor", "months_since_2021_01",
    "extra_distance_to_mrt_m", "rail_access_category", "mrt_stations_400_to_800m", "flat_type_town",
    "mrt_km_x_2_room", "mrt_km_x_3_room", "mrt_km_x_4_room", "mrt_km_x_5_room", "mrt_km_x_executive",
]

# ----------------------------------------------------------------------------------
# NEW FEATURES
# ----------------------------------------------------------------------------------
def add_new_features(df, test_start=None):
    """24 new features (groups A-H). Rent-based ones use earlier months only."""
    df = df.copy()
    df["_m"] = (df["rent_year"] - 2021) * 12 + df["rent_month"] - 1          # month index
    n_months = int(df["_m"].max()) + 1
    cut = n_months if test_start is None else (int(test_start[:4]) - 2021) * 12 + int(test_start[5:7]) - 1
    df["_cut"] = cut
    df["_n_months"] = n_months
    df["_t"] = np.minimum(df["_m"], cut)          # rows after the cutoff see history up to the cutoff
    ft = df["flat_type"].astype(str)
    is_lrt = df["nearest_exit_is_lrt"].astype(str).str.upper().eq("TRUE")

    # ---------------- A. Market level (lagged) ----------------
    ft_med, _ = rolling_median_table(df, "flat_type", window=3, min_rows=5, n_months=n_months)
    all_med, _ = rolling_median_table(df.assign(_all="ALL"), "_all", window=3, min_rows=1, n_months=n_months)
    # sparse flat types (1-ROOM): fall back to overall median x average past ratio
    ratio = (ft_med.div(all_med["ALL"], axis=0)).expanding().mean().shift(1)
    ft_med = ft_med.fillna(ratio.mul(all_med["ALL"], axis=0))
    df["ft_median_rent_lag3m"] = lookup_month_table(ft_med, ft, df["_t"])

    df["_town_ft"] = df["town"].astype(str) + "|" + ft
    tf_med, _ = rolling_median_table(df, "_town_ft", window=6, min_rows=10, n_months=n_months)
    df["town_ft_median_rent_lag6m"] = lookup_month_table(tf_med, df["_town_ft"], df["_t"])
    df["town_ft_median_rent_lag6m"] = df["town_ft_median_rent_lag6m"].fillna(df["ft_median_rent_lag3m"])

    # (ft_rent_yoy_change_lag removed: 20% blank, and blanks must not be filled)

    # ---------------- B. Location premiums (past-only target encoding) ----------------
    # residual = log(rent) - log(median rent of the same flat type in the same month)
    same_month_med = df.groupby(["flat_type", "_m"])["monthly_rent"].transform("median")
    df["_resid"] = np.log(df["monthly_rent"]) - np.log(same_month_med)

    df["block_ft_past_premium"], df["block_ft_past_n"] = past_premium(df, ["address_key", "flat_type"], shrink=3)
    df["block_ft_past_n"] = df["block_ft_past_n"].astype(int)
    df["street_past_premium"], _ = past_premium(df, ["street_name"], shrink=10)
    df["_station"] = df["nearest_exit_station"].replace(STATION_ALIAS)
    df["station_past_premium"], _ = past_premium(df, ["_station"], shrink=20)
    df["expected_rent_block_ft"] = df["ft_median_rent_lag3m"] * np.exp(df["block_ft_past_premium"])

    # ---------------- C. Location from existing columns ----------------
    sector = (df["postal"] // 10000).astype(int)
    df["postal_sector"] = sector.map("{:02d}".format)
    df["postal_district"] = sector.map(SECTOR_TO_DISTRICT).map(lambda d: f"D{int(d):02d}" if pd.notna(d) else np.nan)
    df["mature_estate"] = df["town"].isin(MATURE_TOWNS).astype(int)
    df["region"] = df["town"].map(TOWN_TO_REGION).fillna("OTHER")

    # ---------------- D. Rail ----------------
    # (exits_per_station_800m removed: 19% blank, and blanks must not be filled)
    df["lrt_dependent"] = (is_lrt & (df["dist_nearest_mrt_km"] > 800)).astype(int)   # column is in metres
    df["mrt_dist_rank_in_town"] = pct_rank_unique_blocks(df, "dist_nearest_mrt_km")

    # ---------------- E. Building ----------------
    df["remaining_lease_years"] = 99 - df["building_age"]
    completion_year = df["rent_year"] - df["building_age"]
    df["completion_era"] = pd.cut(completion_year, bins=[-np.inf, 1979, 1989, 1999, 2009, np.inf],
                                  labels=["<1980", "1980s", "1990s", "2000s", "2010+"]).astype(str)
    blk_year = df.drop_duplicates(["address_key", "rent_year"])
    town_age_med = blk_year.groupby(["town", "rent_year"])["building_age"].median().rename("_town_age_med")
    df = df.join(town_age_med, on=["town", "rent_year"])
    df["age_vs_town_median"] = df["building_age"] - df["_town_age_med"]
    df["floor_rank_in_town"] = pct_rank_unique_blocks(df, "block_max_floor")
    stock = (df.drop_duplicates(["address_key", "flat_type"])
             .groupby(["town", "flat_type"])["block_units_this_flat_type"].sum().rename("town_ft_stock"))
    df = df.join(stock, on=["town", "flat_type"])

    # ---------------- F. Rental activity ----------------
    df["block_rentals_past12m"] = rolling_count(df, ["address_key"], window=12).astype(int)
    df["block_rental_intensity"] = df["block_rentals_past12m"] / df["block_total_units"].replace(0, np.nan)
    df["town_ft_rentals_lag3m"] = rolling_count(df, ["town", "flat_type"], window=3).astype(int)

    # ---------------- G. Unit / data quality ----------------
    df["flat_type_ord"] = ft.map(FLAT_TYPE_ORD)
    df["flat_type_missing_in_block"] = (df["block_units_this_flat_type"] == 0).astype(int)

    # ---------------- H. Fill the few blanks (1.54% of rows) ----------------
    # Jan 2021 has no earlier month, and a few 1-ROOM months have too little history.
    # Fill with the NEAREST LATER month's value of the same flat type (the first market level
    # that exists), and flag the rows so the model / analyst knows they were filled.
    df["market_level_filled"] = df["ft_median_rent_lag3m"].isna().astype(int)
    ft_first = ft_med.bfill()                                    # earliest known level per flat type
    fill_vals = pd.Series(lookup_month_table(ft_first, ft, df["_t"]), index=df.index)
    df["ft_median_rent_lag3m"] = df["ft_median_rent_lag3m"].fillna(fill_vals)
    df["town_ft_median_rent_lag6m"] = df["town_ft_median_rent_lag6m"].fillna(df["ft_median_rent_lag3m"])
    df["expected_rent_block_ft"] = df["ft_median_rent_lag3m"] * np.exp(df["block_ft_past_premium"])

    helper_cols = [c for c in df.columns if c.startswith("_")]
    return df.drop(columns=helper_cols)


NEW_COLUMNS = [
    "ft_median_rent_lag3m", "town_ft_median_rent_lag6m", "block_ft_past_premium", "block_ft_past_n",
    "street_past_premium", "station_past_premium", "expected_rent_block_ft",
    "postal_sector", "postal_district", "mature_estate", "region",
    "lrt_dependent", "mrt_dist_rank_in_town",
    "remaining_lease_years", "completion_era", "age_vs_town_median", "floor_rank_in_town", "town_ft_stock",
    "block_rentals_past12m", "block_rental_intensity", "town_ft_rentals_lag3m",
    "flat_type_ord", "flat_type_missing_in_block", "market_level_filled",
]


def build_all_features(raw, test_start=None):
    """Original cleaned columns, then 15 refined existing features, then 24 new features."""
    original = list(raw.columns)
    out = add_new_features(add_refined_existing_features(raw), test_start=test_start)
    return out[original + REFINED_COLUMNS + NEW_COLUMNS]


if __name__ == "__main__":
    import sys
    # Notebooks pass their own "--f=kernel.json" argument, so only accept .csv arguments.
    csv_args = [a for a in sys.argv[1:] if a.lower().endswith(".csv")]
    in_path = Path(csv_args[0]) if csv_args else INPUT_PATH
    out_path = Path(csv_args[1]) if len(csv_args) > 1 else OUTPUT_PATH

    raw = pd.read_csv(in_path, dtype={"block": str})
    out = build_all_features(raw, test_start=TEST_START)
    assert len(out) == len(raw), "row count changed"
    out.to_csv(out_path, index=False)

    added = REFINED_COLUMNS + NEW_COLUMNS
    blanks = out[added].isna().sum()
    print(f"Saved {out_path}")
    print(f"{len(out):,} rows | {raw.shape[1]} original + {len(REFINED_COLUMNS)} refined + "
          f"{len(NEW_COLUMNS)} new = {out.shape[1]} columns")
    print("Blank values in added columns:", int(blanks.sum()))
    print("Rows with filled market level (market_level_filled = 1):", int(out["market_level_filled"].sum()))
