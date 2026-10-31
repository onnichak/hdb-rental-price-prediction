# Feature reference: HDB rental modelling file

**Output:** `hdb_rentals_with_features.csv`
**Size:** 209,610 rows × **81 columns** = 42 cleaned columns + 15 refined features + 24 new features
**Blank values in the added columns:** **0**

| File | What it is |
|---|---|
| `hdb_rentals_clean.csv` | Cleaned input (42 columns). Unchanged. |
| `create_features.py` | Script that builds the modelling file (pandas + numpy only, about 20 seconds) |
| `hdb_rentals_with_features.csv` | Output described here |
| `feature_dictionary.csv` | Dictionary for all 39 added columns, in **English and Chinese**: definition, why it was added, formula, dependent columns, data type, blanks, notes. Open it in Excel; Chinese displays correctly. |

**How to run:** open `create_features.py` in VS Code or Jupyter and run it. The input and output paths are set at the top (`INPUT_PATH`, `OUTPUT_PATH`).

---

## 1. Column layout

The columns come in this order:

| Columns | Count | Where they come from |
|---|---:|---|
| `rent_approval_date` … `postal` | 42 | Cleaned data, unchanged |
| `large_flat_stock_share` … `mrt_km_x_executive` | 15 | **Refined**: the useful features kept from the original `hdb_engineered_feature_dictionary.csv` |
| `ft_median_rent_lag3m` … `market_level_filled` | 24 | **New** features |

### Refined: what was kept and dropped from the original feature dictionary
The original dictionary had 22 features. **15 were kept** and **7 were dropped** because they add little or repeat other columns:

| Dropped | Why |
|---|---|
| `hdb_rental_stock_share`, `small_flat_stock_share` | 94% of rows are 0, so almost no information |
| `matches_dominant_block_type` | 5% blank, and mostly repeats `block_share_this_flat_type` (0.77 correlation) |
| `rent_month_sin`, `rent_month_cos` | No real seasonality; the "month effect" is just the 2022–23 price surge |
| `multiple_mrt_stations_within_800m` | Repeats `n_mrt_stations_within_800m` (0.8 correlation) |
| `mrt_km_x_1_room` | Only 61 one-room rows in the whole file |

### New: columns considered but not created
| Not created | Would have been blank | Why |
|---|---:|---|
| `ft_rent_yoy_change_lag` | 20.10% | Needs 12 months of history; one market-wide number per flat type per month; repeats the time trend |
| `exits_per_station_800m` | 18.86% | Blank whenever no MRT station is within 800m; no real link to rent (correlation 0.00) |

---

## 2. What the new features are (24)

| Group | Columns | In one line |
|---|---|---|
| **A. Market level** | `ft_median_rent_lag3m`, `town_ft_median_rent_lag6m` | Recent market rent for the same flat type (and town) |
| **B. Location premium** | `block_ft_past_premium`, `block_ft_past_n`, `street_past_premium`, `station_past_premium`, `expected_rent_block_ft` | How much this block, street or station area has rented above or below the market in the past, plus a simple rent estimate |
| **C. Location** | `postal_sector`, `postal_district`, `mature_estate`, `region` | Finer or standard location groups |
| **D. Rail** | `lrt_dependent`, `mrt_dist_rank_in_town` | LRT-only access; MRT closeness compared with the rest of the town |
| **E. Building** | `remaining_lease_years`, `completion_era`, `age_vs_town_median`, `floor_rank_in_town`, `town_ft_stock` | Lease, build generation, age and height compared with the town, supply |
| **F. Rental activity** | `block_rentals_past12m`, `block_rental_intensity`, `town_ft_rentals_lag3m` | How busy the rental market is for the block and the town |
| **G. Unit / quality** | `flat_type_ord`, `flat_type_missing_in_block` | Flat type as a number; data-mismatch flag |
| **H. Fill flag** | `market_level_filled` | 1 = the market-level values in this row were filled (see section 4) |

The strongest new column is `expected_rent_block_ft` (correlation with rent **0.77**). Full definitions and formulas are in the dictionary.

---

## 3. The "past months only" rule (no leakage)

Columns in groups **A, B and F** are built from past rentals. They only use rentals from **months before the row's month**, never the row's own rent or anything else from the same month. We tested this: changing May 2024 rents did not change any May 2024 feature.

**If you split train/test by time**, set `TEST_START = "2025-09"` (or your first test month) at the top of the script before running. Test rows will then only use history from before the test period, like a real forecast. With the default (`None`), every row uses all months before it.

---

## 4. Blanks and how they were filled

After removing the two high-blank columns above, only 3 columns had blanks, all in the **same 3,232 rows (1.54%)**:

| Column | Rows filled | Why blank | How filled |
|---|---:|---|---|
| `ft_median_rent_lag3m` | 3,232 | Jan 2021 (3,218 rows) has no earlier month; 14 one-room rows had too little history (only 61 one-room rows in total) | The **nearest later month's value** for the same flat type. For Jan 2021 this is the Jan 2021 median (e.g. 4-ROOM $2,100, 3-ROOM $1,800) |
| `town_ft_median_rent_lag6m` | 3,232 | Same rows | Filled `ft_median_rent_lag3m` |
| `expected_rent_block_ft` | 3,232 | Same rows | Recalculated from the filled `ft_median_rent_lag3m` |

- These rows are flagged: `market_level_filled = 1`.
- The fill uses the month's own median for these 1.54% of rows. That's a small exception to the "past months only" rule. To be strict, exclude `market_level_filled == 1` rows from the **test** set, or from all training.

**Every other added column has no blanks.**

---

## 5. Zeros are real values, not missing

| Column | Rows = 0 | Meaning |
|---|---:|---|
| `block_ft_past_n` | 17,052 | First time this block and flat type appears in the data |
| `block_ft_past_premium` | 18,341 | No history yet, so it's assumed to rent at the market level |
| `street_past_premium` / `station_past_premium` | 3,396 / 3,231 | No earlier rentals on that street or near that station (mostly Jan 2021) |
| `block_rentals_past12m`, `block_rental_intensity` | 16,322 | No rentals in the block in the past 12 months |
| `extra_distance_to_mrt_m` | 179,319 | The nearest rail exit is already an MRT exit |
| `mrt_km_x_*` | most rows | The row is a different flat type (each column is only filled for its own type) |
| `commercial_x_units_per_floor` | 167,390 | The block has no commercial use |
| `town_ft_stock` | 4 | 1-ROOM in Pasir Ris / Choa Chu Kang: property data has no one-room units (data mismatch) |

---

## 6. How to use it in a model

**Don't feed all 80 non-target columns in blindly.** Some columns repeat each other, and some are IDs:

| Column(s) | What to do |
|---|---|
| `monthly_rent` | **Target**: never a feature |
| `rent_approval_date`, `block`, `street_name`, `address_key`, `postal` | IDs or keys: don't use as features |
| `rent_year` + `rent_month` **vs** `months_since_2021_01` | Use **one**: either the pair or `months_since_2021_01` (using all three made results worse in testing) |
| `building_age` vs `remaining_lease_years` | Same information: use one |
| `flat_type` vs `flat_type_ord` | Same information: use one in linear models |
| `ft_median_rent_lag3m`, `town_ft_median_rent_lag6m`, `expected_rent_block_ft` | 0.91–0.93 correlated with each other. Fine for trees; linear models keep only `expected_rent_block_ft` |
| `flat_type_town`, `mrt_km_x_*`, `commercial_x_units_per_floor`, `rail_access_category`, `mature_estate`, `region` | Mainly help **linear** models; tree models learn these patterns themselves |

**Categories:** convert these to category type before modelling (`df[col] = df[col].astype("category")`):
`town, flat_type, flag_*, nearest_exit_station, dominant_block_flat_type, rail_access_category, flat_type_town, postal_sector, postal_district, region, completion_era`

**Reading the file back:**
```python
df = pd.read_csv("hdb_rentals_with_features.csv", dtype={"postal_sector": str})
df["postal_sector"] = df["postal_sector"].str.zfill(2)      # keep "05", not 5
```

---

## 7. Things to know

1. **Distances are in metres.** `dist_nearest_exit_km` and `dist_nearest_mrt_km` are stored in metres (median about 460) despite the `_km` name. All new rail features account for this; `mrt_km_x_*` are true kilometres.
2. **Expect only a modest gain in accuracy.** Flats of the same type, in the same block, rented in the same month, still differ by about $424 (standard deviation). That comes from things the data doesn't have: storey, renovation, furnishing. In a quick time-split test, a LightGBM model already reached about $430 RMSE with the original columns. So most new features help **linear** models (Ridge: $651 → $478 RMSE with the location-premium features) far more than tree models.
3. **Check the 179 rows** with `flat_type_missing_in_block = 1`. The rental record's flat type doesn't exist in that block's property data.
