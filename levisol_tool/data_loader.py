"""
Levisol Planning Tool — Data Loader
Parses the case study Excel (Exhibits A–J) into clean pandas tables.
Deterministic: no randomness anywhere in this pipeline.
"""
import re
import numpy as np
import pandas as pd

MONTHS = ['Jul-25', 'Aug-25', 'Sep-25', 'Oct-25', 'Nov-25', 'Dec-25']
WORKING_DAYS_PER_MONTH = 30  # per case instruction

# Tier cut-offs on cumulative volume share (Exhibit F volume slabs)
TIER_CUTS = [('A', 0.50), ('B', 0.80), ('C', 0.95), ('D', 1.01)]
TIER_Z = {'A': 2.054, 'B': 1.881, 'C': 1.405, 'D': 1.405}  # z for 98/97/92/92%
HUB_Z = 2.054  # 98% hub service level, per assignment

LINE_SLABS = ['<=1.5LT', '3-5LT', '7-20LT', '50LT', '180-210LT']


def _unit_size_lt(pack: str) -> float:
    """'20 X 900 ML' -> 0.9 ; '1 X 210 LT' -> 210 ; '1 X 180 KG' -> 180 (treated as LT-class)."""
    m = re.search(r'X\s*([\d.]+)\s*(ML|LT|KG)', pack, re.IGNORECASE)
    size, unit = float(m.group(1)), m.group(2).upper()
    return size / 1000.0 if unit == 'ML' else size


def slab_for_pack(pack: str) -> str:
    s = _unit_size_lt(pack)
    if s <= 1.5:
        return '<=1.5LT'
    if 3 <= s <= 5:
        return '3-5LT'
    if 7 <= s <= 20:
        return '7-20LT'
    if s == 50:
        return '50LT'
    return '180-210LT'


def load_all(path: str) -> dict:
    """Load every exhibit; return dict of tidy DataFrames + lookup dicts."""
    plants = pd.read_excel(path, sheet_name='A - Plants & Production', header=2).dropna(subset=['Plant Code'])
    plants = plants[pd.to_numeric(plants['Production Cost (₹/kl)'], errors='coerce').notna()]
    plants.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in plants.columns]
    cap_cols = {
        '<=1.5LT': [c for c in plants.columns if '<=1.5' in c][0],
        '3-5LT': [c for c in plants.columns if '3- 5' in c][0],
        '7-20LT': [c for c in plants.columns if '7- 20' in c][0],
        '50LT': [c for c in plants.columns if '50 LT' in c][0],
        '180-210LT': [c for c in plants.columns if '180' in c][0],
    }
    plant_capacity = {p: {slab: float(plants.set_index('Plant Code').loc[p, col])
                          for slab, col in cap_cols.items()}
                      for p in plants['Plant Code']}
    prod_cost = plants.set_index('Plant Code')['Production Cost (₹/kl)'].astype(float).to_dict()

    b = pd.read_excel(path, sheet_name='B - Plant-Hub Transport', header=2).dropna(subset=['From Plant'])
    b = b[b['From Plant'].isin(['Mumbai', 'Ahmedabad', 'Kolkata'])]
    name2code = {'Mumbai': 'BOM', 'Ahmedabad': 'AHM', 'Kolkata': 'KOL'}
    plant_hub_cost = {}
    for _, r in b.iterrows():
        p = name2code[r['From Plant']]
        plant_hub_cost[(p, 'MHW')] = float(r['To Mother Hub West (MHW)'])
        plant_hub_cost[(p, 'MHE')] = float(r['To Mother Hub East (MHE)'])

    c = pd.read_excel(path, sheet_name='C -Hub-CFA Transport', header=2).dropna(subset=['CFA'])
    c = c[pd.to_numeric(c['From Mother Hub West (MHW)'], errors='coerce').notna()]
    hub_cfa_cost, cfa_region = {}, {}
    for _, r in c.iterrows():
        cfa = str(r['CFA']).strip()
        hub_cfa_cost[('MHW', cfa)] = float(r['From Mother Hub West (MHW)'])
        hub_cfa_cost[('MHE', cfa)] = float(r['From Mother Hub East (MHE)'])
        cfa_region[cfa] = str(r['Region']).strip()

    d = pd.read_excel(path, sheet_name='D -SKU Portfolio+Penalty matrix', header=2).dropna(subset=['Product Name'])
    d['Contractual'] = d['Contractual?'].astype(str).str.upper().str.contains('YES')
    sku_penalty = d.set_index('Product Name')['Penalty cost (per kL)'].astype(float).to_dict()
    sku_contract = d.set_index('Product Name')['Contractual'].to_dict()
    sku_pack = d.set_index('Product Name')['Pack size'].astype(str).to_dict()
    sku_slab = {k: slab_for_pack(v) for k, v in sku_pack.items()}

    e = pd.read_excel(path, sheet_name='E - Source + LT data', header=2).dropna(subset=['Product Name'])
    e['CFA'] = e['CFA'].astype(str).str.replace(' CFA', '', regex=False).str.strip()
    e.columns = [re.sub(r'\s+', ' ', str(col)).strip() for col in e.columns]
    e = e.rename(columns={
        'LT (Plant to Hub)(in days)': 'lt_plant_hub',
        'LT (Hub to CFA ) (in days)': 'lt_hub_cfa',
        'Production lead time (in days)': 'lt_prod',
        'Production variability (in days)': 'var_prod',
        'Transit lead variability (in days)': 'var_transit',
    })
    e['hub_hist'] = np.where(e['Source'].str.strip().str.lower() == 'east', 'MHE', 'MHW')

    def _hist(sheet, header):
        g = pd.read_excel(path, sheet_name=sheet, header=header).dropna(subset=['Product Name'])
        g['CFA'] = g['CFA'].astype(str).str.replace(' CFA', '', regex=False).str.strip()
        g = g.rename(columns={f'{m} (in kL)': m for m in MONTHS})
        return g

    sales = _hist('G - Sales History', 2)
    fcst = _hist('H - Forecast History', 3)

    inv = pd.read_excel(path, sheet_name='I - Expected opening Inventory', header=3).dropna(subset=['Product Name'])
    inv['CFA'] = inv['CFA'].astype(str).str.strip()
    inv_col = [c2 for c2 in inv.columns if 'Jan' in str(c2)][0]
    hub_inv = inv[inv['CFA'].str.contains('Mother Hub')].copy()
    hub_inv['Hub'] = np.where(hub_inv['CFA'].str.contains('West'), 'MHW', 'MHE')
    cfa_inv = inv[~inv['CFA'].str.contains('Mother Hub')].copy()
    cfa_inv['CFA'] = cfa_inv['CFA'].str.replace(' CFA', '', regex=False).str.strip()

    jan = pd.read_excel(path, sheet_name='J - Jan Forecast', header=3).dropna(subset=['Product Name'])
    jan['CFA'] = jan['CFA'].astype(str).str.replace(' CFA', '', regex=False).str.strip()
    jan_col = [c2 for c2 in jan.columns if 'Jan' in str(c2)][0]

    # ---- Tier assignment from 6-month national volume (Exhibit F slabs) ----
    vol = sales.groupby('Product Name')[MONTHS].sum().sum(axis=1).sort_values(ascending=False)
    cum_share = vol.cumsum() / vol.sum()
    tier = {}
    for skucode, cshare in cum_share.items():
        for t, cut in TIER_CUTS:
            if cshare <= cut:
                tier[skucode] = t
                break
    return dict(
        plant_capacity=plant_capacity, prod_cost=prod_cost,
        plant_hub_cost=plant_hub_cost, hub_cfa_cost=hub_cfa_cost, cfa_region=cfa_region,
        sku_penalty=sku_penalty, sku_contract=sku_contract, sku_pack=sku_pack, sku_slab=sku_slab,
        lt=e, sales=sales, fcst=fcst,
        cfa_inv=cfa_inv.rename(columns={inv_col: 'open_inv'}),
        hub_inv=hub_inv.rename(columns={inv_col: 'open_inv'}),
        jan=jan.rename(columns={jan_col: 'jan_fcst'}),
        tier=tier,
    )
