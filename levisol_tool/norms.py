"""
Levisol Planning Tool — Deterministic Inventory Norms
Closed-form (King's formula) safety stock / reorder point engine.

Per SKU x CFA:
  d        = avg daily demand   = mean(6-month sales) / 30 working days
  sigma_d  = daily demand risk  = stdev(monthly forecast error) / sqrt(30)
             (buffers against FORECAST ERROR, not raw demand — the forecast
              is what the plan is built on, so the error is the true risk)
  L        = production LT + plant->hub LT + hub->CFA LT   (days)
  sigma_L  = sqrt(production variability^2 + transit variability^2)
  SS       = z * sqrt( L*sigma_d^2 + d^2*sigma_L^2 )
  ROP      = d*L + SS
  DOC      = ROP / d
z per tier (Exhibit F): A=98% -> 2.054, B=97% -> 1.881, C/D=92% -> 1.405.

Per SKU x Hub (98% service level mandated):
  demand = sum of daily demand of CFAs historically served by that hub,
  variance pooled across those CFAs (independence assumed, per assignment),
  L = production LT + plant->hub LT, sigma_L = production variability.
"""
import numpy as np
import pandas as pd
from .data_loader import MONTHS, WORKING_DAYS_PER_MONTH, TIER_Z, HUB_Z


def compute_cfa_norms(data: dict) -> pd.DataFrame:
    sales, fcst, lt = data['sales'], data['fcst'], data['lt']
    key = ['Product Name', 'CFA']
    m = sales[key + MONTHS].merge(fcst[key + MONTHS], on=key, suffixes=('_act', '_fc'))
    m = m.merge(lt[key + ['hub_hist', 'lt_plant_hub', 'lt_hub_cfa', 'lt_prod',
                          'var_prod', 'var_transit', 'CFA region']], on=key, how='left')

    act = m[[f'{mm}_act' for mm in MONTHS]].to_numpy(float)
    fc = m[[f'{mm}_fc' for mm in MONTHS]].to_numpy(float)

    d_daily = act.mean(axis=1) / WORKING_DAYS_PER_MONTH
    fe = act - fc                                   # forecast error per month
    sigma_fe_m = fe.std(axis=1, ddof=1)             # monthly sigma of error
    sigma_d = sigma_fe_m / np.sqrt(WORKING_DAYS_PER_MONTH)   # daily sigma

    L = (m['lt_prod'] + m['lt_plant_hub'] + m['lt_hub_cfa']).to_numpy(float)
    sigma_L = np.sqrt(m['var_prod'].to_numpy(float) ** 2 + m['var_transit'].to_numpy(float) ** 2)

    tier = m['Product Name'].map(data['tier'])
    z = tier.map(TIER_Z).to_numpy(float)

    ss = z * np.sqrt(L * sigma_d ** 2 + (d_daily ** 2) * sigma_L ** 2)
    rop = d_daily * L + ss
    doc = np.divide(rop, d_daily, out=np.zeros_like(rop), where=d_daily > 1e-9)

    out = pd.DataFrame({
        'SKU': m['Product Name'], 'CFA': m['CFA'], 'Region': m['CFA region'],
        'Hub (historical)': m['hub_hist'], 'Tier': tier, 'Service level z': z,
        'Avg daily demand (kL)': d_daily, 'Sigma demand daily (kL)': sigma_d,
        'Lead time (days)': L, 'Sigma lead time (days)': sigma_L,
        'Safety stock (kL)': ss, 'Reorder point (kL)': rop, 'Days of cover': doc,
    })
    return out.round(3)


def compute_hub_norms(data: dict, cfa_norms: pd.DataFrame) -> pd.DataFrame:
    """Pool CFA demand up to the hub that historically serves it (Exhibit E)."""
    lt = data['lt']
    g = cfa_norms.copy()
    g['var_daily'] = g['Sigma demand daily (kL)'] ** 2
    agg = g.groupby(['SKU', 'Hub (historical)']).agg(
        d_daily=('Avg daily demand (kL)', 'sum'),
        var_daily=('var_daily', 'sum'),            # independence assumed
    ).reset_index()

    up = lt.groupby(['Product Name', 'hub_hist']).agg(
        lt_up=('lt_prod', 'max'), lt_ph=('lt_plant_hub', 'max'), var_prod=('var_prod', 'max')
    ).reset_index().rename(columns={'Product Name': 'SKU', 'hub_hist': 'Hub (historical)'})
    agg = agg.merge(up, on=['SKU', 'Hub (historical)'], how='left')

    L = (agg['lt_up'] + agg['lt_ph']).to_numpy(float)
    sigma_L = agg['var_prod'].to_numpy(float)
    d = agg['d_daily'].to_numpy(float)
    sd = np.sqrt(agg['var_daily'].to_numpy(float))
    ss = HUB_Z * np.sqrt(L * sd ** 2 + d ** 2 * sigma_L ** 2)
    rop = d * L + ss
    doc = np.divide(rop, d, out=np.zeros_like(rop), where=d > 1e-9)

    return pd.DataFrame({
        'SKU': agg['SKU'], 'Hub': agg['Hub (historical)'],
        'Avg daily demand (kL)': d, 'Sigma demand daily (kL)': sd,
        'Lead time (days)': L, 'Sigma lead time (days)': sigma_L,
        'Service level z': HUB_Z,
        'Safety stock (kL)': ss, 'Reorder point (kL)': rop, 'Days of cover': doc,
    }).round(3)
