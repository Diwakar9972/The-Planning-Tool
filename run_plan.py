#!/usr/bin/env python3
"""
Levisol Monthly Planning Tool — command-line runner.

Usage:
    python run_plan.py <input_case_data.xlsx> <output_plan.xlsx>
        [--cfa-target SS|ROP|none] [--contract-mult 10]
        [--hub-shortfall-cost 50000] [--time-limit 60]

Everything is deterministic: same inputs -> same plan.
"""
import argparse
import sys
import time

from levisol_tool.data_loader import load_all
from levisol_tool.norms import compute_cfa_norms, compute_hub_norms
from levisol_tool.optimizer import build_and_solve
from levisol_tool.report import write_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--cfa-target', default='SS', choices=['SS', 'ROP', 'none'])
    ap.add_argument('--contract-mult', type=float, default=10.0)
    ap.add_argument('--hub-shortfall-cost', type=float, default=50000.0)
    ap.add_argument('--hub-ss-scale', type=float, default=1.0)
    ap.add_argument('--time-limit', type=int, default=60)
    a = ap.parse_args()

    t0 = time.time()
    print('Loading data ...', flush=True)
    data = load_all(a.input)
    print('Computing inventory norms ...', flush=True)
    cn = compute_cfa_norms(data)
    hn = compute_hub_norms(data, cn)
    print('Solving production & distribution MILP (HiGHS) ...', flush=True)
    res = build_and_solve(data, cn, hn,
                          cfa_stock_target=a.cfa_target,
                          contractual_multiplier=a.contract_mult,
                          hub_shortfall_cost=a.hub_shortfall_cost,
                          hub_ss_scale=a.hub_ss_scale,
                          time_limit=a.time_limit)
    params = dict(cfa_stock_target=a.cfa_target, contractual_multiplier=a.contract_mult,
                  hub_shortfall_cost=a.hub_shortfall_cost, hub_ss_scale=a.hub_ss_scale)
    write_report(a.output, data, cn, hn, res, params)

    print(f"\nDone in {time.time()-t0:.0f}s  |  {res['mip_status']}")
    print(res['cost_summary'].to_string(index=False))
    u = res['unmet']
    if len(u):
        print(f"\nUnmet demand: {u['Unmet (kL)'].sum():.2f} kL across {len(u)} SKU-CFA rows "
              f"(see 'Unmet Demand' sheet — all deliberate, priced trade-offs).")
    else:
        print('\nAll CFA net requirements fully met.')
    print(f'Plan written to {a.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
