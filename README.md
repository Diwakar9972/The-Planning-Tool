# Levisol Monthly Planning Tool

Deterministic inventory norms + exact least-cost production & distribution MILP.
Same inputs always produce the same plan.

## Requirements
Python 3.10+, `pandas`, `scipy>=1.9`, `openpyxl` (and `streamlit` for the UI).
    pip install pandas scipy openpyxl streamlit

## Two ways to run

**1. Command line (fastest):**
    python run_plan.py "Supply Chain Case Study - Data.xlsx" Plan_Jan2026.xlsx
Options: --cfa-target SS|ROP|none  --contract-mult 10  --hub-shortfall-cost 50000  --time-limit 60

**2. Planner UI (for the live demo):**
    streamlit run app.py
Upload the data workbook, adjust levers in the sidebar (capacities, freight,
policy parameters), click Generate plan, download the output workbook.

## Handling the assessment-day modified inputs
Edit the input workbook (capacities in Exhibit A, freight in B/C, demand in J,
penalties in D) or override capacities/freight directly in the UI sidebar —
then rerun. Shortage never crashes the tool: it appears as priced unmet
demand with SKU, CFA, volume, and cost.

## Package layout
    levisol_tool/data_loader.py  parse exhibits, tiering, pack-size slabs
    levisol_tool/norms.py        safety stock / ROP / days-of-cover engine
    levisol_tool/optimizer.py    HiGHS MILP (production, routing, unmet, hub SS)
    levisol_tool/report.py       formatted Excel output with live cost formulas
    run_plan.py                  CLI
    app.py                       Streamlit planner UI
    METHODOLOGY.md               methodology & assumptions document
