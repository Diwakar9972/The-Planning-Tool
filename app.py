"""
Levisol Monthly Planning Tool — planner-facing UI (Streamlit).

Run locally:   pip install streamlit pandas scipy openpyxl
               streamlit run app.py

For a non-technical planner:
  1. Drop in the monthly data workbook (same layout as the case file).
  2. Adjust levers in the sidebar (capacities, freight, hub buffers, policy).
  3. Click "Generate plan" — norms + optimal plan + costs in about a minute.
  4. Compare saved scenarios side by side; see routing on the India map.
Shortage never crashes the tool: it appears as priced unmet demand.
"""
import io
import time

import pandas as pd
import streamlit as st

from levisol_tool.data_loader import load_all
from levisol_tool.norms import compute_cfa_norms, compute_hub_norms
from levisol_tool.optimizer import build_and_solve, PLANTS, HUBS
from levisol_tool.report import write_report

# ---- geography for the routing map ----
COORDS = {
    'BOM': (19.076, 72.877), 'AHM_PLANT': (23.023, 72.571), 'KOL_PLANT': (22.573, 88.364),
    'MHW': (19.300, 73.060), 'MHE': (22.650, 88.450),
    'Guwahati': (26.144, 91.736), 'Kolkata': (22.573, 88.364), 'Jamshedpur': (22.805, 86.203),
    'Kanpur': (26.450, 80.332), 'Haryana': (28.459, 77.029), 'Rajpura': (30.484, 76.594),
    'Bhiwandi': (19.297, 73.063), 'Ahmedabad': (23.023, 72.571),
    'Bangalore': (12.972, 77.594), 'Hyderabad': (17.385, 78.487),
}
PLANT_COORD = {'BOM': COORDS['BOM'], 'AHM': COORDS['AHM_PLANT'], 'KOL': COORDS['KOL_PLANT']}

st.set_page_config(page_title='Levisol Planning Tool', layout='wide', page_icon='🛢️')
st.title('🛢️ Levisol Monthly Planning Tool')
st.caption('Deterministic inventory norms + exact least-cost production & distribution plan. '
           'Same inputs always give the same plan.')

if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}

up = st.file_uploader('Step 1 — upload the monthly data workbook (.xlsx, case-file layout)',
                      type=['xlsx'])

with st.sidebar:
    st.header('Planning levers')
    scen_name = st.text_input('Scenario name', value='Base')
    cfa_target = st.selectbox('CFA end-of-month stock target', ['SS', 'ROP', 'none'], index=0,
                              help='Top-up level beyond the Jan forecast at each CFA.')
    cmult = st.number_input('Contractual unmet-demand multiplier (×)', 1.0, 100.0, 10.0, 1.0)
    hub_sc = st.number_input('Hub SS shortfall cost (₹/kL)', 0.0, 500000.0, 50000.0, 5000.0)
    hub_scale = st.slider('Hub safety-stock requirement (× computed norm)', 0.0, 2.0, 1.0, 0.05,
                          help='Scale every hub buffer up or down — e.g. 0.5 releases '
                               'working capital, 1.5 adds resilience.')
    tlim = st.slider('Solver time limit (seconds)', 15, 300, 60, 15)

if up:
    with st.spinner('Reading workbook…'):
        data = load_all(io.BytesIO(up.read()))
    st.success(f"Loaded {len(data['jan'])} SKU-CFA demand rows, "
               f"{len(data['sku_penalty'])} SKUs, 3 plants, 2 hubs, 10 CFAs.")

    with st.sidebar:
        st.divider()
        st.header('What-if overrides')
        cap_df = pd.DataFrame(data['plant_capacity']).T
        cap_df.index.name = 'Plant'
        new_cap = st.data_editor(cap_df, key='cap', use_container_width=True)
        st.caption('Line capacities (kL/month).')
        ph_rows = [(p, h, data['plant_hub_cost'][(p, h)]) for p in PLANTS for h in HUBS]
        new_ph = st.data_editor(pd.DataFrame(ph_rows, columns=['Plant', 'Hub', '₹/kL']),
                                key='ph', hide_index=True, use_container_width=True)
        st.caption('Plant→Hub freight (₹/kL).')

    for p in new_cap.index:
        data['plant_capacity'][p] = new_cap.loc[p].astype(float).to_dict()
    for _, r in new_ph.iterrows():
        data['plant_hub_cost'][(r['Plant'], r['Hub'])] = float(r['₹/kL'])

    if st.button('Step 2 — Generate plan', type='primary'):
        t0 = time.time()
        with st.spinner('Computing inventory norms…'):
            cn = compute_cfa_norms(data)
            hn = compute_hub_norms(data, cn)
        with st.spinner(f'Optimising (exact MILP, up to {tlim}s)…'):
            res = build_and_solve(data, cn, hn, cfa_stock_target=cfa_target,
                                  contractual_multiplier=cmult,
                                  hub_shortfall_cost=hub_sc,
                                  hub_ss_scale=hub_scale, time_limit=tlim)
        st.session_state.scenarios[scen_name] = dict(
            res=res, cn=cn, hn=hn,
            params=dict(cfa_stock_target=cfa_target, contractual_multiplier=cmult,
                        hub_shortfall_cost=hub_sc, hub_ss_scale=hub_scale))
        st.info(f"Solved in {time.time()-t0:.0f}s — {res['mip_status']}")

    if st.session_state.scenarios:
        pick = st.selectbox('View scenario', list(st.session_state.scenarios.keys()),
                            index=len(st.session_state.scenarios) - 1)
        sc = st.session_state.scenarios[pick]
        res, cn, hn = sc['res'], sc['cn'], sc['hn']
        cs = res['cost_summary'].set_index('Cost head')['₹']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Total plan cost', f"₹{cs['TOTAL']/1e7:.2f} Cr")
        c2.metric('Production', f"₹{cs['Production']/1e7:.2f} Cr")
        c3.metric('Total freight',
                  f"₹{(cs['Plant→Hub freight']+cs['Hub→CFA freight'])/1e7:.2f} Cr")
        unmet_kl = res['unmet']['Unmet (kL)'].sum() if len(res['unmet']) else 0.0
        c4.metric('Unmet demand', f'{unmet_kl:.1f} kL', delta_color='inverse',
                  delta=None if unmet_kl == 0 else
                  f"₹{cs['Unmet-demand penalty']/1e5:.1f} L penalty")

        tabs = st.tabs(['Cost summary', 'Routing map', 'Compare scenarios',
                        'Production plan', 'Plant→Hub', 'Hub→CFA', 'Unmet demand',
                        'Hub stock', 'Norms — CFA', 'Norms — Hub'])

        tabs[0].dataframe(res['cost_summary'], use_container_width=True)

        # ---- map-based routing output (pydeck ships with streamlit) ----
        with tabs[1]:
            import pydeck as pdk
            ph = res['plant_hub'].groupby(['Plant', 'Hub'])['Volume (kL)'].sum().reset_index()
            hc = res['hub_cfa'].groupby(['Hub', 'CFA'])['Volume (kL)'].sum().reset_index()
            arcs = []
            for _, r in ph.iterrows():
                (la1, lo1), (la2, lo2) = PLANT_COORD[r['Plant']], COORDS[r['Hub']]
                arcs.append(dict(frm=[lo1, la1], to=[lo2, la2], vol=r['Volume (kL)'],
                                 label=f"{r['Plant']}→{r['Hub']}: {r['Volume (kL)']:.0f} kL",
                                 color=[31, 78, 51]))
            for _, r in hc.iterrows():
                (la1, lo1), (la2, lo2) = COORDS[r['Hub']], COORDS[r['CFA']]
                arcs.append(dict(frm=[lo1, la1], to=[lo2, la2], vol=r['Volume (kL)'],
                                 label=f"{r['Hub']}→{r['CFA']}: {r['Volume (kL)']:.0f} kL",
                                 color=[201, 162, 39]))
            arc_df = pd.DataFrame(arcs)
            arc_df['width'] = 1 + 9 * arc_df['vol'] / arc_df['vol'].max()
            pts = pd.DataFrame(
                [dict(name=k, lon=v[1], lat=v[0]) for k, v in
                 {**{p: PLANT_COORD[p] for p in PLANTS},
                  **{h: COORDS[h] for h in HUBS},
                  **{c: COORDS[c] for c in hc['CFA'].unique()}}.items()])
            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(latitude=22, longitude=80, zoom=4),
                layers=[
                    pdk.Layer('ArcLayer', arc_df, get_source_position='frm',
                              get_target_position='to', get_width='width',
                              get_source_color='color', get_target_color='color',
                              pickable=True),
                    pdk.Layer('ScatterplotLayer', pts, get_position='[lon, lat]',
                              get_radius=25000, get_fill_color=[31, 78, 51, 200]),
                    pdk.Layer('TextLayer', pts, get_position='[lon, lat]', get_text='name',
                              get_size=12, get_color=[40, 40, 40]),
                ],
                tooltip={'text': '{label}'}))
            st.caption('Green arcs: plant→hub. Gold arcs: hub→CFA. Width ∝ volume. '
                       'Hover an arc for the lane volume.')

        # ---- scenario comparison ----
        with tabs[2]:
            if len(st.session_state.scenarios) < 2:
                st.info('Run a second scenario (change a lever, rename it, Generate) '
                        'to compare side by side.')
            else:
                comp = {}
                for name, s in st.session_state.scenarios.items():
                    c = s['res']['cost_summary'].set_index('Cost head')['₹']
                    u = s['res']['unmet']
                    comp[name] = {
                        'Total (₹ Cr)': c['TOTAL'] / 1e7,
                        'Production (₹ Cr)': c['Production'] / 1e7,
                        'Freight (₹ Cr)': (c['Plant→Hub freight'] + c['Hub→CFA freight']) / 1e7,
                        'Penalties (₹ Cr)': (c['Unmet-demand penalty'] +
                                             c['Hub SS shortfall penalty']) / 1e7,
                        'Unmet (kL)': u['Unmet (kL)'].sum() if len(u) else 0.0,
                        'Hub SS scale (×)': s['params']['hub_ss_scale'],
                    }
                cmp_df = pd.DataFrame(comp).round(3)
                st.dataframe(cmp_df, use_container_width=True)
                base_col = st.selectbox('Delta vs', list(comp.keys()))
                st.dataframe((cmp_df.sub(cmp_df[base_col], axis=0)).round(3)
                             .style.format('{:+.3f}'), use_container_width=True)

        tabs[3].dataframe(res['production'], use_container_width=True)
        tabs[4].dataframe(res['plant_hub'], use_container_width=True)
        tabs[5].dataframe(res['hub_cfa'], use_container_width=True)
        if len(res['unmet']):
            u = res['unmet'].copy()
            u['Tier'] = u['SKU'].map(data['tier'])
            u['Contractual'] = u['SKU'].map(data['sku_contract'])
            tabs[6].warning(f'{unmet_kl:.2f} kL deliberately unserved — details below.')
            tabs[6].dataframe(u, use_container_width=True)
        else:
            tabs[6].success('All CFA net requirements fully met.')
        tabs[7].dataframe(res['hub_ending'], use_container_width=True)
        tabs[8].dataframe(cn, use_container_width=True)
        tabs[9].dataframe(hn, use_container_width=True)

        buf = io.BytesIO()
        write_report(buf, data, cn, hn, res, sc['params'])
        st.download_button('⬇️ Download full plan workbook (.xlsx)', buf.getvalue(),
                           file_name=f'Levisol_Plan_{pick}.xlsx')
else:
    st.info('Upload the data workbook to begin. If demand exceeds capacity the tool shows '
            'exactly what is unmet, by how much, and at what cost — it never crashes.')
