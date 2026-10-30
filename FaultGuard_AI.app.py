import hmac
import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import faultguard_core as fg
import report_pdf as pdfrep

st.set_page_config(page_title="FaultGuard AI", page_icon="⚡", layout="wide")

STRETCH = "stretch"
NAVY, BLUE = "#0b2e59", "#4da6ff"
GREEN, AMBER, RED, TESTC = "#1a9e5c", "#d99a00", "#d63a3a", "#2a78d6"
STATUS_COLOR = {'EXCELLENT': GREEN, 'WATCH': AMBER, 'ACTION': RED, 'TEST': TESTC}
STATUS_LABEL = {'EXCELLENT': 'NORMAL OPERATION', 'WATCH': 'WATCH (Review Recommended)', 'ACTION': 'ACTION REQUIRED',
                'TEST': 'TEST RECORD (not a system fault)'}
PH_COL = {'A': '#d63a3a', 'B': '#d99a00', 'C': '#2a78d6', 'N': '#555555'}
MODE_ORDER = fg.MODE_ORDER
SRC_PAGES, SRC_DEMO = "Results from the Incident Analysis page", "Demo substation (7 incidents)"

st.markdown("""
<style>
.pg-header {background:#0b2e59;padding:16px 22px;border-radius:10px;margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;gap:16px;}
.pg-header img {height:64px;background:#fff;border-radius:8px;padding:4px 8px;}
.pg-header h1 {color:#fff;margin:0;font-size:1.9rem;}
.pg-header p {color:#9fd0ff;margin:4px 0 0 0;font-size:1rem;}
.pg-banner {border:2px solid;border-radius:8px;padding:10px 14px;text-align:center;margin:6px 0 12px 0;}
.pg-banner h2 {margin:0;font-size:1.5rem;}
.pg-banner p {margin:2px 0 0 0;font-size:.95rem;}
.pg-note {background:rgba(77,166,255,.10);border-left:4px solid #4da6ff;padding:8px 12px;border-radius:4px;font-size:.92rem;margin-bottom:10px;}
.pg-card {border:1px solid #d5dce6;border-radius:10px;padding:12px 14px;text-align:center;}
.pg-card h4 {margin:0;color:#0b2e59;font-size:1.05rem;}
.pg-card .big {font-size:1.6rem;font-weight:700;margin:4px 0;}
.pg-card .sub {font-size:.85rem;color:#5a6575;}
.pg-foot {text-align:center;color:#888;font-size:12px;padding:10px;}
</style>
""", unsafe_allow_html=True)

FOOTER = """
<div class='pg-foot'><strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
<em>Confidential and Proprietary. The predictive algorithms and expert system overrides within this application are protected intellectual property.</em></div>
"""


def banner(title, sub, color):
    st.markdown(f"<div class='pg-banner' style='border-color:{color}'><h2 style='color:{color}'>{title}</h2>"
                f"<p>{sub}</p></div>", unsafe_allow_html=True)


def secrets_ok():
    try:
        return "APP_USERNAME" in st.secrets and "APP_PASSWORD" in st.secrets
    except Exception:
        return False


# ==========================================
# --- 1. Login ---
# ==========================================
for key, default in [('logged_in', False), ('register', []), ('last', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state['logged_in']:
    st.markdown("<br><br><h1 style='text-align:center;color:#4da6ff;'>⚡ FaultGuard AI Login</h1>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.warning("🔒 Restricted Access. Authorized Personnel Only.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Secure Login", width=STRETCH):
                if not secrets_ok():
                    st.error("Login is not configured: add APP_USERNAME and APP_PASSWORD to the app's Secrets.")
                elif hmac.compare_digest(username, st.secrets["APP_USERNAME"]) and \
                        hmac.compare_digest(password, st.secrets["APP_PASSWORD"]):
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password")
    st.markdown(FOOTER, unsafe_allow_html=True)
    st.stop()

# ==========================================
# --- 2. Header and model ---
# ==========================================
_, col_r = st.columns([10, 1])
with col_r:
    if st.button("Logout 🚪"):
        st.session_state['logged_in'] = False
        st.rerun()


@st.cache_data
def logo_b64():
    try:
        import base64
        from PIL import Image
        im = Image.open('logo.png')
        im.thumbnail((360, 200))
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


_logo = logo_b64()
_logo_html = f"<img src='data:image/png;base64,{_logo}' alt='PowerGuard AI'>" if _logo else ""
st.markdown(f"""
<div class='pg-header'>
  <div>
    <h1>⚡ FaultGuard AI: Trip and Incident Analysis</h1>
    <p>Reads the relay disturbance record and event list and writes the incident report: fault type, location, protection and breaker performance, auto-reclose and actions</p>
  </div>
  {_logo_html}
</div>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the AI model…")
def MODEL():
    return fg.load_model('faultguard_model.pkl')


if MODEL() is None:
    st.error("AI model file 'faultguard_model.pkl' not found.")
    st.stop()


# ==========================================
# --- 3. Records and demo ---
# ==========================================
def status_key(a):
    return 'TEST' if a.get('likely_test') else a['Status']


def to_record(a, source):
    m = a.get('meta', {})
    loc = f"{a['loc_km']:.1f} km ({a['loc_pu'] * 100:.0f}%)" if a.get('loc_km') is not None else '-'
    f1 = lambda v: round(float(v), 1) if v is not None else None
    return {'Incident': m.get('Incident') or '-', 'Date': m.get('Date') or (str(a.get('record_start'))[:19] if a.get('record_start') else '-'),
            'Station': m.get('Station') or '-', 'Line': m.get('Line') or '-', 'Relay': m.get('Relay') or '-',
            'Fault Type': a.get('type', '-') if a.get('fault') else 'None', 'Fault Description': fg.TYPE_TEXT.get(a.get('type'), '-') if a.get('fault') else 'No fault',
            'AI Confidence (%)': round(a.get('type_conf', 0) * 100) if a.get('fault') else None, 'Classic Type': a.get('type_rule', '-'),
            'Location': loc, 'Location (km)': f1(a.get('loc_km')), 'Location (%)': round(a['loc_pu'] * 100) if a.get('loc_pu') is not None else None,
            'Fault Current (kA)': f1(a.get('I_fault_max_kA')), 'Lowest Voltage (%)': round(a['V_min_pu'] * 100) if a.get('V_min_pu') is not None else None,
            'Protection': ', '.join(e[1] for e in a.get('elements', [])) or '-', 'Operate (ms)': f1(a.get('operate_ms')),
            'Breaker (ms)': f1(a.get('breaker_ms')), 'Clearing (ms)': f1(a.get('clearing_ms')), 'Auto-Reclose': a.get('ar', '-'),
            'Governing Aspect': a['Governing Mode'], 'Overall Risk (%)': a['Overall Risk (%)'], 'Status': a['Status'], 'Likely Test': bool(a.get('likely_test')),
            'S': a['S'], 'O': a['O'], 'D': a['D'], 'RPN': a['RPN'], 'RPN Level': a['RPN Level'], 'Action': a['Action'],
            'Mode Risks': a['Mode Risks'], 'FMEA': a['FMEA'], 'Findings': fg.findings(a), 'Summary': fg.summary_text(a),
            'Source': source, 'Analyzed': datetime.now().strftime('%Y-%m-%d %H:%M')}


@st.cache_data(show_spinner=False)
def demo_files(i):
    return fg.demo_incident(i)


@st.cache_data(show_spinner="Analyzing the demo incidents…")
def demo_snapshot():
    out = []
    for i in range(len(fg.DEMO_INCIDENTS)):
        meta, cfg, dat, E, line = demo_files(i)
        a = fg.analyze(fg.read_comtrade(cfg, dat), E, line, model=MODEL(), meta=meta)
        out.append(to_record(a, 'Demo substation'))
    return out


def current_records():
    if st.session_state.get('ov_src', SRC_PAGES) == SRC_DEMO:
        return demo_snapshot()
    return list(st.session_state['register'])


# ==========================================
# --- 4. Charts ---
# ==========================================
def gauge(value, color, title='Incident Risk (%)', height=290):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), number={'suffix': '%', 'font': {'size': 34}, 'valueformat': '.0f'},
        title={'text': title, 'font': {'size': 15}}, domain={'x': [0, 1], 'y': [0, 0.85]},
        gauge={'axis': {'range': [0, 100], 'tickvals': [0, 35, 70, 100], 'tickfont': {'size': 12}},
               'bar': {'color': color, 'thickness': 0.3},
               'steps': [{'range': [0, 35], 'color': 'rgba(26,158,92,.25)'},
                         {'range': [35, 70], 'color': 'rgba(217,154,0,.25)'},
                         {'range': [70, 100], 'color': 'rgba(214,58,58,.25)'}]}))
    fig.update_layout(height=height, margin=dict(l=35, r=35, t=40, b=10))
    return fig


def band_color(v):
    return GREEN if v <= fg.EXCELLENT_MAX else AMBER if v <= fg.WATCH_MAX else RED


def rpn_color(r):
    return GREEN if r < 40 else AMBER if r < 100 else RED


def mode_plot(risks, title='Risk by aspect', height=280):
    names = sorted(risks, key=lambda k: risks[k])
    vals = [risks[n] for n in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation='h', marker_color=[band_color(v) for v in vals],
                           text=[f"{v:.0f}%" for v in vals], textposition='outside', cliponaxis=False,
                           hovertemplate='%{y}: %{x:.1f}%<extra></extra>'))
    for x in (35, 70):
        fig.add_vline(x=x, line_dash='dash', line_color='#888')
    fig.update_layout(title=title, height=height, xaxis=dict(range=[0, 115], title='Risk (%)'), margin=dict(l=10, r=20, t=45, b=40))
    return fig


def fmea_rpn_plot(fmea, height=250):
    t = sorted(fmea, key=lambda m: m['RPN'])
    fig = go.Figure(go.Bar(x=[m['RPN'] for m in t], y=[m['Failure Mode'] for m in t], orientation='h',
                           marker_color=[rpn_color(m['RPN']) for m in t], text=[str(m['RPN']) for m in t],
                           textposition='outside', cliponaxis=False, hovertemplate='%{y}: RPN %{x}<extra></extra>'))
    for x in (40, 100):
        fig.add_vline(x=x, line_dash='dash', line_color='#888')
    fig.update_layout(title='RPN by aspect', height=height, xaxis=dict(range=[0, 200], title='RPN'), margin=dict(l=10, r=20, t=45, b=40))
    return fig


def wave_plot(rec, a, height=620):
    A = a.get('A')
    if A is None:
        A = fg.scale_primary(rec)
    ch = a['channels']
    t = (rec['t'] - rec['t'][0]) * 1000
    step = max(1, len(t) // 6000)
    sl = slice(None, None, step)
    digs = [(tm, nm) for tm, nm, v in a.get('binary', []) if v == 1]
    rows = 3 if len(rec['digital']) else 2
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.38, 0.38, 0.24][:rows], subplot_titles=['Currents (kA, primary)', 'Voltages (kV, primary)', 'Binary signals'][:rows])
    for p in 'ABC':
        if ch.get('I' + p) is not None:
            fig.add_scatter(x=t[sl], y=A[sl, ch['I' + p]] / 1000, mode='lines', name='I' + p, line=dict(color=PH_COL[p], width=1), row=1, col=1)
        if ch.get('V' + p) is not None:
            fig.add_scatter(x=t[sl], y=A[sl, ch['V' + p]] / 1000, mode='lines', name='V' + p, line=dict(color=PH_COL[p], width=1, dash='solid'),
                            row=2, col=1, showlegend=True)
    if ch.get('IN') is not None:
        fig.add_scatter(x=t[sl], y=A[sl, ch['IN']] / 1000, mode='lines', name='IN', line=dict(color=PH_COL['N'], width=1), row=1, col=1)
    if rows == 3:
        D = rec['D']
        act = [j for j in range(D.shape[1]) if D[:, j].any()][:14]
        for k, j in enumerate(act):
            fig.add_scatter(x=t[sl], y=D[sl, j] * 0.7 + k, mode='lines', name=rec['digital'][j].strip(), line=dict(width=1.4), showlegend=False,
                            hovertemplate=rec['digital'][j].strip() + ': %{x:.1f} ms<extra></extra>', row=3, col=1)
        fig.update_yaxes(tickvals=list(range(len(act))), ticktext=[rec['digital'][j].strip()[:18] for j in act], row=3, col=1)
    for key, col, lab in [('t_inception_ms', AMBER, 'Fault'), ('t_trip_ms', RED, 'Trip'), ('t_interruption_ms', GREEN, 'Cleared'),
                          ('t_reclose_ms', NAVY, 'Reclose'), ('t_refault_ms', RED, 'Re-fault')]:
        if a.get(key) is not None:
            fig.add_vline(x=a[key], line_dash='dash', line_color=col, annotation_text=lab, annotation_position='top')
    fig.update_layout(height=height + (180 if rows == 3 else 0), legend=dict(orientation='h', y=-0.06), margin=dict(l=50, r=20, t=40, b=40),
                      hovermode='x unified')
    fig.update_xaxes(title='ms from record start', row=rows, col=1)
    return fig


def rx_plot(a, height=480):
    line, stt = a['line'], a['setting']
    Z = complex(line['r1'], line['x1']) * line['length_km']
    th = np.linspace(0, 2 * np.pi, 200)
    fig = go.Figure()
    for reach, col, lab in [(stt['z1_reach'], RED, 'Zone 1'), (stt['z2_reach'], AMBER, 'Zone 2')]:
        c = Z * reach / 2
        fig.add_scatter(x=c.real + abs(c) * np.cos(th), y=c.imag + abs(c) * np.sin(th), mode='lines', name=f"{lab} ({reach * 100:.0f}%)",
                        line=dict(color=col, width=1.5), hoverinfo='skip')
    fig.add_scatter(x=[0, Z.real], y=[0, Z.imag], mode='lines', name='Line impedance', line=dict(color='#555', width=2.5))
    z = a.get('traj')
    if z is not None and len(z):
        z = z[np.isfinite(z)]
        z = z[np.abs(z) < 5 * abs(Z)]
        fig.add_scatter(x=z.real, y=z.imag, mode='markers+lines', name='Apparent impedance', marker=dict(size=4, color=NAVY), line=dict(width=0.6, color=NAVY),
                        hovertemplate='R %{x:.1f} Ω, X %{y:.1f} Ω<extra></extra>')
    if a.get('Zapp') is not None:
        fig.add_scatter(x=[a['Zapp'].real], y=[a['Zapp'].imag], mode='markers', name='At the fault', marker=dict(size=13, color=RED, symbol='x'))
    lim = 1.5 * abs(Z) * stt['z2_reach']
    fig.update_layout(title='Impedance seen by the relay (R-X, primary Ω)', height=height, xaxis=dict(range=[-0.5 * lim, lim], title='R (Ω)'),
                      yaxis=dict(range=[-0.3 * lim, 1.1 * lim], title='X (Ω)', scaleanchor='x'), legend=dict(orientation='h', y=-0.18),
                      margin=dict(l=50, r=10, t=45, b=40))
    return fig


def prob_plot(a, height=260):
    p = sorted(a['type_probs'].items(), key=lambda kv: kv[1])[-5:]
    fig = go.Figure(go.Bar(x=[v * 100 for _, v in p], y=[k for k, _ in p], orientation='h',
                           marker_color=[NAVY if k == a['type'] else '#9fb3c8' for k, _ in p], text=[f"{v * 100:.0f}%" for _, v in p],
                           textposition='outside', cliponaxis=False))
    fig.update_layout(title='AI fault-type probability', height=height, xaxis=dict(range=[0, 115], title='%'), yaxis=dict(type='category'),
                      margin=dict(l=10, r=20, t=45, b=40))
    return fig


def fmea_table(a):
    return pd.DataFrame([{'Aspect': m['Failure Mode'], 'Risk (%)': round(m['Risk (%)']), 'S': m['S'], 'O': m['O'], 'D': m['D'],
                          'RPN': m['RPN'], 'RPN Level': m['RPN Level']} for m in a['FMEA']])


def measured_table(a):
    rows = []
    for k in ('IA', 'IB', 'IC', 'IN', 'VA', 'VB', 'VC'):
        if k in a.get('P', {}):
            pre, flt = a['P'][k]; u = 'kA' if k.startswith('I') else 'kV'
            rows.append({'Channel': k, f'Pre-fault ({u})': round(abs(pre) / 1000, 3), f'Fault ({u})': round(abs(flt) / 1000, 3),
                         'Fault angle (°)': round(float(np.degrees(np.angle(flt))))})
    return pd.DataFrame(rows)


def compare_table(a):
    return fg.compare_table(a)


# ==========================================
# --- 5. Incident Analysis page ---
# ==========================================
LINE_KEYS = [('length_km', 'Line length (km)', 0.1), ('r1', 'R1 (Ω/km)', 0.001), ('x1', 'X1 (Ω/km)', 0.001),
             ('r0', 'R0 (Ω/km)', 0.001), ('x0', 'X0 (Ω/km)', 0.001), ('kv', 'Voltage (kV)', 1.0)]
SET_DEF = {'expected_inst_ms': 40.0, 'cb_rated_ms': 50.0, 'critical_ms': 120.0, 'cb_rating_ka': 50.0, 'z1_reach': 80.0, 'z2_reach': 120.0, 'z2_time_ms': 400.0}


def init_fp():
    for k, _, _ in LINE_KEYS:
        st.session_state.setdefault('ln_' + k, float(fg.DEFAULT_LINE[k]))
    for k, v in SET_DEF.items():
        st.session_state.setdefault('st_' + k, v)
    for k in ('inc_id', 'inc_station', 'inc_line', 'inc_relay'):
        st.session_state.setdefault(k, '')
    if st.session_state.get('fp_init') is None:
        st.session_state['fp_init'] = True
        fill_demo()


def demo_label(i):
    d = fg.DEMO_INCIDENTS[i]
    return f"{d[0]} · {d[1]} · {d[4]} · {d[12][:10]}"


def fill_demo():
    i = st.session_state.get('demo_i', 0)
    meta, cfg, dat, E, line = demo_files(i)
    for k, _, _ in LINE_KEYS:
        st.session_state['ln_' + k] = float(line[k])
    st.session_state['inc_id'] = meta['Incident']; st.session_state['inc_station'] = meta['Station']
    st.session_state['inc_line'] = meta['Line']; st.session_state['inc_relay'] = meta['Relay']


def src_changed():
    if st.session_state.get('fp_src') == "Upload relay files":
        for k in ('inc_id', 'inc_station', 'inc_line', 'inc_relay'):
            st.session_state[k] = ''
    else:
        fill_demo()


def read_records(files):
    """All disturbance records in the uploads: COMTRADE pairs (.cfg + .dat with the same name, also inside .zip files) and CSV records."""
    by = {}
    for f in files:
        name = f.name; data = f.getvalue()
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for n in z.namelist():
                    if not n.endswith('/'):
                        by[n.rsplit('/', 1)[-1]] = z.read(n)
        else:
            by[name] = data
    low = {n.lower(): n for n in by}
    out, missing = [], []
    for n in by:
        if n.lower().endswith('.cfg'):
            stem = n[:-4].lower()
            dn = low.get(stem + '.dat')
            if dn is None:
                dats = [x for x in by if x.lower().endswith('.dat')]
                dn = dats[0] if len(dats) == 1 and sum(x.lower().endswith('.cfg') for x in by) == 1 else None
            if dn is None:
                missing.append(n); continue
            out.append((n, fg.read_comtrade(by[n].decode('latin-1'), by[dn])))
        elif n.lower().endswith('.csv'):
            out.append((n, fg.read_record_csv(by[n], st.session_state.get('csv_freq', 50.0))))
    return out, missing


def read_docs(files, kind):
    """Event lists or settings files in any format (PDF, scanned PDF, picture, CSV, TXT, Excel)."""
    res, bad = [], []
    for f in files or []:
        try:
            if kind == 'events':
                df = fg.parse_events(f.getvalue(), f.name)
                if df is not None and len(df):
                    res.append((f.name, df))
                else:
                    bad.append(f.name)
            else:
                txt, how = fg.extract_text(f.getvalue(), f.name)
                sv = fg.parse_settings(txt); sv['_read_as'] = how
                if [k for k in sv if not k.startswith('_') and k != 'device']:
                    res.append((f.name, sv))
                else:
                    bad.append(f.name)
        except Exception as e:
            bad.append(f"{f.name} ({e})")
    return res, bad


def analysis_page():
    init_fp()
    st.markdown("<div class='pg-note'>Upload the files of every relay involved: <b>disturbance records</b> (COMTRADE .cfg + .dat, any number of relays), "
                "<b>event lists</b> and <b>settings</b> (PDF, scanned PDF, picture, CSV, TXT or Excel, any number). FaultGuard merges them into one trip "
                "analysis. Without a settings file, enter the line data below. Or pick a demo incident.</div>", unsafe_allow_html=True)
    src = st.radio("Data source", ["Demo incident", "Upload relay files"], horizontal=True, key='fp_src', on_change=src_changed)
    events, rec, rec_name, ev_name, settings = None, None, None, None, None
    if src == "Demo incident":
        c1, c2 = st.columns([1.3, 1])
        c1.selectbox("Demo incident", range(len(fg.DEMO_INCIDENTS)), format_func=demo_label, key='demo_i', on_change=fill_demo)
        i = st.session_state['demo_i']
        meta, cfg, dat, E, line = demo_files(i)
        with c2:
            st.caption("Download the demo files to try the upload path:")
            d1, d2, d3 = st.columns(3)
            stem = fg.DEMO_INCIDENTS[i][0]
            d1.download_button("📥 .cfg", cfg.encode(), f"{stem}.cfg", key='dl_cfg')
            d2.download_button("📥 .dat", dat, f"{stem}.dat", key='dl_dat')
            d3.download_button("📥 events.txt", fg.events_to_text(E).encode(), f"{stem}_events.txt", key='dl_ev')
    else:
        DOC = ['pdf', 'csv', 'txt', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp']
        DOC = DOC + [x.upper() for x in DOC]
        c1, c2, c3 = st.columns(3)
        up_rec = c1.file_uploader("1. Disturbance records (COMTRADE .cfg + .dat, .zip, or CSV), any number of relays",
                                  type=['cfg', 'dat', 'zip', 'csv', 'CFG', 'DAT', 'ZIP', 'CSV'], accept_multiple_files=True, key='up_rec')
        up_ev = c2.file_uploader("2. Event lists (PDF, picture, CSV, TXT, Excel), any number", type=DOC, accept_multiple_files=True, key='up_ev')
        up_set = c3.file_uploader("3. Relay settings (PDF, picture, CSV, TXT, Excel), any number", type=DOC, accept_multiple_files=True, key='up_set')
        c1.number_input("Frequency for CSV records (Hz)", 45.0, 65.0, 50.0, 10.0, key='csv_freq')
        c3.checkbox("Use the settings files for the line data and protection settings", True, key='use_set')
    st.markdown("##### Incident")
    c = st.columns(4)
    c[0].text_input("Incident ID", key='inc_id'); c[1].text_input("Substation", key='inc_station')
    c[2].text_input("Line / bay", key='inc_line'); c[3].text_input("Relay", key='inc_relay')
    st.markdown("##### Line data")
    mode = st.radio("Enter the line data as", ["Per-km sequence impedances", "Relay settings (line impedance, angle, kZN)"], horizontal=True, key='ln_mode')
    if mode.startswith('Per-km'):
        c = st.columns(6)
        for col, (k, lab, step) in zip(c, LINE_KEYS):
            col.number_input(lab, min_value=0.0, step=step, format="%.3f" if step < 0.01 else "%.1f", key='ln_' + k)
    else:
        c = st.columns(6)
        c[0].number_input("Line length (km)", 0.1, 2000.0, step=0.1, key='ln_length_km')
        c[1].number_input("Line impedance |Z1| (Ω, whole line)", 0.01, 5000.0, 38.0, step=0.1, key='rs_z1')
        c[2].number_input("Line angle (°)", 30.0, 90.0, 85.0, step=0.5, key='rs_ang')
        c[3].number_input("kZN residual comp.", 0.0, 5.0, 0.7, step=0.01, key='rs_kzn')
        c[4].number_input("kZN angle (°)", -90.0, 90.0, 0.0, step=0.5, key='rs_kang')
        c[5].number_input("Voltage (kV)", 1.0, 1200.0, step=1.0, key='ln_kv')
        sec = st.checkbox("Line impedance is in secondary Ω (convert with the CT and VT ratios under settings)", key='rs_sec')
        z1 = st.session_state['rs_z1'] * ((st.session_state.get('vt', 3636.4) / st.session_state.get('ct', 2000.0)) if sec else 1.0) / max(st.session_state['ln_length_km'], 0.1)
        z1c = z1 * np.exp(1j * np.radians(st.session_state['rs_ang']))
        z0c = z1c * (1 + 3 * st.session_state['rs_kzn'] * np.exp(1j * np.radians(st.session_state['rs_kang'])))
        st.session_state['ln_r1'], st.session_state['ln_x1'] = float(z1c.real), float(z1c.imag)
        st.session_state['ln_r0'], st.session_state['ln_x0'] = float(z0c.real), float(z0c.imag)
        st.caption(f"Per km: Z1 = {z1c.real:.4f} + j{z1c.imag:.4f} Ω, Z0 = {z0c.real:.4f} + j{z0c.imag:.4f} Ω (Z0 = Z1 · (1 + 3·kZN)).")
    with st.expander("⚙️ Protection settings, breaker data and scaling"):
        c = st.columns(4)
        c[0].number_input("Zone 1 reach (% of line)", 10.0, 120.0, step=5.0, key='st_z1_reach')
        c[1].number_input("Zone 2 reach (% of line)", 50.0, 250.0, step=5.0, key='st_z2_reach')
        c[2].number_input("Zone 2 time (ms)", 100.0, 1500.0, step=50.0, key='st_z2_time_ms')
        c[3].number_input("Expected instantaneous operate (ms)", 10.0, 100.0, step=5.0, key='st_expected_inst_ms')
        c = st.columns(4)
        c[0].number_input("Breaker rated break time (ms)", 20.0, 150.0, step=5.0, key='st_cb_rated_ms')
        c[1].number_input("Critical clearing time (ms)", 60.0, 500.0, step=10.0, key='st_critical_ms')
        c[2].number_input("Breaker rating (kA)", 10.0, 80.0, step=1.0, key='st_cb_rating_ka')
        c[3].checkbox("Auto-reclose in service (normally off: external recloser)", False, key='st_ar')
        c = st.columns(3)
        c[0].selectbox("Record values are", ['Auto', 'Primary', 'Secondary'], key='scale',
                       help="Auto: use the COMTRADE primary/secondary flag and ratios. Secondary: multiply by the CT and VT ratios below.")
        c[1].number_input("CT ratio (e.g. 2000/1 → 2000)", 1.0, 100000.0, 2000.0, key='ct')
        c[2].number_input("VT ratio (e.g. 400kV/110V → 3636)", 1.0, 100000.0, 3636.4, key='vt')
    if st.button("🔍 Analyze Incident", type='primary', width=STRETCH):
        try:
            if src == "Demo incident":
                rec = fg.read_comtrade(cfg, dat); events = E; rec_name = f"{stem} (demo)"
            else:
                if not up_rec:
                    st.error("Upload at least one disturbance record."); return
                with st.spinner("Reading the files (pictures and scanned PDFs are read by OCR)…"):
                    recs, missing = read_records(up_rec)
                    for n in missing:
                        st.warning(f"{n}: its .dat file is missing, so this record is left out.")
                    if not recs:
                        st.error("No complete disturbance record: upload each .cfg with its .dat."); return
                    events, bad = read_docs(up_ev, 'events')
                    for n in bad:
                        st.warning(f"No time-stamped events could be read from {n}: it is left out.")
                    settings, bad = read_docs(up_set, 'settings')
                    for n in bad:
                        st.warning(f"No settings could be read from {n}: it is left out.")
                rec_name = ', '.join(n for n, _ in recs); ev_name = ', '.join(n for n, _ in events)
                events = events or None; rec = recs[0][1]
            ss = st.session_state
            line = {k: float(ss['ln_' + k]) for k, _, _ in LINE_KEYS}
            setting = {k: float(ss['st_' + k]) for k in SET_DEF}
            setting['z1_reach'] /= 100; setting['z2_reach'] /= 100; setting['ar_expected'] = bool(ss['st_ar'])
            meta = {'Incident': ss['inc_id'] or rec_name, 'Station': ss['inc_station'] or (rec.get('station') or '').title(),
                    'Line': ss['inc_line'] or (rec.get('device') or '').title(), 'Relay': ss['inc_relay'] or (rec.get('device') or '').title(),
                    'Date': str(rec.get('start'))[:19] if rec.get('start') else ''}
            if src == "Demo incident":
                a = fg.analyze(rec, events, line, setting, model=MODEL(), scale=ss['scale'], ct=ss['ct'], vt=ss['vt'], meta=meta)
            else:
                a, rec = fg.analyze_incident(recs, events, settings, line, setting, model=MODEL(), scale=ss['scale'], ct=ss['ct'], vt=ss['vt'],
                                             meta=meta, use_settings=ss.get('use_set', True))
                if not ss['inc_station']:
                    a['meta']['Station'] = (rec.get('station') or '').title()
                if not ss['inc_relay'] and a.get('relay_reports'):
                    a['meta']['Relay'] = ' / '.join(a['relay_reports'])
        except Exception as e:
            st.error(f"Could not analyze the files: {e}")
            return
        a['files'] = (rec_name, ev_name)
        st.session_state['last'] = (a, rec)
        r = to_record(a, 'Demo' if src == "Demo incident" else 'Uploaded files')
        st.session_state['register'] = [x for x in st.session_state['register'] if x['Incident'] != r['Incident']] + [r]
    if st.session_state['last'] is not None:
        show_incident(*st.session_state['last'])


def show_incident(a, rec):
    sk = status_key(a)
    col = STATUS_COLOR[sk]
    st.divider()
    m = a.get('meta', {})
    st.markdown(f"### {m.get('Incident', '')} · {m.get('Line', '')} · {m.get('Date', '')}")
    if a.get('likely_test'):
        banner(STATUS_LABEL['TEST'], fg.summary_text(a), col)
    else:
        banner(f"{STATUS_LABEL[sk]} · incident risk {a['Overall Risk (%)']:.0f}%", fg.summary_text(a), col)
    st.caption(f"Record: {rec['format']}, {len(rec['analog'])} analog / {len(rec['digital'])} binary channels, "
               f"{(rec['t'][-1] - rec['t'][0]) * 1000:.0f} ms, {1 / np.median(np.diff(rec['t'])):.0f} samples/s"
               + (f" · event list: {len(a['events'])} events" if a.get('events') is not None else " · no event list"))
    for n in a.get('notes', []):
        st.info(n)
    if len(a.get('records') or []) > 1:
        st.markdown("#### Disturbance records of this incident")
        st.dataframe(pd.DataFrame(a['records']), hide_index=True, width=STRETCH)
        st.caption(f"The report is built on {a.get('record_name')} (the most complete record); the others are compared above.")
    if a.get('settings_all'):
        with st.expander(f"⚙️ Settings read from {len(a['settings_all'])} file(s)", expanded=False):
            st.dataframe(fg.settings_table(a['settings_all']), hide_index=True, width=STRETCH)
            st.caption('Read as: ' + ', '.join(f"{n}: {sv.get('_read_as', '')}" for n, sv in a['settings_all']))
    if a.get('restored'):
        rs = a['restored']
        st.caption(f"Line re-energised at {pd.Timestamp(rs['time']):%Y-%m-%d %H:%M:%S}, {fg.fmt_dur(rs['after_s'])} after the fault ({rs['device']}: {rs['event']}).")
    if not a.get('fault'):
        return
    k = st.columns(6)
    k[0].metric("Fault type", a['type'], f"AI {a['type_conf'] * 100:.0f}%", delta_arrow='off', delta_color='off')
    k[1].metric("Location", f"{a['loc_km']:.1f} km" if a.get('loc_km') is not None else '-',
                f"{a['loc_pu'] * 100:.0f}% of line" if a.get('loc_pu') is not None else None, delta_arrow='off', delta_color='off')
    k[2].metric("Fault current", f"{a['I_fault_max_kA']:.2f} kA", f"peak {a['I_peak_kA']:.1f} kA", delta_arrow='off', delta_color='off')
    f0 = lambda v: '-' if v is None else f"{v:.0f} ms"
    k[3].metric("Relay operate", f0(a.get('operate_ms')), ', '.join(e[1] for e in a['elements'])[:28] or None, delta_arrow='off', delta_color='off')
    k[4].metric("Breaker", f0(a.get('breaker_ms')), f"clearing {f0(a.get('clearing_ms'))}", delta_arrow='off', delta_color='off')
    ar0 = a['ar'].split(';')[0].split(' (')[0]
    k[5].metric("Auto-reclose", ar0.split(',')[0], (a['ar'].split('; ')[1] if '; ' in a['ar'] else None), delta_arrow='off', delta_color='off')
    c1, c2, c3 = st.columns([0.9, 1.1, 1])
    c1.plotly_chart(gauge(a['Overall Risk (%)'], col), width=STRETCH, key='fp_g')
    c2.plotly_chart(mode_plot(a['Mode Risks']), width=STRETCH, key='fp_m')
    c3.plotly_chart(prob_plot(a), width=STRETCH, key='fp_p')
    st.plotly_chart(wave_plot(rec, a), width=STRETCH, key='fp_w')
    c1, c2 = st.columns([1, 1.1])
    c1.plotly_chart(rx_plot(a), width=STRETCH, key='fp_rx')
    with c2:
        st.markdown("#### FaultGuard AI vs the relays' own fault records")
        st.dataframe(compare_table(a), hide_index=True, width=STRETCH)
        st.markdown("#### Measured values (primary)")
        st.dataframe(measured_table(a), hide_index=True, width=STRETCH)
    st.markdown("#### Assessment")
    for t in fg.findings(a):
        st.markdown(f"- {t}")
    (st.info if a.get('likely_test') else st.warning if a['Status'] != 'EXCELLENT' else st.success)(f"**Action:** {a['Action']}")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("#### FMEA")
        st.dataframe(fmea_table(a), hide_index=True, width=STRETCH)
    c2.plotly_chart(fmea_rpn_plot(a['FMEA']), width=STRETCH, key='fp_rpn')
    st.markdown("#### Sequence of events (record and event list)")
    so = fg.soe(a)
    st.dataframe(so, hide_index=True, width=STRETCH, height=min(420, 38 + 35 * len(so)))
    stem = str(m.get('Incident') or 'incident').replace(' ', '_').replace('/', '-')
    d = st.columns(3)
    d[0].download_button("📄 Download Incident Report (PDF)", pdfrep.incident_pdf(a, rec), f"FaultGuard_{stem}.pdf", 'application/pdf', key='fp_pdf')
    d[1].download_button("📥 Sequence of events (CSV)", so.to_csv(index=False).encode('utf-8-sig'), f"FaultGuard_{stem}_SOE.csv", 'text/csv', key='fp_soe')
    summ = pd.DataFrame([{k: v for k, v in to_record(a, '').items() if k not in ('Mode Risks', 'FMEA', 'Findings', 'Source', 'Analyzed')}])
    d[2].download_button("📥 Incident summary (CSV)", summ.to_csv(index=False).encode('utf-8-sig'), f"FaultGuard_{stem}_summary.csv", 'text/csv', key='fp_sum')


# ==========================================
# --- 6. AI model page ---
# ==========================================
def model_page():
    b = MODEL()
    st.markdown("<div class='pg-note'>The fault type is given by a <b>Random Forest</b> trained on simulated incidents through the same pipeline as a real "
                "record (sequence-network faults on random 132–500 kV lines, both sources, fault resistance up to 150 Ω, weak infeed, harmonics, "
                "CT saturation and noise). It is checked against the classic phase-selection rules. The fault location uses the Takagi method.</div>",
                unsafe_allow_html=True)
    k = st.columns(5)
    k[0].metric("AI accuracy (unseen test set)", f"{b['accuracy'] * 100:.1f}%")
    k[1].metric("Classic phase selection", f"{b['rule_accuracy'] * 100:.1f}%")
    if b.get('acc_high_rf') is not None:
        k[2].metric("AI, high fault resistance (>60 Ω)", f"{b['acc_high_rf'] * 100:.0f}%", f"{b['n_high_rf']} cases", delta_arrow='off', delta_color='off')
    if b.get('acc_ct_sat') is not None:
        k[3].metric("AI, CT saturation", f"{b['acc_ct_sat'] * 100:.0f}%", f"{b['n_ct_sat']} cases", delta_arrow='off', delta_color='off')
    k[4].metric("Location error (median)", f"{b['loc_err_all']:.1f}% of line", f"low Rf {b['loc_err_lowrf']:.1f}%", delta_arrow='off', delta_color='off')
    st.caption(f"Trained on {b['n_train']} simulated incidents, tested on {b['n_test']} others never seen in training.")
    c1, c2 = st.columns([1.2, 1])
    cm = np.array(b['confusion'])
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    fig = go.Figure(go.Heatmap(z=pct, x=b['labels'], y=b['labels'], colorscale='Blues', zmin=0, zmax=100, text=cm, texttemplate='%{text}',
                               hovertemplate='True %{y} → AI %{x}: %{text}<extra></extra>', colorbar=dict(title='%')))
    fig.update_layout(title='Confusion matrix (rows: true type, columns: AI type)', height=480, xaxis=dict(type='category', title='AI'),
                      yaxis=dict(type='category', autorange='reversed', title='True'), margin=dict(l=50, r=10, t=50, b=40))
    c1.plotly_chart(fig, width=STRETCH, key='m_cm')
    le = b['loc_err']
    fig = go.Figure(go.Bar(x=list(le), y=list(le.values()), marker_color=NAVY, text=[f"{v:.1f}%" for v in le.values()], textposition='outside'))
    fig.update_layout(title='Fault location: median error by fault type (% of line)', height=480, yaxis=dict(title='% of line'),
                      xaxis=dict(type='category'), margin=dict(l=50, r=10, t=50, b=40))
    c2.plotly_chart(fig, width=STRETCH, key='m_le')
    imp = getattr(b['model'], 'feature_importances_', None)
    if imp is not None:
        o = np.argsort(imp)[-12:]
        fig = go.Figure(go.Bar(x=imp[o] * 100, y=[b['features'][i] for i in o], orientation='h', marker_color=BLUE))
        fig.update_layout(title='What the AI looks at (top features, % importance)', height=380, xaxis=dict(title='%'), margin=dict(l=10, r=10, t=45, b=40))
        st.plotly_chart(fig, width=STRETCH, key='m_imp')
    st.markdown("""
**How an incident is analyzed**
1. The record is scaled to primary values and the channels are recognised (IA, IB, IC, IN, VA, VB, VC and the binary signals).
2. The fault start is found from the step in the phase currents (superimposed current); the interruption, reclosing and any re-fault from the current envelope.
3. One-cycle Fourier phasors before and during the fault give 23 scale-free features (current steps, sequence ratios, voltage dips) for the AI model.
4. The location uses the Takagi method on the faulted loop with zero-sequence compensation; the apparent impedance is drawn on the R-X diagram with the zones.
5. The relay and breaker times come from the binary channels and the event list; the event list also gives the relay's own fault location and trip time for comparison.
6. Five aspects are scored (protection operation, breaker, severity, permanent fault, record quality) with an FMEA (RPN = S × O × D).
7. Records that look like secondary injection tests (no line voltage, current kept flowing after the trip, unsynchronised clock) are flagged and kept out of the statistics.
""")


# ==========================================
# --- 7. Statistics ---
# ==========================================
def stats_page(R):
    R = R[~R['Likely Test']].copy()
    if R.empty:
        st.info("Only test records so far: no system faults to count.")
        return
    R['When'] = pd.to_datetime(R['Date'], errors='coerce')
    R['Month'] = R['When'].dt.to_period('M').astype(str)
    k = st.columns(5)
    ar = R[R['Auto-Reclose'].str.startswith(('Successful', 'Unsuccessful'))]
    k[0].metric("System faults", len(R))
    k[1].metric("Lines with faults", R['Line'].nunique())
    k[2].metric("Auto-reclose success", f"{(ar['Auto-Reclose'].str.startswith('Successful').mean() * 100):.0f}%" if len(ar) else '-', f"{len(ar)} attempts",
                delta_arrow='off', delta_color='off')
    k[3].metric("Median clearing", f"{R['Clearing (ms)'].median():.0f} ms" if R['Clearing (ms)'].notna().any() else '-')
    k[4].metric("Mal-operations / slow breakers", int(((R['Governing Aspect'].isin(['Protection operation', 'Breaker performance'])) & (R['Status'] == 'ACTION')).sum()))
    c1, c2 = st.columns(2)
    g = R.groupby(['Line', 'Status']).size().unstack(fill_value=0)
    fig = go.Figure()
    for s in ('EXCELLENT', 'WATCH', 'ACTION'):
        if s in g:
            fig.add_bar(x=g.index, y=g[s], name={'EXCELLENT': 'Normal', 'WATCH': 'Watch', 'ACTION': 'Action'}[s], marker_color=STATUS_COLOR[s])
    fig.update_layout(barmode='stack', title='Faults by line', height=340, yaxis=dict(title='Faults', dtick=1), legend=dict(orientation='h', y=-0.25),
                      margin=dict(l=40, r=10, t=50, b=40))
    c1.plotly_chart(fig, width=STRETCH, key='s_line')
    t = R['Fault Type'].value_counts().reindex(fg.FAULT_TYPES).dropna()
    fig = go.Figure(go.Bar(x=t.index, y=t.values, marker_color=NAVY, text=t.values.astype(int), textposition='outside'))
    fig.update_layout(title='Faults by type', height=340, xaxis=dict(type='category'), yaxis=dict(title='Faults', dtick=1), margin=dict(l=40, r=10, t=50, b=40))
    c2.plotly_chart(fig, width=STRETCH, key='s_type')
    c1, c2 = st.columns(2)
    mth = R.groupby('Month').size()
    fig = go.Figure(go.Bar(x=mth.index, y=mth.values, marker_color=BLUE))
    fig.update_layout(title='Faults by month', height=320, xaxis=dict(type='category'), yaxis=dict(title='Faults', dtick=1), margin=dict(l=40, r=10, t=50, b=40))
    c1.plotly_chart(fig, width=STRETCH, key='s_month')
    fig = go.Figure()
    for ln, d in R.groupby('Line'):
        fig.add_scatter(x=d['Location (%)'], y=[ln] * len(d), mode='markers', marker=dict(size=13, color=[STATUS_COLOR[s] for s in d['Status']]),
                        text=d['Incident'] + ' · ' + d['Fault Type'], hovertemplate='%{text}: %{x}%<extra></extra>', showlegend=False)
    fig.add_vline(x=100, line_dash='dash', line_color='#888', annotation_text='Remote end')
    fig.update_layout(title='Where the faults are (% of each line)', height=320, xaxis=dict(range=[0, 140], title='% of line'),
                      yaxis=dict(type='category'), margin=dict(l=10, r=10, t=50, b=40))
    c2.plotly_chart(fig, width=STRETCH, key='s_loc')
    rep = []
    for ln, d in R.dropna(subset=['Location (km)']).groupby('Line'):
        d = d.sort_values('When')
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                if abs(d.iloc[i]['Location (km)'] - d.iloc[j]['Location (km)']) <= 3:
                    rep.append({'Line': ln, 'First': d.iloc[i]['Incident'], 'Second': d.iloc[j]['Incident'],
                                'Locations (km)': f"{d.iloc[i]['Location (km)']:.1f} / {d.iloc[j]['Location (km)']:.1f}"})
    st.markdown("#### Repeated faults at the same place (within 3 km)")
    if rep:
        st.warning("Faults repeating at the same tower span point to a weak spot (insulator, vegetation, conductor clearance): patrol it.")
        st.dataframe(pd.DataFrame(rep), hide_index=True, width=STRETCH)
    else:
        st.success("No repeated faults at the same place.")


# ==========================================
# --- 8. Tabs ---
# ==========================================
tabs = st.tabs(["🏭 Incidents Overview", "⚡ Incident Analysis", "🧠 AI Fault Classifier & Accuracy", "📄 Combined Report", "📈 Incident Statistics"])

OV_COLS = ['Incident', 'Date', 'Line', 'Fault Type', 'Location', 'Fault Current (kA)', 'Protection', 'Operate (ms)', 'Breaker (ms)', 'Clearing (ms)',
           'Auto-Reclose', 'Governing Aspect', 'Overall Risk (%)', 'Status', 'RPN', 'RPN Level', 'Source', 'Action']

with tabs[0]:
    st.radio("Show", [SRC_PAGES, SRC_DEMO], horizontal=True, key='ov_src')
    R0 = current_records()
    if st.session_state['ov_src'] == SRC_PAGES:
        st.markdown("<div class='pg-note'>Every incident analyzed on the <b>Incident Analysis</b> page is collected here (latest per incident ID).</div>",
                    unsafe_allow_html=True)
        if R0 and st.button("🗑 Clear results"):
            st.session_state['register'] = []
            st.rerun()
    else:
        st.info("Synthetic demo substation: 7 incidents on four 400 kV lines (single-phase, phase-phase, three-phase and permanent faults, a slow breaker, "
                "a zone 1 over-reach and a high-resistance fault). Simulated records, not real measurements.")
    if not R0:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:30px'>💡 No results yet: analyze an incident on the Incident Analysis page, "
                    "or show the demo substation.</h3>", unsafe_allow_html=True)
    else:
        R = pd.DataFrame(R0).sort_values('Overall Risk (%)', ascending=False).reset_index(drop=True)
        tests = R['Likely Test']
        Rs = R[~tests]
        k = st.columns(5)
        k[0].metric("Records", len(R)); k[1].metric("🔴 Action required", int((Rs['Status'] == 'ACTION').sum()))
        k[2].metric("🟠 Watch", int((Rs['Status'] == 'WATCH').sum())); k[3].metric("🟢 Normal operation", int((Rs['Status'] == 'EXCELLENT').sum()))
        k[4].metric("🔵 Test records", int(tests.sum()))
        if len(Rs):
            w = Rs.iloc[0]
            perm = Rs[Rs['Auto-Reclose'].str.startswith('Unsuccessful')]
            slow = Rs[Rs['Governing Aspect'] == 'Breaker performance']
            cl = Rs['Clearing (ms)'].dropna()
            cards = [("Highest risk", f"{w['Overall Risk (%)']:.0f}%", STATUS_COLOR[w['Status']], f"{w['Incident']} · {w['Governing Aspect']}"),
                     ("Permanent faults", f"{len(perm)}", RED if len(perm) else GREEN, ', '.join(perm['Line'].head(3)) or 'none'),
                     ("Slowest clearing", f"{cl.max():.0f} ms" if len(cl) else '-', AMBER if len(cl) and cl.max() > 120 else GREEN,
                      Rs.loc[cl.idxmax(), 'Incident'] if len(cl) else ''),
                     ("Highest fault current", f"{Rs['Fault Current (kA)'].max():.1f} kA", NAVY, Rs.loc[Rs['Fault Current (kA)'].idxmax(), 'Incident']),
                     ("Breaker issues", f"{int((slow['Status'] != 'EXCELLENT').sum())}", AMBER if (slow['Status'] != 'EXCELLENT').any() else GREEN,
                      ', '.join(slow[slow['Status'] != 'EXCELLENT']['Incident'].head(3)) or 'none')]
            for col, (t, big, c, sub) in zip(st.columns(5), cards):
                col.markdown(f"<div class='pg-card' style='border-top:5px solid {c}'><h4>{t}</h4><div class='big' style='color:{c}'>{big}</div>"
                             f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)
            st.write("")
        t = R.iloc[::-1]
        colors = [STATUS_COLOR['TEST'] if x else STATUS_COLOR[s] for x, s in zip(t['Likely Test'], t['Status'])]
        fig = go.Figure(go.Bar(x=t['Overall Risk (%)'], y=t['Incident'] + ' · ' + t['Line'], orientation='h', marker_color=colors,
                               text=[('Test record' if x else f"{r:.0f}% · {g}") for x, r, g in zip(t['Likely Test'], t['Overall Risk (%)'], t['Governing Aspect'])],
                               textposition='outside', cliponaxis=False))
        for x, tx in [(35, 'Watch'), (70, 'Action')]:
            fig.add_vline(x=x, line_dash='dash', line_color='#888', annotation_text=tx, annotation_position='top')
        fig.update_layout(title='Incident risk ranking', height=140 + 36 * len(t), xaxis=dict(range=[0, 140], title='Incident Risk (%)'),
                          yaxis=dict(type='category'), margin=dict(l=10, r=30, t=60, b=40))
        c1, c2 = st.columns([1.1, 1])
        c1.plotly_chart(fig, width=STRETCH, key='ov_rank')
        z = [[r['Mode Risks'][m] for m in MODE_ORDER] for _, r in R.iterrows()]
        fig = go.Figure(go.Heatmap(z=z, x=MODE_ORDER, y=list(R['Incident']), zmin=0, zmax=100,
                                   colorscale=[[0, '#e8f5ee'], [0.35, '#bfe3cd'], [0.351, '#fbe8b3'], [0.7, '#f2c14e'], [0.701, '#f2a6a6'], [1, RED]],
                                   text=[[f"{v:.0f}" for v in row] for row in z], texttemplate='%{text}', colorbar=dict(title='%', thickness=12)))
        fig.update_layout(title='Risk by incident and aspect', height=150 + 36 * len(R), yaxis=dict(type='category', autorange='reversed'),
                          margin=dict(l=10, r=10, t=50, b=40))
        c2.plotly_chart(fig, width=STRETCH, key='ov_heat')
        st.markdown("#### All incidents")
        st.dataframe(R[OV_COLS], hide_index=True, width=STRETCH)
        st.download_button("📥 Download incident list (CSV)", R[OV_COLS].to_csv(index=False).encode('utf-8-sig'),
                           f"FaultGuard_incidents_{datetime.now():%Y%m%d}.csv", 'text/csv')

with tabs[1]:
    analysis_page()
with tabs[2]:
    model_page()

with tabs[3]:
    recs = current_records()
    if not recs:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:40px'>💡 Analyze incidents on the Incident Analysis page "
                    "(or show the demo substation on the Overview page) first.</h3>", unsafe_allow_html=True)
    else:
        R = pd.DataFrame(recs).sort_values('Overall Risk (%)', ascending=False).reset_index(drop=True)
        Rs = R[~R['Likely Test']]
        n_a, n_w = int((Rs['Status'] == 'ACTION').sum()), int((Rs['Status'] == 'WATCH').sum())
        n_t = int(R['Likely Test'].sum())
        src = st.session_state.get('ov_src', SRC_PAGES).lower()
        st.markdown(f"<div class='pg-note'><b>📄 Incident Summary Report</b><br>{len(R)} records · source: {src} · "
                    f"generated {datetime.now():%Y-%m-%d %H:%M}</div>", unsafe_allow_html=True)
        if n_a:
            top = Rs.iloc[0]
            summary_txt = (f"{n_a} incident(s) need action and {n_w} need review. Highest: {top['Incident']} on {top['Line']} "
                           f"({top['Governing Aspect']}, {top['Overall Risk (%)']:.0f}%).")
        elif n_w:
            summary_txt = f"No incident needs immediate action; {n_w} need review."
        else:
            summary_txt = "All faults were cleared correctly by the protection and the breakers."
        if n_t:
            summary_txt += f" {n_t} record(s) look like secondary injection tests and are kept out of the fault statistics."
        st.markdown(f"**Executive summary:** {summary_txt}")
        k = st.columns(4)
        k[0].metric("Records", len(R)); k[1].metric("🔴 Action", n_a); k[2].metric("🟠 Watch", n_w); k[3].metric("🟢 Normal", len(Rs) - n_a - n_w)
        cols = ['Incident', 'Date', 'Line', 'Fault Type', 'Location', 'Clearing (ms)', 'Auto-Reclose', 'Governing Aspect', 'Overall Risk (%)', 'Status', 'RPN', 'Action']
        st.dataframe(R[cols], hide_index=True, width=STRETCH)
        st.markdown("#### Incidents needing attention")
        above = Rs[Rs['Status'] != 'EXCELLENT']
        if above.empty:
            st.success("No incident above the normal band.")
        for i, (_, r) in enumerate(above.iterrows()):
            with st.expander(f"{r['Incident']}  ·  {r['Line']}  ·  {r['Governing Aspect']}  ·  {r['Overall Risk (%)']:.0f}% {r['Status']}", expanded=i < 3):
                e1, e2 = st.columns([1, 1.1])
                with e1:
                    st.markdown(f"_{r['Summary']}_")
                    for t in r['Findings']:
                        st.markdown(f"- {t}")
                    st.markdown(f"**RPN** = {r['S']} × {r['O']} × {r['D']} = **{r['RPN']}** ({r['RPN Level']})")
                e2.plotly_chart(mode_plot(r['Mode Risks'], height=260), width=STRETCH, key=f"rep_m_{i}")
                st.warning(f"**Action:** {r['Action']}")
        lines = ["=========================================", "FAULTGUARD AI - INCIDENT SUMMARY REPORT", "=========================================",
                 f"Generated: {datetime.now():%Y-%m-%d %H:%M}", f"Records: {len(R)}  (action {n_a}, watch {n_w}, test {n_t})", "",
                 "EXECUTIVE SUMMARY:", summary_txt, ""]
        for _, r in R.iterrows():
            lines += ["-----------------------------------------", f"{r['Incident']} - {r['Line']} - {r['Date']} - "
                      + ('TEST RECORD' if r['Likely Test'] else f"{r['Status']} - risk {r['Overall Risk (%)']:.0f}%"), r['Summary'],
                      f"Governing aspect: {r['Governing Aspect']}  |  RPN = {r['S']} x {r['O']} x {r['D']} = {r['RPN']} ({r['RPN Level']})"] + \
                     [f"* {t}" for t in r['Findings']] + [f"Action: {r['Action']}"]
        lines += ["=========================================", "Generated by FaultGuard AI Engine"]
        fm = pd.DataFrame([dict(**{'Incident': r['Incident']}, **m) for _, r in R.iterrows() for m in r['FMEA']])
        d = st.columns(4)
        d[0].download_button("📄 Download Report (PDF)", pdfrep.combined_pdf(R, summary_txt, src, n_a, n_w),
                             f"FaultGuard_Report_{datetime.now():%Y%m%d}.pdf", 'application/pdf', key='rep_pdf')
        d[1].download_button("📥 Download Full Text Report", "\n".join(lines).encode('utf-8'), f"FaultGuard_Report_{datetime.now():%Y%m%d}.txt", 'text/plain')
        d[2].download_button("📥 Download Table (CSV)", R[OV_COLS].to_csv(index=False).encode('utf-8-sig'), f"FaultGuard_Report_{datetime.now():%Y%m%d}.csv", 'text/csv')
        d[3].download_button("📥 Download FMEA (CSV)", fm.to_csv(index=False).encode('utf-8-sig'), f"FaultGuard_FMEA_{datetime.now():%Y%m%d}.csv", 'text/csv')

with tabs[4]:
    recs = current_records()
    st.markdown("<div class='pg-note'>Fault statistics over all analyzed incidents (source chosen on the Overview page): by line, type and month, "
                "auto-reclose success and repeated faults at the same place. Test records are excluded.</div>", unsafe_allow_html=True)
    if not recs:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:30px'>💡 No incidents yet.</h3>", unsafe_allow_html=True)
    else:
        stats_page(pd.DataFrame(recs))

st.markdown(FOOTER, unsafe_allow_html=True)
