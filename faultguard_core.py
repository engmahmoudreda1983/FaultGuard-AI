"""
FaultGuard AI - automatic trip and fault analysis from relay disturbance records and event lists.

One engine used by every page of the app:
  * reads a disturbance record (COMTRADE .cfg/.dat or CSV) and a relay event list (CSV, TXT or PDF),
  * finds the fault inception, computes pre-fault and fault phasors (one-cycle Fourier),
  * classifies the fault type with an AI model (Random Forest trained on simulated faults) and checks it with the
    classic superimposed-current method,
  * locates the fault on the line (Takagi method, with the line impedance),
  * measures relay operate time, breaker interruption time, total clearing time and the auto-reclose outcome,
  * finds the protection elements that operated (record binary channels and event list),
  * assesses the incident (protection operation, breaker, fault severity, permanent fault, record quality) into a
    risk with FMEA and actions, and builds the incident report.
"""
import io
import math
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import comtrade_features as cf

EXCELLENT_MAX, WATCH_MAX = 35.0, 70.0
DETECTION = 2
A_OP = np.exp(2j * np.pi / 3)
AMAT = np.array([[1, 1, 1], [1, A_OP ** 2, A_OP], [1, A_OP, A_OP ** 2]])      # phase = AMAT @ (0, 1, 2)
AINV = np.linalg.inv(AMAT)
FAULT_TYPES = ['AG', 'BG', 'CG', 'AB', 'BC', 'CA', 'ABG', 'BCG', 'CAG', 'ABC']
TYPE_TEXT = {'AG': 'Phase A to ground', 'BG': 'Phase B to ground', 'CG': 'Phase C to ground', 'AB': 'Phase A to phase B',
             'BC': 'Phase B to phase C', 'CA': 'Phase C to phase A', 'ABG': 'Phases A-B to ground', 'BCG': 'Phases B-C to ground',
             'CAG': 'Phases C-A to ground', 'ABC': 'Three-phase'}

MODES = {
    'Protection operation': {'S': 9, 'text': 'Protection operated wrongly or too slowly (mal-operation, over- or under-reach)'},
    'Breaker performance': {'S': 9, 'text': 'Breaker slow to interrupt, or interrupting near its rating'},
    'Fault severity': {'S': 7, 'text': 'Heavy fault current, deep voltage dip or long clearing time stressing the system'},
    'Permanent fault': {'S': 6, 'text': 'Fault still present at reclosing: line damage to be found and repaired'},
    'Record quality': {'S': 3, 'text': 'Record incomplete or doubtful (clock, channels, CT circuit)'},
}
MODE_ORDER = list(MODES)
ACTIONS = {
    'Protection operation': 'Review the relay settings and the fault record with the protection engineer; check the zone reaches, the aided scheme and the CT/VT circuits; test the relay before returning it to normal service.',
    'Breaker performance': 'Schedule a breaker timing test and mechanism inspection (trip coils, latch, operating energy); check the interrupted current against the rating.',
    'Fault severity': 'Report the fault to the system operator; check the equipment that carried the fault current (CTs, line traps, breaker); confirm the clearing time against the grid code.',
    'Permanent fault': 'Arrange a line patrol at the fault location; do not re-energise until the cause is found; check insulators, conductors and towers near the located distance.',
    'Record quality': 'Synchronise the relay clock, check the record channels and the CT/VT wiring so future records are complete.',
}

# ---------------------------------------------------------------- generic helpers
def lin(x, x0, x1, y0=0.0, y1=100.0):
    if x1 == x0:
        return y1 if x >= x1 else y0
    return float(min(max(y0, y1), max(min(y0, y1), y0 + (y1 - y0) * (x - x0) / (x1 - x0))))


def status_of(r):
    return 'EXCELLENT' if r <= EXCELLENT_MAX else ('WATCH' if r <= WATCH_MAX else 'ACTION')


def rpn_level(rpn):
    return 'Low' if rpn < 40 else ('Medium' if rpn < 100 else 'High')


# ---------------------------------------------------------------- reading records
def _ratios(cfg_text, n_analog):
    """Primary/secondary ratio and P/S flag of each analog channel (COMTRADE 1999 columns 11-13)."""
    L = cfg_text.splitlines()
    out = []
    for i in range(n_analog):
        f = L[2 + i].split(',')
        try:
            pr, se, ps = float(f[10]), float(f[11]), f[12].strip().upper()[:1]
        except Exception:
            pr, se, ps = 1.0, 1.0, 'P'
        out.append((pr, se, ps))
    return out


def read_comtrade(cfg_text, dat_bytes):
    cfg = cf.read_cfg(text=cfg_text)
    t, A, D = cf.read_dat(None, cfg, dat_bytes)
    return {'cfg': cfg, 't': np.asarray(t, float), 'A': np.asarray(A, float), 'D': np.asarray(D, int),
            'analog': [c['name'] for c in cfg['analog']], 'units': [c['unit'] for c in cfg['analog']],
            'digital': [d['name'] for d in cfg['digital']], 'freq': float(cfg['freq'] or 50.0),
            'start': cfg['t_first'], 'trigger': cfg['t_trigger'], 'station': cfg['station'], 'device': cfg['device'],
            'ratios': _ratios(cfg_text, len(cfg['analog'])), 'format': 'COMTRADE'}


def read_record_csv(data, freq=50.0):
    """CSV record: a time column (s or ms) then analog columns (IA, IB, IC, IN, VA, VB, VC ...) and 0/1 binary columns."""
    df = pd.read_csv(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    tcol = next((c for c in df.columns if re.search(r'^t$|time|sec|ms', str(c).lower())), df.columns[0])
    t = pd.to_numeric(df[tcol], errors='coerce').to_numpy(float)
    if np.nanmax(np.abs(t)) > 50:                     # milliseconds
        t = t / 1000.0
    an, dg = [], []
    for c in df.columns:
        if c == tcol:
            continue
        v = pd.to_numeric(df[c], errors='coerce')
        if v.notna().mean() < 0.9:
            continue
        uniq = set(np.unique(v.dropna().to_numpy())[:5])
        (dg if uniq <= {0, 1} and len(np.unique(v.dropna())) <= 2 and not re.match(r'^[iv]', str(c).strip().lower()) else an).append(c)
    units = ['kV' if re.search(r'kv', str(c).lower()) else 'kA' if re.search(r'ka', str(c).lower()) else
             ('V' if str(c).strip().upper().startswith('V') or 'volt' in str(c).lower() else 'A') for c in an]
    names = [re.sub(r'\s*\(.*\)$', '', str(c)).strip() for c in an]
    return {'cfg': None, 't': t - t[0], 'A': df[an].apply(pd.to_numeric, errors='coerce').to_numpy(float),
            'D': df[dg].fillna(0).astype(int).to_numpy() if dg else np.zeros((len(t), 0), int),
            'analog': names, 'units': units, 'digital': [str(c) for c in dg], 'freq': freq, 'start': None, 'trigger': None,
            'station': '', 'device': '', 'ratios': [(1.0, 1.0, 'P')] * len(an), 'format': 'CSV'}


# ---------------------------------------------------------------- channel mapping
def _clean(n):
    return re.sub(r'[^a-z0-9]', '', str(n).lower())


def map_channels(rec):
    """Indices of IA, IB, IC, IN, VA, VB, VC (None when absent)."""
    names = [_clean(n) for n in rec['analog']]
    units = [str(u).upper().strip() for u in rec['units']]
    isI = [u in ('A', 'KA') for u in units]; isV = [u in ('V', 'KV') for u in units]
    m = {}
    for ph in 'abc':
        m['I' + ph.upper()] = next((i for i, n in enumerate(names) if isI[i] and re.fullmatch(rf'(i|il|current)?{ph}|i{ph}\w{{0,3}}|il?{"123"["abc".index(ph)]}', n)
                                    and 'sens' not in n), None)
        m['V' + ph.upper()] = next((i for i, n in enumerate(names) if isV[i] and re.fullmatch(rf'(v|u|vl|ul)?{ph}(n|g)?|v{ph}\w{{0,3}}|u?l?{"123"["abc".index(ph)]}(n|e)?', n)
                                    and 'sync' not in n and 'check' not in n), None)
    m['IN'] = next((i for i, n in enumerate(names) if isI[i] and re.fullmatch(r'i?(n|e|0|res|g|3i0)|in\w{0,2}|ie|neutral', n) and 'sens' not in n), None)
    cur = [i for i in range(len(names)) if isI[i] and i != m['IN'] and 'sens' not in names[i]]
    vol = [i for i in range(len(names)) if isV[i] and not re.search(r'sync|check|vn$|v0|bus', names[i])]
    for k, pool in (('I', cur), ('V', vol)):
        if any(m[k + p] is None for p in 'ABC') and len(pool) >= 3:
            for p, i in zip('ABC', pool[:3]):
                m[k + p] = i
    return m


def scale_primary(rec, mode='Auto', ct=1.0, vt=1.0):
    """Analog values in primary amperes and volts. mode: Auto (record as stored), Primary, Secondary (apply ct, vt)."""
    A = rec['A'].copy()
    for j, u in enumerate(rec['units']):
        u = str(u).upper().strip()
        if u in ('KA', 'KV'):
            A[:, j] *= 1000.0
        pr, se, ps = rec['ratios'][j]
        if mode == 'Secondary':
            A[:, j] *= (ct if u in ('A', 'KA') else vt)
        elif mode == 'Auto' and ps == 'S' and se > 0 and pr / se > 1.01:
            A[:, j] *= pr / se
    return A


# ---------------------------------------------------------------- phasors
def phasors(x, N):
    """One-cycle full Fourier phasor (rms, complex) of every sample; the window ends at the sample."""
    n = np.arange(N)
    c = np.exp(-2j * np.pi * n / N)
    k = np.sqrt(2) / N
    y = np.convolve(np.nan_to_num(x), c[::-1], 'full')[:len(x)] * k
    # rotate so that a steady cosine keeps a constant angle
    idx = np.arange(len(x))
    y = y * np.exp(-2j * np.pi * (idx - N + 1) / N)
    y[:N - 1] = np.nan
    return y


def cycle_rms(x, N):
    s = pd.Series(np.nan_to_num(x) ** 2).rolling(N, min_periods=N).mean().to_numpy()
    return np.sqrt(s)


# ---------------------------------------------------------------- fault detection and features
def detect(rec, A, ch):
    """Inception, pre-fault and fault sample indices, interruption and reclosing."""
    t = rec['t']; f0 = rec['freq']
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / (f0 * 32)
    N = max(8, int(round(1 / (f0 * dt))))
    I = [A[:, ch[k]] for k in ('IA', 'IB', 'IC') if ch.get(k) is not None]
    if len(I) < 3:
        raise ValueError('three phase currents were not found in the record')
    I = np.vstack(I)
    d = np.zeros(I.shape[1]); d[N:] = np.abs(I[:, N:] - I[:, :-N]).max(0)
    pre_n = min(len(d) // 4, 2 * N)
    noise = float(np.percentile(d[N:N + pre_n], 99)) if pre_n > 2 else 0.0
    rmsI = np.vstack([cycle_rms(x, N) for x in I])
    peak = np.nanmax(rmsI)
    thr = max(4 * noise, 0.08 * np.sqrt(2) * peak)
    hits = np.where(d > thr)[0]
    hits = hits[hits > N]
    if not len(hits):
        return {'N': N, 'dt': dt, 'inception': None}
    inc = int(hits[0])
    # move back to the true start (superimposed value above 25% of the threshold)
    while inc > N and d[inc - 1] > 0.25 * thr:
        inc -= 1
    pre = inc - 2
    # interruption: every phase current below 10% of the fault level (cycle rms), after the inception
    fault_level = np.nanmax(rmsI[:, inc:inc + 3 * N], axis=1)
    low = rmsI < np.maximum(0.1 * np.nanmax(fault_level), 1e-6)[..., None]
    allow = np.where(low.all(0) & (np.arange(rmsI.shape[1]) > inc + N // 2))[0]
    intr = int(allow[0]) - N + 1 if len(allow) else None
    fault_end = intr if intr is not None else min(len(d) - 1, inc + 6 * N)
    span = fault_end - inc
    fidx = inc + min(int(1.5 * N), max(N, span - 1)) if span > N else inc + N - 1
    fidx = min(fidx, len(d) - 1)
    # reclosing: current returns after interruption
    reclose, refault = None, None
    if intr is not None:
        on = np.where((rmsI.max(0) > max(0.05 * peak, 3 * noise / np.sqrt(2) + 1e-9)) & (np.arange(rmsI.shape[1]) > intr + 2 * N))[0]
        if len(on):
            reclose = int(on[0]) - N + 1
            h2 = np.where(d[reclose + N:] > thr)[0]
            if len(h2):
                j = reclose + N + int(h2[0])
                if rmsI.max(0)[min(len(d) - 1, j + N)] > 0.5 * np.nanmax(fault_level):
                    refault = j
    return {'N': N, 'dt': dt, 'inception': inc, 'pre': pre, 'fault': fidx, 'interruption': intr, 'reclose': reclose, 'refault': refault,
            'threshold': thr, 'rmsI': rmsI}


def fault_phasors(A, ch, ev, N):
    out = {}
    for k in ('IA', 'IB', 'IC', 'IN', 'VA', 'VB', 'VC'):
        if ch.get(k) is None:
            continue
        p = phasors(A[:, ch[k]], N)
        out[k] = (p[ev['pre']], p[ev['fault']])
    return out


def features_from_phasors(P):
    """Scale-free features for the fault-type model."""
    Ip = np.array([P[k][0] for k in ('IA', 'IB', 'IC')]); If = np.array([P[k][1] for k in ('IA', 'IB', 'IC')])
    dI = If - Ip
    has_v = all(k in P for k in ('VA', 'VB', 'VC'))
    if has_v:
        Vp = np.array([P[k][0] for k in ('VA', 'VB', 'VC')]); Vf = np.array([P[k][1] for k in ('VA', 'VB', 'VC')])
    else:
        Vp = np.ones(3, complex); Vf = np.ones(3, complex)
    m = np.abs(dI).max() + 1e-9
    s = AINV @ If; ds = AINV @ dI
    vr = np.abs(Vf) / np.maximum(np.abs(Vp), 1e-9)
    Vpp_p = np.array([Vp[0] - Vp[1], Vp[1] - Vp[2], Vp[2] - Vp[0]]); Vpp_f = np.array([Vf[0] - Vf[1], Vf[1] - Vf[2], Vf[2] - Vf[0]])
    vpp = np.abs(Vpp_f) / np.maximum(np.abs(Vpp_p), 1e-9)
    dpp = np.array([dI[0] - dI[1], dI[1] - dI[2], dI[2] - dI[0]])
    ang = lambda a, b: np.angle(a * np.conj(b))
    a20 = ang(ds[2], ds[0]) if abs(ds[0]) > 0.02 * m else 0.0
    feat = np.r_[np.abs(dI) / m, np.abs(If) / (np.abs(If).max() + 1e-9), abs(ds[0]) / (abs(ds[1]) + 1e-9), abs(ds[2]) / (abs(ds[1]) + 1e-9),
                 abs(s[0]) / (abs(s[1]) + 1e-9), vr, vpp, np.abs(dpp) / (np.abs(dpp).max() + 1e-9),
                 np.cos(a20), np.sin(a20), np.cos(ang(dI[0], dI[1])), np.cos(ang(dI[1], dI[2])), np.cos(ang(dI[2], dI[0]))]
    return np.clip(np.nan_to_num(feat, nan=0.0, posinf=10, neginf=-10), -10, 10)


FEATURE_NAMES = ['dIa', 'dIb', 'dIc', 'Ia', 'Ib', 'Ic', 'dI0/dI1', 'dI2/dI1', 'I0/I1', 'Va dip', 'Vb dip', 'Vc dip', 'Vab dip', 'Vbc dip', 'Vca dip',
                 'dIab', 'dIbc', 'dIca', 'cos(I2,I0)', 'sin(I2,I0)', 'cos(dIa,dIb)', 'cos(dIb,dIc)', 'cos(dIc,dIa)']


def rule_type(P):
    """Classic phase selection from superimposed currents (to cross-check the AI)."""
    Ip = np.array([P[k][0] for k in ('IA', 'IB', 'IC')]); If = np.array([P[k][1] for k in ('IA', 'IB', 'IC')])
    dI = If - Ip
    dpp = np.abs([dI[0] - dI[1], dI[1] - dI[2], dI[2] - dI[0]])
    ds = AINV @ dI
    ground = abs(ds[0]) > 0.12 * abs(ds[1])
    mx = dpp.max()
    big = dpp > 0.6 * mx
    if big.all():
        return 'ABC'
    pairs = ['AB', 'BC', 'CA']
    if ground:
        mag = np.abs(dI); ph = 'ABC'
        hi = [ph[i] for i in range(3) if mag[i] > 0.55 * mag.max()]
        if len(hi) == 1:
            return hi[0] + 'G'
        if len(hi) == 2:
            s = ''.join(hi)
            return {'AB': 'ABG', 'BC': 'BCG', 'AC': 'CAG'}[s]
        return 'ABC'
    return pairs[int(np.argmax(dpp))]


# ---------------------------------------------------------------- fault location (Takagi)
def locate(P, ftype, line):
    """Distance to the fault (km) by the Takagi method; line = dict(length_km, r1, x1, r0, x0 per km)."""
    Z1 = complex(line['r1'], line['x1']); Z0 = complex(line['r0'], line['x0'])
    k0 = (Z0 - Z1) / (3 * Z1)
    g = lambda k, i: P[k][i]
    if not all(k in P for k in ('VA', 'VB', 'VC')):
        return None, None
    I = {p: (g('I' + p, 0), g('I' + p, 1)) for p in 'ABC'}
    V = {p: g('V' + p, 1) for p in 'ABC'}
    i0f = (I['A'][1] + I['B'][1] + I['C'][1]) / 3
    i0p = (I['A'][0] + I['B'][0] + I['C'][0]) / 3
    if ftype in ('AG', 'BG', 'CG'):
        p = ftype[0]
        Vl = V[p]; Il = I[p][1] + k0 * 3 * i0f; dI = 3 * (i0f - i0p)          # modified Takagi with zero-sequence current
    else:
        pair = {'AB': 'AB', 'BC': 'BC', 'CA': 'CA', 'ABG': 'AB', 'BCG': 'BC', 'CAG': 'CA', 'ABC': 'AB'}[ftype]
        p, q = pair
        Vl = V[p] - V[q]; Il = I[p][1] - I[q][1]; dI = (I[p][1] - I[p][0]) - (I[q][1] - I[q][0])
    den = (Z1 * Il * np.conj(dI)).imag
    if abs(den) < 1e-12:
        return None, None
    x = (Vl * np.conj(dI)).imag / den
    Zapp = Vl / Il if abs(Il) > 0 else None
    return float(x), Zapp


def impedance_trajectory(A, ch, ftype, line, N, i0, i1):
    """Apparent loop impedance (ohm, primary) from i0 to i1 for the R-X plot."""
    Z1 = complex(line['r1'], line['x1']); Z0 = complex(line['r0'], line['x0']); k0 = (Z0 - Z1) / (3 * Z1)
    ph = {k: phasors(A[:, ch[k]], N)[i0:i1] for k in ('IA', 'IB', 'IC', 'VA', 'VB', 'VC') if ch.get(k) is not None}
    if len(ph) < 6:
        return None
    if ftype in ('AG', 'BG', 'CG'):
        p = ftype[0]; i0s = (ph['IA'] + ph['IB'] + ph['IC']) / 3
        z = ph['V' + p] / (ph['I' + p] + k0 * 3 * i0s)
    else:
        pair = {'AB': 'AB', 'BC': 'BC', 'CA': 'CA', 'ABG': 'AB', 'BCG': 'BC', 'CAG': 'CA', 'ABC': 'AB'}[ftype]
        p, q = pair
        z = (ph['V' + p] - ph['V' + q]) / (ph['I' + p] - ph['I' + q])
    return z


# ---------------------------------------------------------------- protection elements from names
ELEMENTS = [
    (r'z(one)?\s*1|dist.*z1|21.*z1|\bz1\b', 'Distance zone 1'), (r'z(one)?\s*2|\bz2\b', 'Distance zone 2'), (r'z(one)?\s*3|\bz3\b', 'Distance zone 3'),
    (r'87\s*l|line diff|\bdiff\b|differential|87', 'Line differential (87L)'), (r'aided|\baid\s*\d|pott|putt|carrier|dcb|85', 'Aided / carrier scheme'),
    (r'\bdef\b|directional earth|67n', 'Directional earth fault (67N)'), (r'in\s*>|ie\s*>|earth fault|e/f|51n|50n', 'Earth fault overcurrent'),
    (r'i\s*>|overcurrent|o/c|\b51\b|\b50\b', 'Phase overcurrent'), (r'sotf|tor|switch on to fault', 'Switch onto fault (SOTF)'),
    (r'dtt|intertrip|direct transfer', 'Direct transfer trip'), (r'cb\s*fail|50bf|breaker fail', 'Breaker failure (50BF)'),
    (r'v\s*<|undervolt|\b27\b(?!\s*(ms|s|km|ka|a)\b)', 'Undervoltage'), (r'v\s*>|overvolt|\b59\b(?!\s*(ms|s|km|ka|a)\b)', 'Overvoltage')]


def element_of(text):
    s = str(text).lower()
    for pat, lab in ELEMENTS:
        if re.search(pat, s):
            return lab
    return None


def is_trip(text):
    s = str(text).lower()
    return bool(re.search(r'trip|\b94\b|\b86\b', s)) and not re.search(r'block|reset|test|circuit sup|tcs|fail to|lockout reset', s)


def binary_changes(rec, i_from=0):
    """(time s from record start, channel, new state) for every binary change after sample i_from."""
    D, t = rec['D'], rec['t']
    out = []
    for j, name in enumerate(rec['digital']):
        x = D[:, j]
        ch = np.where(np.diff(x) != 0)[0] + 1
        for k in ch:
            if k >= i_from:
                out.append((float(t[k] - t[0]), name, int(x[k])))
    return sorted(out)


# ---------------------------------------------------------------- event lists
DT_PATTERNS = [
    (r'(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})[ ,T]+(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{1,6}))?', 'dmy'),
    (r'(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})[ ,T]+(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{1,6}))?', 'ymd'),
    (r'(\d{1,2})[ \-]([A-Za-z]{3,9})[ \-](\d{4})[ ,T]+(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{1,6}))?', 'dMy'),
]
MONTHS = {m: i + 1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}


def _dt(m, kind):
    g = m.groups()
    frac = int((g[6] or '0').ljust(6, '0')[:6])
    if kind == 'dmy':
        d, mo, y = int(g[0]), int(g[1]), int(g[2])
        if mo > 12 and d <= 12:
            d, mo = mo, d
    elif kind == 'ymd':
        y, mo, d = int(g[0]), int(g[1]), int(g[2])
    else:
        d, mo, y = int(g[0]), MONTHS.get(g[1].lower()[:3], 1), int(g[2])
    return datetime(y, mo, d, int(g[3]), int(g[4]), int(g[5]), frac)


WEEKDAY = r'(?:(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*,?\s+)?'
HEADER_SKIP = re.compile(r'printed on|^\s*page\s*:|events report|^\s*file\s*:|model number:.*printed', re.I)


def parse_event_text(txt):
    """Time-stamped events from a text export. Two layouts:
    one event per line (time anywhere on the line), or blocks (MiCOM S1 / similar) where a line starting with the
    time is the event and the indented lines under it are its details (fault record values, bit lists)."""
    lines = [ln.rstrip() for ln in txt.splitlines()]
    starts = 0
    for ln in lines:
        for pat, _ in DT_PATTERNS:
            if re.match(r'.{0,25}?' + WEEKDAY + pat, ln.strip(), re.I):
                starts += 1; break
    block = starts >= 3
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s or HEADER_SKIP.search(s):
            continue
        hit = None
        for pat, kind in DT_PATTERNS:
            m = (re.match(r'.{0,25}?' + WEEKDAY + pat, s, re.I) if block else re.search(pat, s))
            if m:
                hit = (m, kind); break
        if hit:
            m, kind = hit
            try:
                ts = _dt(m, kind)
            except (ValueError, AttributeError):
                continue
            ev = (s[m.end():] if block else (s[:m.start()] + ' ' + s[m.end():])).strip(' \t,;|-:')
            ev = re.sub(r'\s{2,}', '  ', ev)
            if ev:
                rows.append([ts, ev, []])
        elif block and rows:
            rows[-1][2].append(re.sub(r'\s{2,}', ' ', s))
    df = pd.DataFrame([(r[0], r[1], '\n'.join(r[2])) for r in rows], columns=['Timestamp', 'Event', 'Detail'])
    return df


def device_of(txt):
    m = re.search(r'model\s*(?:number|no\.?)\s*:?\s*([A-Z]\d{3}[A-Z0-9]*)', txt, re.I)
    return m.group(1)[:4].upper() if m else ''


IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def _ocr_image(im):
    """Text from a picture of a printout or a screen (Tesseract). Upscales small pictures for better reading."""
    import pytesseract
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(im).convert('L')
    if im.width < 1800:
        f = 1800 / im.width
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
    im = ImageOps.autocontrast(im)
    return pytesseract.image_to_string(im, config='--psm 6')


def extract_text(data, name=''):
    """Plain text of an uploaded file: PDF (text layer, or OCR when scanned), picture (OCR), CSV / TXT, Excel.
    Returns (text, how) where how says which reader was used."""
    low = name.lower()
    if low.endswith('.pdf') or data[:4] == b'%PDF':
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text(extraction_mode='layout') or '') for p in r.pages]
        txt = '\n'.join(pages)
        if len(re.sub(r'\s', '', txt)) >= 40 * max(1, len(pages)) * 0.3:
            return txt, 'PDF text'
        try:                                    # scanned PDF: render the pages and read them
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(data)
            out = []
            for i in range(min(len(doc), 60)):
                out.append(_ocr_image(doc[i].render(scale=300 / 72).to_pil()))
            return '\n'.join(out), 'scanned PDF (OCR)'
        except Exception as e:
            return txt, f'PDF text (OCR not available: {e})'
    if low.endswith(IMAGE_EXT) or data[:8] == b'\x89PNG\r\n\x1a\n' or data[:3] == b'\xff\xd8\xff':
        from PIL import Image
        return _ocr_image(Image.open(io.BytesIO(data))), 'picture (OCR)'
    if low.endswith(('.xlsx', '.xls')):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
        rows = []
        for df in sheets.values():
            for r in df.astype(str).replace('nan', '').values.tolist():
                rows.append('  '.join(x for x in r if x))
        return '\n'.join(rows), 'Excel'
    for enc in ('utf-8-sig', 'utf-16', 'cp1252', 'latin-1'):
        try:
            return data.decode(enc), 'text'
        except Exception:
            continue
    return '', 'unreadable'


def parse_events(data, name=''):
    """Relay event list from CSV, TXT, PDF (text or scanned), a picture or Excel: returns Timestamp, Event, Detail."""
    low = name.lower()
    if low.endswith(('.csv', '.txt')) or not (low.endswith(('.pdf', '.xlsx', '.xls') + IMAGE_EXT)):
        txt, how = extract_text(data, name)
        if low.endswith('.csv') or (txt.count(',') > txt.count('\n') and not low.endswith('.txt')):
            try:
                df = pd.read_csv(io.StringIO(txt), sep=None, engine='python')
                tcols = [c for c in df.columns if re.search(r'time|date|stamp', str(c).lower())]
                if tcols:
                    ts = pd.to_datetime(df[tcols].astype(str).agg(' '.join, axis=1), errors='coerce', dayfirst=True)
                    other = [c for c in df.columns if c not in tcols]
                    ev = df[other].astype(str).replace('nan', '').agg('  '.join, axis=1).str.strip()
                    out = pd.DataFrame({'Timestamp': ts, 'Event': ev, 'Detail': ''}).dropna(subset=['Timestamp'])
                    out = out[out['Event'] != '']
                    if len(out):
                        out = out.reset_index(drop=True); out.attrs.update({'device': device_of(txt), 'read_as': 'CSV'})
                        return out
            except Exception:
                pass
    else:
        txt, how = extract_text(data, name)
    df = parse_event_text(txt)
    df.attrs.update({'device': device_of(txt), 'read_as': how})
    return df


# ---------------------------------------------------------------- relay settings files
_NUM = r'([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*(k|m|M)?\s*(ohms?|Ω|ohm|km|mi|m|%|deg|°|kv|v|ka|a|ms|s)?(?![a-z])'
SET_KEYS = [
    ('line_length', r'line\s*length'),
    ('line_angle', r'line\s*angle'),
    ('line_z', r'line\s*imped\w*|line\s*z1?\b(?!\s*angle)'),
    ('kzn_angle', r'k\s*z\s*n\d?\s*(?:res\w*\s*)?ang\w*|k0\s*ang\w*|kz0?\s*ang\w*'),
    ('kzn', r'k\s*z\s*n\d?\s*(?:res\w*\s*)?(?:comp\w*|magn?\w*|factor)?(?!\s*(?:res\w*\s*)?ang)|residual\s*comp\w*\s*factor|\bk0\b(?!\s*ang)'),
    ('kzm', r'k\s*z\s*m\s*(?:mutual\s*)?(?:comp\w*|set\w*)?(?!\s*ang)'),
    ('z1_reach', r'z(?:one)?\s*1\s*(?:ph\w*|gnd|ground|g|p)?\.?\s*reach|z1\s*(?:ph|gnd)?\s*reach'),
    ('z2_reach', r'z(?:one)?\s*2\s*(?:ph\w*|gnd|ground|g|p)?\.?\s*reach|z2\s*(?:ph|gnd)?\s*reach'),
    ('z3_reach', r'z(?:one)?\s*3\s*(?:ph\w*|gnd|ground|g|p)?\.?\s*reach|z3\s*(?:ph|gnd)?\s*reach'),
    ('r1_ph', r'r\s*1\s*ph\w*\.?\s*res\w*'), ('r2_ph', r'r\s*2\s*ph\w*\.?\s*res\w*'), ('r3_ph', r'r\s*3\s*ph\w*\.?\s*res\w*'),
    ('r1_gnd', r'r\s*1\s*g(?:n)?d\w*\.?\s*res\w*'), ('r2_gnd', r'r\s*2\s*g(?:n)?d\w*\.?\s*res\w*'), ('r3_gnd', r'r\s*3\s*g(?:n)?d\w*\.?\s*res\w*'),
    ('tilt', r'z\s*1\s*tilt\w*(?:\s*top\s*line)?'),
    ('z2_time', r't\s*z(?:one)?\s*2\s*(?:ph\w*|gnd|ground)?\.?\s*(?:delay|time)?|z(?:one)?\s*2\s*(?:ph\w*|gnd)?\s*(?:time\s*)?delay'),
    ('ct_primary', r'(?:phase\s*)?ct\s*prim\w*'),
    ('ct_secondary', r'(?:phase\s*)?ct\s*sec\w*'),
    ('vt_primary', r'(?:main\s*)?vt\s*prim\w*'),
    ('vt_secondary', r'(?:main\s*)?vt\s*sec\w*'),
]
SET_LABEL = {'line_length': 'Line length (km)', 'line_z': 'Line impedance (Ω)', 'line_angle': 'Line angle (°)', 'kzn': 'kZN', 'kzn_angle': 'kZN angle (°)',
             'kzm': 'kZm', 'z1_reach': 'Zone 1 reach', 'z2_reach': 'Zone 2 reach', 'z2_time': 'Zone 2 time (ms)', 'ct_primary': 'CT primary (A)',
             'ct_secondary': 'CT secondary (A)', 'vt_primary': 'VT primary (V)', 'vt_secondary': 'VT secondary (V)', 'values_in': 'Setting values',
             'auto_reclose': 'Auto-reclose', 'device': 'Relay', 'description': 'Description'}


def _num_unit(text):
    m = re.search(_NUM, text, re.I)
    if not m:
        return None, None
    v = float(m.group(1).replace(',', '.'))
    pre, u = (m.group(2) or ''), (m.group(3) or '').lower()
    if pre == 'k':
        v *= 1000
    elif pre == 'm' and u not in ('', 'm'):
        v /= 1000
    elif pre == 'M':
        v *= 1e6
    return v, u


def parse_settings(text):
    """Line and protection settings from a relay settings export (text, PDF text or OCR). Tolerant of layouts:
    'Line Length   120.0 km', '0C01 Line Length: 120.0km', 'kZN Res Comp 800.0E-3' ..."""
    out = {'device': device_of(text)}
    m = re.search(r'description\s*[:=]?\s*(.+)', text, re.I)
    if m:
        out['description'] = m.group(1).strip()[:60]
    for k, pat in (('char_ph', r'phase\s*char\w*\.?\s*[:=]?\s*(quad\w*|mho)'), ('char_gnd', r'(?:ground|gnd|earth)\s*char\w*\.?\s*[:=]?\s*(quad\w*|mho)')):
        m = re.search(pat, text, re.I)
        if m:
            out[k] = 'quad' if m.group(1).lower().startswith('quad') else 'mho'
    m = re.search(r'setting\s*values\s*[:=]?\s*(primary|secondary)', text, re.I)
    if m:
        out['values_in'] = m.group(1).title()
    m = re.search(r'auto[\s\-]*reclos\w*\s*(?:mode|status)?\s*[:=]?\s*(enabled|disabled|in\s*service|out\s*of\s*service|on|off)\b', text, re.I)
    if m:
        out['auto_reclose'] = 'In service' if m.group(1).lower() in ('enabled', 'on', 'in service', 'inservice') else 'Out of service'
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        for key, pat in SET_KEYS:
            if key in out:
                continue
            m = re.search(r'(?:^|[\s:.\d])(' + pat + r')\s*[:=]?\s*', s, re.I)
            if not m:
                continue
            v, u = _num_unit(s[m.end():])
            if v is None:
                continue
            if key == 'line_length':
                v = v * 1.609 if u == 'mi' else (v / 1000 if u == 'm' and v > 2000 else v)
            if key == 'z2_time':
                v = v * 1000 if (u == 's' or (not u and v < 5)) else v
            if key in ('z1_reach', 'z2_reach'):
                out[key + '_unit'] = '%' if u == '%' else 'ohm'
            if key in ('kzn',) and v > 5:            # a percentage written without a unit
                v = v / 100
            out[key] = v
            break
    return out


def settings_line(sts, ct=None, vt=None):
    """Line data (per-km Z1/Z0) and protection settings from the parsed settings of one relay. Returns (line, setting, notes)."""
    line, setg, notes = {}, {}, []
    L = sts.get('line_length')
    if L:
        line['length_km'] = float(L)
    ctr = (sts['ct_primary'] / sts['ct_secondary']) if sts.get('ct_primary') and sts.get('ct_secondary') else ct
    vtr = (sts['vt_primary'] / sts['vt_secondary']) if sts.get('vt_primary') and sts.get('vt_secondary') else vt
    if sts.get('vt_primary'):
        line['kv'] = sts['vt_primary'] / 1000 if sts['vt_primary'] > 1000 else sts['vt_primary']
    Zl = sts.get('line_z')
    if Zl and L:
        if sts.get('values_in') == 'Secondary':
            if ctr and vtr:
                Zl = Zl * vtr / ctr; notes.append(f"Line impedance converted from secondary with CT {ctr:g} and VT {vtr:g}.")
            else:
                notes.append('Line impedance is in secondary ohms but the CT/VT ratios were not found: enter them under settings.')
        ang = np.radians(sts.get('line_angle', 85.0))
        z1 = Zl / L * np.exp(1j * ang)
        k = sts.get('kzn', 1.0 / 3 * 2)
        z0 = z1 * (1 + 3 * k * np.exp(1j * np.radians(sts.get('kzn_angle', 0.0))))
        line.update({'r1': float(z1.real), 'x1': float(z1.imag), 'r0': float(z0.real), 'x0': float(z0.imag)})
        if 'kzn' not in sts:
            notes.append('kZN not found in the settings: 0.67 assumed.')
    for z in ('z1', 'z2'):
        r = sts.get(z + '_reach')
        if r:
            if sts.get(z + '_reach_unit') == '%':
                setg[z + '_reach'] = r / 100
            elif Zl:
                setg[z + '_reach'] = r / sts['line_z']          # same base (primary or secondary) as the line impedance
    if sts.get('z2_time'):
        setg['z2_time_ms'] = float(sts['z2_time'])
    k = (vtr / ctr) if (sts.get('values_in') == 'Secondary' and ctr and vtr) else 1.0
    if sts.get('z3_reach') and sts.get('line_z'):
        setg['z3_reach'] = sts['z3_reach'] / 100 if sts.get('z3_reach_unit') == '%' else sts['z3_reach'] / sts['line_z']
    for kind in ('ph', 'gnd'):
        if sts.get('char_' + kind):
            setg['char_' + kind] = sts['char_' + kind]
        rr = [sts.get(f'r{i}_{kind}') for i in (1, 2, 3)]
        if any(x is not None for x in rr):
            setg['r_' + kind] = [None if x is None else float(x) * k for x in rr]
    if sts.get('tilt') is not None:
        setg['tilt_deg'] = float(sts['tilt'])
    if sts.get('auto_reclose'):
        setg['ar_expected'] = sts['auto_reclose'] == 'In service'
    if ctr:
        setg['_ct'] = ctr
    if vtr:
        setg['_vt'] = vtr
    return line, setg, notes


def best_settings(all_sts):
    """The settings file that carries the line data (normally the distance relay)."""
    score = lambda s: sum(k in s for k in ('line_length', 'line_z', 'line_angle', 'kzn', 'z1_reach')) * 10 + len(s)
    return max(all_sts, key=lambda kv: score(kv[1])) if all_sts else None



def event_values(ev):
    """Values the relay itself reported in its fault record (fault location, duration, trip time, CB time, phases, elements)."""
    out = {}
    det = ev['Detail'] if 'Detail' in ev else pd.Series([''] * len(ev), index=ev.index)
    for e, d in zip(ev['Event'].astype(str), det.fillna('').astype(str)):
        is_rec = bool(re.search(r'fault\s*rec', e.lower()))
        s = (e + '\n' + d).lower()
        m = re.search(r'fault\s*loc\w*\D{0,12}?(-?\d+(?:\.\d+)?)\s*(km|%|mi)?', s)
        if m and 'Relay Fault Location' not in out:
            out['Relay Fault Location'] = f"{m.group(1)} {m.group(2) or 'km'}"
        m = re.search(r'fault\s*dur\w*\D{0,12}?(\d+(?:\.\d+)?)\s*(ms|s)', s)
        if m and 'Relay Fault Duration' not in out:
            out['Relay Fault Duration'] = f"{m.group(1)} {m.group(2)}"
        m = re.search(r'(relay|prot\w*)\s*trip\s*time\D{0,12}?(\d+(?:\.\d+)?)\s*(ms|s)', s)
        if m and 'Relay Trip Time' not in out:
            out['Relay Trip Time'] = f"{float(m.group(2)) * (1000 if m.group(3) == 's' else 1):g} ms"
        m = re.search(r'cb\s*op\w*\s*time\D{0,12}?(\d+(?:\.\d+)?)\s*(ms|s)', s)
        if m and 'Relay CB Time' not in out:
            out['Relay CB Time'] = f"{float(m.group(1)) * (1000 if m.group(2) == 's' else 1):g} ms"
        m = re.search(r'(faulted\s*phases?|fault\s*type)\s*[:=]?\s*([abc][abc/\-]{0,4}(?:\s*-?\s*[ng])?)\b', s)
        if m and 'Relay Faulted Phases' not in out:
            out['Relay Faulted Phases'] = m.group(2).upper().replace(' ', '').replace('/', '')
        if is_rec and d:
            ph = re.findall(r'start\s+([abcn])\s*:\s*on', d.lower())
            if ph and 'Relay Faulted Phases' not in out:
                p = ''.join(x.upper() for x in ph if x != 'n'); out['Relay Faulted Phases'] = p + ('G' if 'n' in ph else '')
            els = [re.sub(r'^\d+\s+', '', x).strip() for x in re.findall(r'^\s*\d+\s+([^:\n]*trip[^:\n]*?)\s*:\s*on', d, re.I | re.M)]
            els = [x for x in els if not re.match(r'(any trip|trip [abcn])$', x.strip(), re.I)]
            if els and 'Relay Trip Elements' not in out:
                out['Relay Trip Elements'] = ', '.join(dict.fromkeys(els))
            m = re.search(r'fault\s*time\s*:\s*(.+)', d, re.I)
            if m and 'Relay Fault Time' not in out:
                for pat, kind in DT_PATTERNS:
                    mm = re.search(pat, m.group(1))
                    if mm:
                        out['Relay Fault Time'] = _dt(mm, kind); break
            for k in ('IA', 'IB', 'IC', 'IN', 'VA', 'VB', 'VC'):
                mm = re.search(rf'^\s*{k}\s+fault\s*:\s*(\d+(?:\.\d+)?)\s*(k?)([av])', d, re.I | re.M)
                if mm and f'Relay {k} Fault' not in out:
                    out[f'Relay {k} Fault'] = float(mm.group(1)) * (1000 if mm.group(2).lower() == 'k' else 1)
            for k in ('IB remote 1', 'IA remote 1', 'IC remote 1'):
                mm = re.search(rf'^\s*{k}\s*:\s*(\d+(?:\.\d+)?)\s*(k?)a', d, re.I | re.M)
                if mm and f'Relay {k}' not in out:
                    out[f'Relay {k}'] = float(mm.group(1)) * (1000 if mm.group(2).lower() == 'k' else 1)
    return out


def align_events(E, t0, dur_s, trigger=None):
    """Shift an event list by whole hours when the relay clock is on another time zone (e.g. UTC) than the record.
    Returns (events, shift hours). The shift that puts the most trip/start events inside the record wins."""
    if E is None or not len(E) or t0 is None:
        return E, 0
    ts = pd.to_datetime(E['Timestamp'])
    key = E['Event'].astype(str).str.contains(r'trip|start|pick|fault', case=False, regex=True)
    best, bh = -1, 0
    for h in range(-12, 13):
        sh = ts + pd.Timedelta(hours=h)
        n = int(((sh >= pd.Timestamp(t0) - pd.Timedelta(seconds=0.5)) & (sh <= pd.Timestamp(t0) + pd.Timedelta(seconds=dur_s + 0.5)) & key).sum())
        if n > best or (n == best and abs(h) < abs(bh)):
            best, bh = n, h
    if bh and best > 0:
        E = E.copy(); E['Timestamp'] = ts + pd.Timedelta(hours=bh)
    return E, (bh if best > 0 else 0)


def classify_event(e):
    s = str(e).lower()
    if re.search(r'auto.?reclos|\bar\b|\b79\b|reclose', s):
        return 'Auto-reclose'
    if re.search(r'\bcb\b|breaker|52', s):
        return 'Breaker'
    if is_trip(s):
        return 'Trip'
    if re.search(r'start|pick.?up|pickup', s):
        return 'Start'
    if re.search(r'fault|loc', s):
        return 'Fault record'
    if re.search(r'alarm|fail|supervision', s):
        return 'Alarm'
    return 'Other'


def compare_table(a):
    """FaultGuard against what each relay reported in its own fault record."""
    f0 = lambda v: '-' if v is None else f"{v:.0f} ms"
    rows = [('Fault type', f"{a['type']} ({TYPE_TEXT.get(a['type'])}), AI {a['type_conf'] * 100:.0f}% · classic {a['type_rule']}", 'Relay Faulted Phases'),
            ('Fault location', f"{a['loc_km']:.1f} km ({a['loc_pu'] * 100:.0f}%)" if a.get('loc_km') is not None else '-', 'Relay Fault Location'),
            ('Trip time', f0(a.get('operate_ms')) + (f" (first: {a['trip_by']})" if a.get('trip_by') not in (None, 'record') else ''), 'Relay Trip Time'),
            ('Breaker time', f0(a.get('breaker_ms')), 'Relay CB Time'),
            ('Fault duration', f0(a.get('clearing_ms')), 'Relay Fault Duration'),
            ('Faulted-phase current', f"{a['I_fault_max_kA']:.2f} kA", None),
            ('Tripping elements', ', '.join(e[1] for e in a['elements']) or '-', 'Relay Trip Elements'),
            ('Auto-reclose', a.get('ar', '-'), None)]
    reps = a.get('relay_reports') or {}
    out = []
    for q, mine, key in rows:
        r = {'Quantity': q, 'FaultGuard AI': mine}
        for dev, v in reps.items():
            if q == 'Faulted-phase current':
                vals = [v.get(f'Relay I{p} Fault') for p in 'ABC']
                vals = [x for x in vals if x is not None]
                r[dev] = f"{max(vals) / 1000:.2f} kA" if vals else '-'
            else:
                r[dev] = str(v.get(key, '-')) if key else '-'
        out.append(r)
    if reps:
        r = {'Quantity': 'Relay clock', 'FaultGuard AI': 'record time'}
        for dev, v in reps.items():
            sh, off = v.get('Clock Shift (h)', 0), v.get('Clock vs record (ms)')
            r[dev] = (f"{sh:+d} h, " if sh else '') + (f"{off:+.0f} ms" if off is not None else '-')
        out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- the analysis
DEFAULT_LINE = {'length_km': 100.0, 'r1': 0.028, 'x1': 0.32, 'r0': 0.21, 'x0': 1.02, 'kv': 400.0}


def analyze(rec, events=None, line=None, setting=None, model=None, scale='Auto', ct=1.0, vt=1.0, meta=None):
    """Full analysis of one incident. setting: expected times and ratings."""
    line = dict(DEFAULT_LINE, **(line or {}))
    st = {'expected_inst_ms': 40.0, 'cb_rated_ms': 50.0, 'critical_ms': 120.0, 'cb_rating_ka': 50.0, 'z1_reach': 0.8, 'z2_reach': 1.2,
          'z2_time_ms': 400.0, 'ar_expected': False}
    st.update(setting or {})
    meta = dict(meta or {})
    A = scale_primary(rec, scale, ct, vt)
    ch = map_channels(rec)
    ev = detect(rec, A, ch)
    out = {'meta': meta, 'line': line, 'setting': st, 'channels': ch, 'N': ev['N'], 'notes': []}
    t = rec['t']; t0 = rec['start']
    out['record_start'] = t0
    if ev['inception'] is None:
        out['fault'] = False
        out['notes'].append('No fault found in the record (no step in the phase currents).')
        return finish(out, rec, events, st)
    out['fault'] = True
    N = ev['N']
    P = fault_phasors(A, ch, ev, N)
    feat = features_from_phasors(P)
    rt = rule_type(P)
    if model is not None:
        pr = model['model'].predict_proba(feat.reshape(1, -1))[0]
        cls = list(model['model'].classes_)
        ai = cls[int(np.argmax(pr))]
        probs = {c: float(p) for c, p in zip(cls, pr)}
    else:
        ai, probs = rt, {rt: 1.0}
    out.update({'P': P, 'features': feat, 'type_ai': ai, 'type_rule': rt, 'type_probs': probs, 'type': ai,
                'type_conf': probs.get(ai, 0.0), 'type_agree': ai == rt})
    ms = lambda i: float((t[i] - t[0]) * 1000) if i is not None else None
    out['t_end_ms'] = float((t[-1] - t[0]) * 1000)
    out['t_inception_ms'] = ms(ev['inception'])
    out['t_interruption_ms'] = ms(ev['interruption'])
    out['t_reclose_ms'] = ms(ev['reclose'])
    out['t_refault_ms'] = ms(ev['refault'])
    # magnitudes
    mag = {}
    for k, (pre, flt) in P.items():
        mag[k] = (abs(pre), abs(flt))
    out['mag'] = mag
    Ifa = np.array([P[k][1] for k in ('IA', 'IB', 'IC')])
    out['I_fault_max_kA'] = float(np.abs(Ifa).max() / 1000)
    out['seq_I'] = np.abs(AINV @ Ifa)
    if all(k in P for k in ('VA', 'VB', 'VC')):
        Vf = np.array([P[k][1] for k in ('VA', 'VB', 'VC')]); Vp = np.array([P[k][0] for k in ('VA', 'VB', 'VC')])
        out['V_min_pu'] = float((np.abs(Vf) / np.maximum(np.abs(Vp), 1e-9)).min())
        out['seq_V'] = np.abs(AINV @ Vf)
    else:
        out['V_min_pu'] = None
    # peak instantaneous current (with DC offset)
    seg = slice(ev['inception'], (ev['interruption'] or ev['inception'] + 4 * N))
    out['I_peak_kA'] = float(np.nanmax(np.abs(A[seg][:, [ch['IA'], ch['IB'], ch['IC']]])) / 1000)
    # location
    x, Zapp = locate(P, ai, line)
    out['loc_km'] = x; out['loc_pu'] = (x / line['length_km']) if x is not None else None; out['Zapp'] = Zapp
    out['traj'] = impedance_trajectory(A, ch, ai, line, N, max(N, ev['inception'] - N), min(len(t), (ev['interruption'] or ev['inception'] + 6 * N) + N))
    # binary channels
    chg = binary_changes(rec, max(0, ev['inception'] - 2 * N))
    out['binary'] = chg
    TRIP_OUT = r'trip\s*(output|out|o/p|cmd|command|3\s*ph|relay|contact)|any\s*trip|general\s*trip|trip\s*[abc123]\b|\b94\b|\b86\b'
    trips = [c for c in chg if c[2] == 1 and is_trip(c[1]) and c[0] * 1000 >= out['t_inception_ms'] - 5]
    outs = [c for c in trips if re.search(TRIP_OUT, c[1].lower())]
    trip_t = [c[0] * 1000 for c in (outs or trips)]
    out['t_trip_ms'] = min(trip_t) if trip_t else None
    out['trip_source'] = 'trip output channel' if outs else ('protection element channel' if trips else None)
    out['three_pole_trip'] = any(re.search(r'3\s*ph|three', c[1].lower()) for c in outs) or \
        (sum(1 for p in 'abc' if any(re.search(rf'trip\s*(output\s*)?{p}\b', c[1].lower()) for c in outs)) == 3)
    elems = []
    for tm, name, v in chg:
        if v == 1 and tm * 1000 >= out['t_inception_ms'] - 5:
            e = element_of(name)
            if e and e not in [x_[1] for x_ in elems] and (is_trip(name) or 'zone' in e.lower() or 'diff' in e.lower()):
                elems.append((tm * 1000, e, name))
    cbo = [c[0] * 1000 for c in chg if re.search(r'cb.*open|52\s*b|cb\s*off|breaker open', c[1].lower()) and c[2] == 1]
    out['t_cb_aux_ms'] = min(cbo) if cbo else None
    arc = [c[0] * 1000 for c in chg if re.search(r'ar.*close|reclose|79.*close|close\s*cmd', c[1].lower()) and c[2] == 1]
    out['t_ar_cmd_ms'] = min(arc) if arc else None
    out['elements'] = elems
    out['ev'] = ev
    out['A'] = A
    return finish(out, rec, events, st)


def fmt_dur(sec):
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f} s"
    h, r = divmod(int(round(sec)), 3600); m, s_ = divmod(r, 60)
    return (f"{h} h {m:02d} min" if h else f"{m} min {s_:02d} s")


def on_items(detail):
    """Names of the bits reported ON in an event's detail lines ('10 L11 Foc 3ph TrP : ON')."""
    return [re.sub(r'^\d+\s+', '', m.strip()) for m in re.findall(r'^.{0,12}?\b\d{1,2}\.?\s*([A-Za-z][^:\n]*?)\s*:\s*ON\s*$', str(detail or ''), re.M)]


def _as_list(events):
    if events is None:
        return []
    if isinstance(events, pd.DataFrame):
        return [('Relay', events)] if len(events) else []
    return [(n, e) for n, e in events if e is not None and len(e)]


def finish(out, rec, events, st):
    """Event-list merge (one or more relays), clock alignment, timing, assessment and risk."""
    t0 = out.get('record_start')
    dur = float(rec['t'][-1] - rec['t'][0])
    parts, reports, clock = [], {}, {}
    groups = {}
    for name, ev in _as_list(events):           # several files of the same relay (PDF pages, pictures) are merged
        dev = getattr(ev, 'attrs', {}).get('device') or name
        e = ev.copy()
        if 'Detail' not in e:
            e['Detail'] = ''
        e['Timestamp'] = pd.to_datetime(e['Timestamp'])
        groups.setdefault(dev, []).append(e)
    for dev, lst in groups.items():
        e = pd.concat(lst, ignore_index=True)
        e = e.drop_duplicates(subset=['Timestamp', 'Event']).sort_values('Timestamp', kind='stable').reset_index(drop=True)
        e, sh = align_events(e, t0, dur)
        e['Device'] = dev
        vals = event_values(e)
        if sh:
            vals['Clock Shift (h)'] = sh
            if isinstance(vals.get('Relay Fault Time'), datetime):
                vals['Relay Fault Time'] = vals['Relay Fault Time'] + timedelta(hours=sh)
        if t0 is not None and out.get('fault'):
            ev_s = e['Event'].astype(str)
            st_ev = e[ev_s.str.contains(r'start', case=False) & ev_s.str.contains(r'\bon\b', case=False)]
            st_ev = st_ev[(st_ev['Timestamp'] >= pd.Timestamp(t0)) & (st_ev['Timestamp'] <= pd.Timestamp(t0) + pd.Timedelta(seconds=dur))]
            if len(st_ev):
                t_start = (st_ev['Timestamp'].min() - pd.Timestamp(t0)).total_seconds() * 1000
                off = t_start - out['t_inception_ms']
                md = re.match(r'(\d+(?:\.\d+)?)\s*(ms|s)', str(vals.get('Relay Fault Duration', '')))
                if md and out.get('t_interruption_ms') is not None:
                    # the relay's own start + fault duration must land on the measured current interruption
                    off = t_start + float(md.group(1)) * (1000 if md.group(2) == 's' else 1) - out['t_interruption_ms']
                vals['Clock vs record (ms)'] = round(off, 1)
                if abs(off) > 20:
                    e['Timestamp'] = e['Timestamp'] - pd.Timedelta(milliseconds=off)
        reports[dev] = vals; clock[dev] = sh
        parts.append(e)
    E_all = pd.concat(parts, ignore_index=True).sort_values('Timestamp', kind='stable').reset_index(drop=True) if parts else None
    out['relay_reports'] = reports
    out['clock_shift'] = clock
    E = None
    if E_all is not None:
        E_all['Category'] = E_all['Event'].map(classify_event)
        E_all['Element'] = E_all['Event'].map(element_of)
        if t0 is not None:
            w = (E_all['Timestamp'] >= pd.Timestamp(t0) - pd.Timedelta(seconds=2)) & (E_all['Timestamp'] <= pd.Timestamp(t0) + pd.Timedelta(seconds=dur + 5))
            E = E_all[w].reset_index(drop=True)
        else:
            E = E_all
        if E is not None and not len(E):
            E = None
    out['events_all'] = E_all
    out['relay_values'] = next(iter(reports.values()), {}) if reports else {}
    out['events'] = E
    if not out.get('fault'):
        out.update({'Mode Risks': {m: 0.0 for m in MODE_ORDER}, 'Reasons': {m: [] for m in MODE_ORDER}})
        out['Reasons']['Record quality'] = out['notes'][:]
        out['Mode Risks']['Record quality'] = 20.0
        return risk(out)
    # trip time from the event list when the record has no trip channel
    if out.get('t_trip_ms') is None and E is not None and t0 is not None:
        tr = E[E['Event'].map(is_trip)]
        if len(tr):
            dtm = (tr['Timestamp'] - pd.Timestamp(t0)).dt.total_seconds() * 1000
            dtm = dtm[(dtm > out['t_inception_ms'] - 50) & (dtm < out['t_inception_ms'] + 5000)]
            if len(dtm):
                out['t_trip_ms'] = float(dtm.min()); out['notes'].append('Trip time taken from the event list (no trip channel in the record).')
    if E is not None:
        for _, r in E.iterrows():
            e = r['Element']
            ev_l = str(r['Event']).lower()
            if isinstance(e, str) and is_trip(r['Event']) and not re.search(r'\boff\b', ev_l) and not re.search(r'fault\s*rec|fault\s*loc', ev_l):
                lab = f"{e} [{r['Device']}]" if len(reports) > 1 else e
                have = [x[1] for x in out['elements']]
                if lab not in have and e not in have:
                    tm = (pd.Timestamp(r['Timestamp']) - pd.Timestamp(t0)).total_seconds() * 1000 if t0 is not None else None
                    out['elements'].append((tm, lab, f"event: {r['Event']}"))
    # restoration after the record (manual close or a later reclose)
    out['restored'] = None
    if E_all is not None and t0 is not None:
        tend = pd.Timestamp(t0) + pd.Timedelta(seconds=dur)
        later = E_all[E_all['Timestamp'] > tend]
        txt = later['Event'].astype(str).str.lower() + ' ' + later['Detail'].fillna('').astype(str).str.lower()
        hit = later[txt.str.contains(r'(?:any|all)\s*poles?\s*dead\s*off|cb\s*clos\w*\s*on|clo\w*\s*c(?:m|om)d\w*\s*:?\s*on|ar\s*close', regex=True)]
        if len(hit):
            first = hit.iloc[0]
            secs = (first['Timestamp'] - pd.Timestamp(t0)).total_seconds() - out['t_inception_ms'] / 1000
            on = on_items(first.get('Detail', ''))
            what = next((x for x in on if re.search(r'clo|close', x.lower())), None)
            out['restored'] = {'time': first['Timestamp'], 'after_s': secs, 'event': (f"{first['Event']}: {what}" if what else first['Event']),
                               'device': first['Device']}
    out['elements'] = sorted(out['elements'], key=lambda x: (x[0] is None, x[0] or 0))
    # earliest trip command from any relay (event lists aligned to the record)
    out['trip_by'] = 'record' if out.get('t_trip_ms') is not None else None
    if E is not None and t0 is not None and len(reports) > 0:
        evl = E['Event'].astype(str).str.lower()
        tr = E[evl.str.contains(r'(?:any\s*trip|trip\s*output\s*[abc]?|trip\s*3\s*ph|general\s*trip)\s*on\b', regex=True)]
        if len(tr):
            tm = (tr['Timestamp'] - pd.Timestamp(t0)).dt.total_seconds() * 1000
            ok = (tm > out['t_inception_ms'] - 5) & (tm < out['t_inception_ms'] + 3000)
            if ok.any():
                i = tm[ok].idxmin()
                if out.get('t_trip_ms') is None or tm[i] < out['t_trip_ms'] - 2:
                    out['t_trip_ms'] = float(tm[i]); out['trip_by'] = str(tr.loc[i, 'Device'])
    inc, intr, trip = out['t_inception_ms'], out['t_interruption_ms'], out.get('t_trip_ms')
    out['operate_ms'] = (trip - inc) if trip is not None else None
    out['breaker_ms'] = (intr - trip) if (intr is not None and trip is not None) else None
    out['clearing_ms'] = (intr - inc) if intr is not None else None
    # auto-reclose
    if out.get('t_reclose_ms') is not None or out.get('t_ar_cmd_ms') is not None:
        out['ar'] = 'Unsuccessful (fault still present: permanent fault)' if out.get('t_refault_ms') else 'Successful (transient fault)'
        out['dead_time_ms'] = (out['t_reclose_ms'] - intr) if (out.get('t_reclose_ms') and intr) else None
    else:
        evs = ' '.join(E['Event'].astype(str).str.lower()) if E is not None else ''
        init = bool(re.search(r'ar\s*init|79\s*init|reclos\w*\s*init', evs))
        if 'lockout' in evs:
            out['ar'] = 'Lockout'
        elif intr is None:
            out['ar'] = 'Fault not cleared in the record'
        elif not st.get('ar_expected'):
            out['ar'] = 'Out of service (external auto-recloser off)'
        else:
            out['ar'] = 'Initiated, no reclose' if init else 'Not attempted / not in record'
        out['ar_initiated'] = init
        out['dead_time_ms'] = None
        rs = out.get('restored')
        if rs and intr is not None:
            out['ar'] += '; line re-energised ' + fmt_dur(rs['after_s']) + ' later'
    if out['ar'].startswith('Unsuccessful') is False and E is not None:
        evs = ' '.join(E['Event'].astype(str).str.lower())
        if re.search(r'ar\s*(unsuccess|fail)|lockout', evs) and out['ar'].startswith('Successful'):
            out['ar'] = 'Unsuccessful (relay reports lockout)'
    return assess(out, st)


def test_signs(out):
    """Signs that the record is a secondary injection test rather than a system fault."""
    sg = []
    P = out.get('P') or {}
    vnom = out['line'].get('kv', 400.0) * 1000 / np.sqrt(3)
    if all(k in P for k in ('VA', 'VB', 'VC')):
        vmax = max(max(abs(P[k][0]), abs(P[k][1])) for k in ('VA', 'VB', 'VC'))
        if vmax < 0.05 * vnom:
            sg.append(f"No line voltage in the record (below {0.05 * vnom / 1000:.0f} kV)")
    trip, intr = out.get('t_trip_ms'), out.get('t_interruption_ms')
    if trip is not None and intr is None:
        dur = out['t_end_ms'] - trip
        if dur > 150:
            sg.append(f"The trip was given but the current kept flowing {dur:.0f} ms to the end of the record")
    t0 = out.get('record_start')
    if t0 is not None and t0.year < 2000:
        sg.append(f"Relay clock at {t0:%Y-%m-%d} (not synchronised, typical of a bench test)")
    return sg


def assess(out, st):
    R, why = {}, {}
    sg = test_signs(out)
    out['test_signs'] = sg
    out['likely_test'] = len(sg) >= 2 or (len(sg) == 1 and sg[0].startswith('No line voltage') and out.get('t_interruption_ms') is None)
    if out['likely_test']:
        R = {m: 0.0 for m in MODE_ORDER}; why = {m: [] for m in MODE_ORDER}
        why['Protection operation'] = [f"Protection operated in {out['operate_ms']:.0f} ms." if out.get('operate_ms') is not None else 'No trip found.']
        R['Record quality'] = 30.0
        why['Record quality'] = ['Likely a secondary injection test, not a system fault: ' + '; '.join(sg) + '. Keep it out of the fault statistics.']
        out['Mode Risks'] = R; out['Reasons'] = why
        return risk(out)
    f0 = lambda v: '-' if v is None else f"{v:.0f}"
    elems = [e[1] for e in out['elements']]
    inst = [e for e in elems if re.search(r'zone 1|87l|differential|sotf|aided|transfer', e.lower())]
    loc = out.get('loc_pu')
    # protection operation
    s, w = 0.0, []
    op = out.get('operate_ms')
    if op is None:
        s = 40.0; w.append('No trip found in the record channels or the event list: the operating element and time cannot be checked.')
    else:
        w.append(f"Protection operated in {f0(op)} ms" + (f" ({', '.join(elems)})." if elems else '.'))
        if inst:
            s = max(s, lin(op, st['expected_inst_ms'], st['expected_inst_ms'] * 2.5, 0, 70))
            if op > st['expected_inst_ms']:
                w.append(f"Slower than the {st['expected_inst_ms']:.0f} ms expected for an instantaneous element.")
        elif any('zone 2' in e.lower() for e in elems):
            s = max(s, lin(op, st['z2_time_ms'] * 1.1, st['z2_time_ms'] * 1.6, 0, 60))
            w.append(f"Cleared by zone 2 (time-delayed, about {st['z2_time_ms']:.0f} ms).")
            if loc is not None and loc < st['z1_reach'] - 0.05:
                s = max(s, 60.0); w.append(f"The fault is at {loc * 100:.0f}% of the line, inside the zone 1 reach ({st['z1_reach'] * 100:.0f}%): zone 1 should have tripped (possible under-reach).")
            else:
                s = max(s, 38.0); w.append('Fault beyond the zone 1 reach: zone 2 is correct, but the clearing is slow; an aided (carrier) scheme would clear it instantaneously.')
        if loc is not None:
            if loc > 1.05 and any('zone 1' in e.lower() for e in elems):
                s = max(s, 90.0); w.append(f"The fault is located beyond the remote end ({loc * 100:.0f}% of the line) but zone 1 tripped: OVER-REACH / mal-operation. The fault was external to this line.")
            elif loc > 1.05 and inst:
                s = max(s, 55.0); w.append(f"The fault is located beyond the remote end ({loc * 100:.0f}% of the line): check whether the instantaneous trip was correct (aided scheme / intertrip).")
            elif st['z1_reach'] < loc <= 1.05 and inst and not any('zone 1' in e.lower() for e in elems):
                w.append(f"The fault is at {loc * 100:.0f}% of the line, beyond the zone 1 reach ({st['z1_reach'] * 100:.0f}%): correctly cleared "
                         f"without delay by the aided scheme / line differential instead of waiting for zone 2.")
            elif st['z1_reach'] < loc <= 1.05 and any('zone 1' in e.lower() for e in elems):
                s = max(s, 40.0); w.append(f"Zone 1 tripped for a fault at {loc * 100:.0f}% of the line, at or beyond its {st['z1_reach'] * 100:.0f}% reach: check the reach setting and the line data.")
    zs = zone_seen(out)
    out['zone_seen'] = zs
    if zs is not None and out.get('Zapp') is not None:
        zt = f"R {out['Zapp'].real:.1f} Ω, X {out['Zapp'].imag:.1f} Ω"
        z1_trip = any('zone 1' in e.lower() for e in elems)
        if zs == 0:
            w.append(f"The impedance seen at the fault ({zt}) is outside all distance zones.")
            if any(re.search(r'zone [123]', e.lower()) for e in elems) and (st.get('r_ph') or st.get('r_gnd')):
                s = max(s, 45.0); w.append('A distance zone tripped although the impedance is outside every zone set in the relay: check the settings used and the VT/CT circuits.')
        else:
            w.append(f"The impedance seen at the fault ({zt}) lies inside zone {zs}" + (' (outside zone 1).' if zs > 1 else '.'))
            if zs == 1 and not inst:
                s = max(s, 70.0); w.append('The impedance entered zone 1 but no instantaneous trip followed: zone 1 FAILED TO OPERATE. Test the relay.')
            elif zs > 1 and z1_trip:
                s = max(s, 80.0); w.append(f"Zone 1 tripped although the impedance stayed in zone {zs}: zone 1 OVER-REACH. Check the zone 1 reach and kZN settings.")
    if out.get('three_pole_trip') and out['type'] in ('AG', 'BG', 'CG'):
        w.append('Three-pole trip for a single-phase fault (forced three-pole trip, three-pole AR mode or single-pole tripping not in service).')
    rv = out.get('relay_reports', {})
    for dev, v in rv.items():
        if v.get('Relay Trip Elements'):
            w.append(f"{dev} fault record: {v['Relay Trip Elements']}.")
    if not out.get('type_agree', True):
        w.append(f"The AI fault type ({out['type_ai']}) and the classic phase selection ({out['type_rule']}) disagree: confirm the faulted phases from the waveforms.")
    R['Protection operation'] = s; why['Protection operation'] = w
    rel = {}
    for dev, v in out.get('relay_reports', {}).items():
        m_ = re.match(r'(-?\d+(?:\.\d+)?)\s*km', str(v.get('Relay Fault Location', '')))
        if m_:
            rel[dev] = float(m_.group(1))
    if rel and out.get('loc_km') is not None:
        tol = max(3.0, 0.05 * out['line']['length_km'])
        diffs = {d: out['loc_km'] - x for d, x in rel.items()}
        txt = ', '.join(f"{d} {x:.1f} km" for d, x in rel.items())
        if all(abs(x) > tol for x in diffs.values()):
            out['notes'].append(f"FaultGuard location {out['loc_km']:.1f} km differs from the relays ({txt}): check the line data "
                                f"(length, line impedance and angle, kZN); mutual coupling of a parallel circuit and remote infeed through fault "
                                f"resistance also affect earth-fault location.")
        else:
            best = min(diffs, key=lambda d: abs(diffs[d]))
            out['notes'].append(f"Fault location: FaultGuard {out['loc_km']:.1f} km; relays {txt}. Closest to {best} "
                                f"(difference {abs(diffs[best]):.1f} km, {abs(diffs[best]) / out['line']['length_km'] * 100:.1f}% of the line).")
    # breaker
    s, w = 0.0, []
    bt = out.get('breaker_ms')
    if bt is not None:
        s = lin(bt / st['cb_rated_ms'], 1.2, 2.0, 0, 90)
        w.append(f"Breaker interrupted the current {f0(bt)} ms after the trip (rated about {st['cb_rated_ms']:.0f} ms" + (', SLOW).' if bt > 1.2 * st['cb_rated_ms'] else ').'))
    elif out.get('t_interruption_ms') is None:
        s = 80.0; w.append('The fault current is not interrupted within the record: check the breaker and the breaker-failure protection.')
    else:
        w.append('Breaker time not measurable (no trip time).')
    duty = out.get('I_fault_max_kA', 0) / st['cb_rating_ka'] if st['cb_rating_ka'] else 0
    if duty > 0.6:
        s = max(s, lin(duty, 0.6, 1.0, 20, 80)); w.append(f"Interrupted {out['I_fault_max_kA']:.1f} kA, {duty * 100:.0f}% of the {st['cb_rating_ka']:.0f} kA breaker rating.")
    R['Breaker performance'] = s; why['Breaker performance'] = w
    # fault severity
    s, w = 0.0, []
    cl = out.get('clearing_ms')
    if cl is not None:
        delayed = any(re.search(r'zone 2|zone 3|overcurrent|earth fault', e.lower()) for e in elems) and not inst
        if delayed:
            s = max(s, 45.0 if cl > st['critical_ms'] else 0.0)
            w.append(f"Total clearing time {f0(cl)} ms by a time-delayed element (backup clearing; the {st['critical_ms']:.0f} ms limit applies to the main protection).")
        else:
            s = max(s, lin(cl / st['critical_ms'], 0.8, 1.5, 0, 90))
            w.append(f"Total clearing time {f0(cl)} ms (limit {st['critical_ms']:.0f} ms).")
    w.append(f"Fault current {out.get('I_fault_max_kA', 0):.1f} kA rms (peak {out.get('I_peak_kA', 0):.1f} kA with DC offset).")
    if out.get('V_min_pu') is not None:
        s = max(s, lin(1 - out['V_min_pu'], 0.5, 0.95, 0, 45)); w.append(f"Lowest phase voltage {out['V_min_pu'] * 100:.0f}% of pre-fault.")
    if out['type'] == 'ABC':
        s = max(s, 40.0); w.append('Three-phase fault: the most severe fault for system stability.')
    R['Fault severity'] = s; why['Fault severity'] = w
    # permanent fault
    s, w = 0.0, []
    ar = out.get('ar', '')
    if ar.startswith('Unsuccessful'):
        s = 75.0; w.append(f"Auto-reclose {ar.lower()}" + (f"; the line locked out." if 'lockout' not in ar.lower() else '.'))
    elif ar.startswith('Successful'):
        w.append('Auto-reclose successful: transient fault' + (f" (dead time {out['dead_time_ms']:.0f} ms)." if out.get('dead_time_ms') else '.'))
    else:
        evs_ = ' '.join(out['events']['Event'].astype(str).str.lower()) if out.get('events') is not None else ''
        blocked = bool(re.search(r'ar\s*block|block\w*\s*(ar|a/r|reclos)|79\s*block|reclos\w*\s*block', evs_))
        delayed = any(re.search(r'zone 2|zone 3|overcurrent|earth fault', e.lower()) for e in elems) and not inst
        if out['type'] in ('AG', 'BG', 'CG') and st['ar_expected'] and out.get('t_interruption_ms') is not None and not blocked and not delayed:
            s = 40.0
            if out.get('ar_initiated'):
                rs = out.get('restored')
                out.setdefault('mode_action', {})['Permanent fault'] = (
                    'Confirm whether auto-reclose is in service on this line and why it did not reclose (AR relay / bay controller, AR mode, '
                    'forced three-pole trip); patrol the line around the located distance to confirm the fault was transient.')
                w.append('Auto-reclose was initiated' + (' (three-pole)' if out.get('three_pole_trip') else '') + ' but no reclose followed' +
                         (f"; the line was re-energised {fmt_dur(rs['after_s'])} later ({rs['device']}: {rs['event']}), which points to a manual close."
                          if rs and rs['after_s'] > 30 else '.') +
                         ' Confirm whether auto-reclose is in service on this line, check the AR relay / bay controller, and confirm the fault was transient (line patrol).')
            else:
                w.append(f"No reclosing seen for a single-phase fault ({ar.lower()}): check the auto-reclose scheme, or the record is too short.")
        elif not st['ar_expected']:
            rs = out.get('restored')
            w.append('Auto-reclose not assessed: the external auto-recloser is kept out of service' +
                     (f"; the line was re-energised {fmt_dur(rs['after_s'])} after the fault ({rs['device']}: {rs['event']})." if rs else '.'))
        else:
            w.append(f"Auto-reclose: {ar.lower()}" + (' (normally blocked for multi-phase faults).' if out['type'] in ('ABC',) else '.'))
    R['Permanent fault'] = s; why['Permanent fault'] = w
    # record quality
    s, w = 0.0, []
    t0 = out.get('record_start')
    if t0 is None or t0.year < 2000 or t0 > datetime.now() + timedelta(days=1):
        s = 30.0; w.append(f"Relay clock not synchronised (record dated {t0 if t0 else 'unknown'}).")
    ch = out['channels']
    miss = [k for k in ('IA', 'IB', 'IC', 'VA', 'VB', 'VC') if ch.get(k) is None]
    if miss:
        s = max(s, 30.0); w.append(f"Channels not found: {', '.join(miss)}.")
    if out.get('P') and 'IN' in out['P'] and all(k in out['P'] for k in ('IA', 'IB', 'IC')):
        P = out['P']; s3 = P['IA'][1] + P['IB'][1] + P['IC'][1]; imax = max(abs(P[k][1]) for k in ('IA', 'IB', 'IC'))
        mis = abs(P['IN'][1] - s3) / imax if imax else 0
        if mis > 0.05:
            s = max(s, lin(mis, 0.05, 0.25, 30, 60)); w.append(f"Measured neutral differs from the sum of the phases by {mis * 100:.0f}% of the fault current: check the CT circuit.")
    if out.get('type_conf', 1) < 0.6:
        s = max(s, 25.0); w.append(f"Fault type confidence only {out['type_conf'] * 100:.0f}%.")
    for dev, v in out.get('relay_reports', {}).items():
        sh = out['clock_shift'].get(dev, 0)
        off = v.get('Clock vs record (ms)')
        if sh:
            w.append(f"{dev} event times are {abs(sh)} h {'behind' if sh > 0 else 'ahead of'} the record (relay clock on another time zone, e.g. UTC); shifted to match.")
        if off is not None and abs(off) > 20:
            s = max(s, 30.0); w.append(f"{dev} clock differs from the record by about {off:.0f} ms" + (' after the time-zone shift' if sh else '') +
                                       ": the relays are not time-synchronised (check GPS / SNTP / IRIG-B). Its events were aligned to the record using its own fault duration.")
    w = w or ['Record complete: all phase currents and voltages present.']
    R['Record quality'] = s; why['Record quality'] = w
    out['Mode Risks'] = {k: round(float(v), 1) for k, v in R.items()}
    out['Reasons'] = why
    return risk(out)


def risk(out):
    risks = out['Mode Risks']
    gov = max(MODE_ORDER, key=lambda k: risks[k])
    others = sum(1 for k in MODE_ORDER if k != gov and risks[k] > EXCELLENT_MAX)
    overall = round(min(100.0, risks[gov] + 5 * others), 1)
    S = MODES[gov]['S']; O = max(1, math.ceil(overall / 10)); rpn = S * O * DETECTION
    fmea = []
    for k in sorted(MODE_ORDER, key=lambda k: -risks[k]):
        o = max(1, math.ceil(risks[k] / 10)); r = MODES[k]['S'] * o * DETECTION
        fmea.append({'Failure Mode': k, 'Risk (%)': risks[k], 'S': MODES[k]['S'], 'O': o, 'D': DETECTION, 'RPN': r, 'RPN Level': rpn_level(r)})
    st = status_of(overall)
    acts = [out.get('mode_action', {}).get(k, ACTIONS[k]) for k in sorted(MODE_ORDER, key=lambda k: -risks[k]) if risks[k] > EXCELLENT_MAX]
    out.update({'Governing Mode': gov, 'Other Modes Above Normal': others, 'Overall Risk (%)': overall, 'Status': st, 'S': S, 'O': O,
                'D': DETECTION, 'RPN': rpn, 'RPN Level': rpn_level(rpn), 'FMEA': fmea,
                'Action': ('Test record: file it with the test results, not as an incident.' if out.get('likely_test') else
                           ' '.join(acts[:2]) if acts else 'Correct protection and breaker operation: file the report; no action needed.')})
    return out


def findings(a):
    r = []
    for k, v in sorted(a['Mode Risks'].items(), key=lambda kv: -kv[1]):
        if a['Reasons'].get(k) and (k == a['Governing Mode'] or v > EXCELLENT_MAX):
            r.append(f"{k} ({v:.0f}%): " + ' '.join(a['Reasons'][k]))
    n = a.get('Other Modes Above Normal', 0)
    if n:
        r.append(f"{n} other aspect(s) above normal add {5 * n} points to the overall risk.")
    return r


def summary_text(a):
    if not a.get('fault'):
        return 'No fault found in the record.'
    if a.get('likely_test'):
        return ('Likely a secondary injection test, not a system fault: ' + '; '.join(a['test_signs']) + '.' +
                (f" Protection operated {a['operate_ms']:.0f} ms after the injection started." if a.get('operate_ms') is not None else ''))
    loc = f"{a['loc_km']:.1f} km ({a['loc_pu'] * 100:.0f}% of the line)" if a.get('loc_km') is not None else 'not located'
    inc = a.get('t_inception_ms') or 0
    els = ', '.join(e[1] + (f" ({e[0] - inc:.0f} ms)" if e[0] is not None else '') for e in a['elements']) or 'element not identified'
    s = (f"{TYPE_TEXT.get(a['type'], a['type'])} fault, {a['I_fault_max_kA']:.1f} kA, located at {loc}. "
         f"Cleared by {els}" + (f" in {a['clearing_ms']:.0f} ms" if a.get('clearing_ms') else '') +
         (f" (first trip {a['operate_ms']:.0f} ms" + (f" by {a['trip_by']}" if a.get('trip_by') not in (None, 'record') else '') +
          f" + breaker {a['breaker_ms']:.0f} ms)" if a.get('operate_ms') is not None and a.get('breaker_ms') is not None else '') +
         f". Auto-reclose: {a['ar'].lower()}.")
    return s


def soe(a):
    """Sequence of events: record binary changes plus the event list, on one time line (ms from the record start)."""
    rows = []
    t0 = a.get('record_start')
    if a.get('fault'):
        rows.append((a['t_inception_ms'], 'Record', 'Fault inception (from the currents)'))
        if a.get('t_interruption_ms') is not None:
            rows.append((a['t_interruption_ms'], 'Record', 'Fault current interrupted'))
        if a.get('t_reclose_ms') is not None:
            rows.append((a['t_reclose_ms'], 'Record', 'Line re-energised (current returns)'))
        if a.get('t_refault_ms') is not None:
            rows.append((a['t_refault_ms'], 'Record', 'Fault reappears after reclosing'))
    for tm, name, v in a.get('binary', []):
        rows.append((tm * 1000, 'Record binary', f"{name.strip()} {'ON' if v else 'OFF'}"))
    E = a.get('events')
    if E is not None and t0 is not None:
        for _, r in E.iterrows():
            txt = str(r['Event'])
            if not re.search(r'\b(on|off)\s*$', txt.lower()):
                on = on_items(r.get('Detail', ''))
                if on:
                    txt += ': ' + ', '.join(on[:8]) + (' ON' if len(on) else '')
            rows.append(((pd.Timestamp(r['Timestamp']) - pd.Timestamp(t0)).total_seconds() * 1000, str(r.get('Device') or 'Event list'), txt))
    df = pd.DataFrame(rows, columns=['Time (ms)', 'Source', 'Event']).sort_values('Time (ms)', kind='stable')
    if t0 is not None:
        df.insert(0, 'Clock Time', [(pd.Timestamp(t0) + pd.Timedelta(milliseconds=float(x))).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for x in df['Time (ms)']])
    df['Time (ms)'] = df['Time (ms)'].round(1)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- simulation (demo incidents and AI training)
def network_fault(ftype, m, Rf, sysp, rng=None):
    """Phasors (primary) at the local relay for a fault at m (per unit of the line; m > 1 = beyond the remote bus).
    sysp: kv, length_km, line per km r1 x1 r0 x0, source impedances zs1, zs0, zr1, zr0 (ohm), delta_deg (load angle)."""
    L = sysp['length_km']
    ZL1 = complex(sysp['r1'], sysp['x1']) * L; ZL0 = complex(sysp['r0'], sysp['x0']) * L
    Zs = {1: sysp['zs1'], 2: sysp['zs1'], 0: sysp['zs0']}; Zr = {1: sysp['zr1'], 2: sysp['zr1'], 0: sysp['zr0']}
    ZL = {1: ZL1, 2: ZL1, 0: ZL0}
    Vph = sysp['kv'] * 1e3 / np.sqrt(3)
    Es = Vph; Er = Vph * np.exp(-1j * np.deg2rad(sysp.get('delta_deg', 10)))
    Il = (Es - Er) / (Zs[1] + ZL1 + Zr[1])
    if m <= 1:
        a_k = {k: Zs[k] + m * ZL[k] for k in (0, 1, 2)}; b_k = {k: Zr[k] + (1 - m) * ZL[k] for k in (0, 1, 2)}
        Vf_pre = Es - (Zs[1] + m * ZL1) * Il
    else:                                   # beyond the remote bus on an adjacent line of the same impedance
        ext = (m - 1) * ZL1; ext0 = (m - 1) * ZL0
        a_k = {k: Zs[k] + ZL[k] + (ext if k else ext0) for k in (0, 1, 2)}; b_k = {k: Zr[k] * 1.0 + 1e-6 for k in (0, 1, 2)}
        b_k = {k: Zr[k] for k in (0, 1, 2)}
        Vf_pre = Es - (Zs[1] + ZL1) * Il
    Zth = {k: a_k[k] * b_k[k] / (a_k[k] + b_k[k]) for k in (0, 1, 2)}
    if m <= 1:
        D = {k: b_k[k] / (a_k[k] + b_k[k]) for k in (0, 1, 2)}
    else:
        # the remote bus splits the fault current between the line (from the local source) and the remote source
        zl = {k: Zs[k] + ZL[k] for k in (0, 1, 2)}; zr = {k: Zr[k] for k in (0, 1, 2)}
        # current from the remote bus to the fault, share from the local side through the line
        D = {k: zr[k] / (zl[k] + zr[k]) for k in (0, 1, 2)}
        Zth = {k: (m - 1) * (ZL1 if k else ZL0) + zl[k] * zr[k] / (zl[k] + zr[k]) for k in (0, 1, 2)}
    Zabc = AMAT @ np.diag([Zth[0], Zth[1], Zth[2]]) @ AINV
    Vth = Vf_pre * np.array([1, A_OP ** 2, A_OP])
    g = 1 / max(Rf, 1e-3); gpp = 1 / max(min(Rf, 2.0), 1e-3)
    Y = np.zeros((3, 3), complex)
    ph = {'A': 0, 'B': 1, 'C': 2}
    ground = ftype.endswith('G')
    phases = [ph[c] for c in ftype.replace('G', '')]
    if ftype == 'ABC':
        for i in range(3):
            for j in range(3):
                Y[i, j] = (2 * gpp if i == j else -gpp)
    else:
        if len(phases) == 2:
            i, j = phases
            Y[i, i] += gpp; Y[j, j] += gpp; Y[i, j] -= gpp; Y[j, i] -= gpp
        if ground:
            if len(phases) == 1:
                Y[phases[0], phases[0]] += g
            else:
                # two phases to ground through Rf (common), phases bonded by low resistance
                for i_ in phases:
                    Y[i_, i_] += g / 2
    If = np.linalg.solve(np.eye(3) + Y @ Zabc, Y @ Vth)
    Ifs = AINV @ If
    dIs = np.array([D[k] * Ifs[k] for k in (0, 1, 2)])
    Ipre_s = np.array([0, Il, 0]); Vpre_loc_s = np.array([0, Es - Zs[1] * Il, 0])
    Iloc_s = Ipre_s + dIs
    Vloc_s = Vpre_loc_s - np.array([Zs[0] * dIs[0], Zs[1] * dIs[1], Zs[2] * dIs[2]])
    return {'I_pre': AMAT @ Ipre_s, 'V_pre': AMAT @ Vpre_loc_s, 'I_f': AMAT @ Iloc_s, 'V_f': AMAT @ Vloc_s, 'tau': abs((Zs[1] + m * ZL1).imag / (Zs[1] + m * ZL1).real) / (2 * np.pi * 50)}


def random_system(rng, kv=None):
    kv = kv or rng.choice([132.0, 230.0, 400.0])
    L = float(rng.uniform(20, 200))
    x1 = {132.0: 0.40, 230.0: 0.38, 400.0: 0.32}[kv] * rng.uniform(0.9, 1.1)
    line = {'length_km': L, 'r1': x1 * rng.uniform(0.06, 0.12), 'x1': x1, 'r0': x1 * rng.uniform(0.5, 0.9), 'x0': x1 * rng.uniform(2.6, 3.4), 'kv': kv}
    Vb = kv * 1e3
    sc_s = rng.uniform(3, 40) * 1e3 * kv / np.sqrt(3) * np.sqrt(3) / 1e3     # ~ MVA scaled
    zs = Vb ** 2 / (rng.uniform(2000, 25000) * 1e6)
    zr = Vb ** 2 / (rng.uniform(1000, 25000) * 1e6)
    xr = rng.uniform(8, 25)
    sysp = dict(line, zs1=complex(zs / xr, zs), zs0=complex(zs / xr, zs) * rng.uniform(0.6, 1.5), zr1=complex(zr / xr, zr),
                zr0=complex(zr / xr, zr) * rng.uniform(0.6, 1.5), delta_deg=float(rng.uniform(-15, 25)))
    return sysp, line


def waveforms(sim, fs=2400.0, f0=50.0, dur=1.0, t_inc=0.1, op_ms=20.0, cb_ms=40.0, ar_dead_ms=None, permanent=False, ar_op_ms=20.0,
              inc_angle=None, rng=None, noise=0.003):
    """Time series of 3 currents, neutral and 3 voltages for the simulated fault (with DC offset, breaker opening at current zeros,
    optional reclosing). Returns t, I (3 x n), IN, V (3 x n), and the event times."""
    rng = rng or np.random.default_rng(0)
    n = int(dur * fs); t = np.arange(n) / fs; w = 2 * np.pi * f0
    if inc_angle is not None:
        t_inc = t_inc + (inc_angle / (2 * np.pi * f0))
    cyc = lambda P, ph: np.sqrt(2) * np.abs(P[ph]) * np.cos(w * t + np.angle(P[ph]))
    I = np.zeros((3, n)); V = np.zeros((3, n))
    t_trip = t_inc + op_ms / 1000; t_cb = t_trip + cb_ms / 1000
    tau = max(0.01, min(sim['tau'], 0.12))
    segs = [(t_inc, t_cb)]
    t_close = None
    if ar_dead_ms:
        t_close = t_cb + ar_dead_ms / 1000
        if permanent:
            segs.append((t_close, t_close + (ar_op_ms + cb_ms) / 1000))
    for p in range(3):
        ip, i_f = cyc(sim['I_pre'], p), cyc(sim['I_f'], p)
        vp, vf = cyc(sim['V_pre'], p), cyc(sim['V_f'], p)
        cur = ip.copy(); vol = vp.copy()
        state_open = np.zeros(n, bool)
        for k, (a, b) in enumerate(segs):
            ia = int(a * fs)
            if ia >= n:
                continue
            ref = cur[ia]
            seg = np.arange(ia, n)
            dc = (ref - i_f[ia]) * np.exp(-(t[seg] - t[ia]) / tau)
            cur[seg] = i_f[seg] + dc; vol[seg] = vf[seg]
            # breaker interrupts at the first current zero after b
            ib = int(b * fs)
            if ib < n:
                z = np.where(np.sign(cur[ib:-1]) != np.sign(cur[ib + 1:]))[0]
                iz = ib + (int(z[0]) + 1 if len(z) else 0)
                cur[iz:] = 0.0; vol[iz:] = vol[iz:] * 0.0
                state_open[iz:] = True
        if t_close is not None and not permanent:
            ic = int(t_close * fs)
            if ic < n:
                cur[ic:] = ip[ic:]; vol[ic:] = vp[ic:]
        elif t_close is not None and permanent:
            pass
        I[p] = cur + rng.normal(0, noise * np.abs(sim['I_f']).max() * np.sqrt(2), n)
        V[p] = vol + rng.normal(0, noise * np.abs(sim['V_pre']).max() * np.sqrt(2), n)
    IN = I.sum(0) + rng.normal(0, noise * np.abs(sim['I_f']).max(), n)
    return t, I, IN, V, {'t_inc': t_inc, 't_trip': t_trip, 't_cb': t_cb, 't_close': t_close}


def to_comtrade(station, device, t, I, IN, V, digital, start, f0=50.0, fs=2400.0):
    """ASCII COMTRADE 1999 (cfg text, dat bytes); analog in primary A and kV; digital = list of (name, 0/1 array)."""
    an = [('IA', 'A', I[0]), ('IB', 'A', I[1]), ('IC', 'A', I[2]), ('IN', 'A', IN), ('VA', 'kV', V[0] / 1000), ('VB', 'kV', V[1] / 1000), ('VC', 'kV', V[2] / 1000)]
    sc = [max(np.abs(x).max(), 1e-6) / 32000 for _, _, x in an]
    lines = [f"{station},{device},1999", f"{len(an) + len(digital)},{len(an)}A,{len(digital)}D"]
    for k, ((nm, u, x), a) in enumerate(zip(an, sc)):
        lines.append(f"{k + 1},{nm},,,{u},{a:.9g},0,0,-32767,32767,1,1,P")
    for k, (nm, _) in enumerate(digital):
        lines.append(f"{k + 1},{nm},,,0")
    lines += [f"{f0:g}", "1", f"{fs:g},{len(t)}", start.strftime('%d/%m/%Y,%H:%M:%S.%f'),
              (start + timedelta(seconds=0.1)).strftime('%d/%m/%Y,%H:%M:%S.%f'), "ASCII", "1"]
    q = [np.round(x / a).astype(int) for (_, _, x), a in zip(an, sc)]
    rows = [f"{i + 1},{int(round(t[i] * 1e6))}," + ','.join(str(c[i]) for c in q) + (',' + ','.join(str(int(d[i])) for _, d in digital) if digital else '')
            for i in range(len(t))]
    return '\n'.join(lines), ('\n'.join(rows)).encode()


DEMO_STATION = 'DEMO SUBSTATION NORTH'
DEMO_INCIDENTS = [
    # id, line, partner, length km, type, m, Rf, element, op ms, cb ms, AR dead ms, permanent, date
    ('INC-01', 'Line 1', 'Station East', 120.0, 'AG', 0.32, 2.0, 'Z1', 22, 42, 800, False, '2026-03-04 14:22:05.310'),
    ('INC-02', 'Line 2', 'Station West', 85.0, 'BC', 0.71, 0.5, '87L', 28, 40, None, False, '2026-04-11 02:47:39.880'),
    ('INC-03', 'Line 1', 'Station East', 120.0, 'BCG', 0.05, 1.0, 'Z1', 18, 41, 800, True, '2026-05-19 16:05:12.044'),
    ('INC-04', 'Line 3', 'Station South', 140.0, 'AG', 0.88, 35.0, 'Z2', 410, 43, None, False, '2026-06-02 11:30:58.502'),
    ('INC-05', 'Line 2', 'Station West', 85.0, 'CG', 0.45, 3.0, 'Z1', 25, 96, 800, False, '2026-07-08 19:12:44.731'),
    ('INC-06', 'Line 4', 'Station North-East', 60.0, 'AG', 1.25, 1.0, 'Z1', 24, 41, 800, False, '2026-08-14 07:58:21.190'),
    ('INC-07', 'Line 3', 'Station South', 140.0, 'ABC', 0.20, 0.5, 'Z1', 19, 40, None, False, '2026-09-22 21:40:03.615'),
]
DEMO_SYS = {'kv': 400.0, 'r1': 0.028, 'x1': 0.32, 'r0': 0.21, 'x0': 1.02, 'zs1': complex(0.9, 12.0), 'zs0': complex(1.4, 16.0),
            'zr1': complex(1.2, 15.0), 'zr0': complex(2.0, 22.0), 'delta_deg': 12.0}


def demo_incident(i, seed=5):
    """(meta, cfg_text, dat_bytes, events DataFrame, line) of demo incident i."""
    iid, ln, partner, L, ft, m, Rf, el, op, cb, ar, perm, when = DEMO_INCIDENTS[i]
    rng = np.random.default_rng(seed + i)
    sysp = dict(DEMO_SYS, length_km=L)
    sim = network_fault(ft, m, Rf, sysp)
    start = datetime.strptime(when, '%Y-%m-%d %H:%M:%S.%f') - timedelta(seconds=0.1)
    dur = 1.5 if ar else 0.8
    t, I, IN, V, tm = waveforms(sim, dur=dur, op_ms=op, cb_ms=cb, ar_dead_ms=ar, permanent=perm, rng=rng, inc_angle=rng.uniform(0, 2 * np.pi))
    n = len(t)
    on = lambda a, b=None: ((t >= a) & (t < (b if b is not None else 99))).astype(int)
    trip_on = on(tm['t_trip'], tm['t_cb'] + 0.02)
    names = {'Z1': 'DIST Z1 TRIP', 'Z2': 'DIST Z2 TRIP', '87L': '87L DIFF TRIP'}
    dig = [('ANY TRIP', trip_on), (names[el], trip_on), ('START A', on(tm['t_inc'] + 0.006, tm['t_cb'] + 0.01) if 'A' in ft.replace('G', '') or ft == 'ABC' else np.zeros(n, int)),
           ('START B', on(tm['t_inc'] + 0.006, tm['t_cb'] + 0.01) if 'B' in ft.replace('G', '') or ft == 'ABC' else np.zeros(n, int)),
           ('START C', on(tm['t_inc'] + 0.006, tm['t_cb'] + 0.01) if 'C' in ft.replace('G', '') or ft == 'ABC' else np.zeros(n, int)),
           ('START N', on(tm['t_inc'] + 0.006, tm['t_cb'] + 0.01) if ft.endswith('G') else np.zeros(n, int))]
    cb_open = on(tm['t_cb'] + 0.004)
    if ar:
        cb_open = cb_open * (1 - on(tm['t_close']))
        if perm:
            cb_open = np.maximum(cb_open, on(tm['t_close'] + (20 + cb) / 1000 + 0.004))
    dig.append(('CB OPEN 52b', cb_open))
    if ar:
        dig.append(('AR IN PROGRESS', on(tm['t_cb'], tm['t_close'] + 0.05)))
        dig.append(('AR CLOSE', on(tm['t_close'] - 0.06, tm['t_close'] + 0.04)))
        if perm:
            sotf = on(tm['t_close'] + 0.02, tm['t_close'] + 0.02 + (cb + 20) / 1000)
            dig.append(('SOTF TRIP', sotf))
            dig[0] = ('ANY TRIP', np.maximum(trip_on, sotf))
    station = DEMO_STATION; device = f"{ln.upper()} MAIN 1"
    cfg, dat = to_comtrade(station, device, t, I, IN, V, dig, start)
    # event list in the style of a relay event export
    T = lambda s: start + timedelta(seconds=s)
    phases = ft.replace('G', '') if ft != 'ABC' else 'ABC'
    ev = [(T(tm['t_inc'] + 0.006), f"Started Phase {'/'.join(phases)}{' N' if ft.endswith('G') else ''} ON"),
          (T(tm['t_trip']), f"{names[el].title().replace('Dist', 'Distance').replace('87L Diff', '87L Differential')} ON"),
          (T(tm['t_trip']), 'Any Trip ON'), (T(tm['t_trip'] + 0.001), 'Trip A ON  Trip B ON  Trip C ON'),
          (T(tm['t_cb'] + 0.004), 'CB Open (52b) ON')]
    loc_rel = m * L * (1 + rng.normal(0, 0.03))
    ev.append((T(tm['t_cb'] + 0.05), f"Fault Record: Faulted Phase {phases}{'-N' if ft.endswith('G') else ''}  Fault Location {loc_rel:.1f} km  "
                                     f"Relay Trip Time {op} ms  Fault Duration {op + cb} ms"))
    if ar:
        ev += [(T(tm['t_cb'] + 0.01), 'AR In Progress ON'), (T(tm['t_close'] - 0.06), 'AR Close Command ON'), (T(tm['t_close'] + 0.004), 'CB Closed ON')]
        if perm:
            ev += [(T(tm['t_close'] + 0.02), 'SOTF Trip ON'), (T(tm['t_close'] + (20 + cb) / 1000 + 0.004), 'CB Open (52b) ON'),
                   (T(tm['t_close'] + 0.2), 'AR Lockout ON')]
        else:
            ev.append((T(tm['t_close'] + 0.3), 'AR Successful'))
    elif ft == 'ABC' or el == 'Z2':
        ev.append((T(tm['t_cb'] + 0.1), 'AR Blocked (multi-phase / time-delayed trip)'))
    E = pd.DataFrame(sorted(ev), columns=['Timestamp', 'Event'])
    meta = {'Incident': iid, 'Station': station.title(), 'Line': f"{ln} to {partner}", 'Relay': device.title(), 'Date': when[:19]}
    line = dict(DEMO_SYS, length_km=L)
    return meta, cfg, dat, E, {k: line[k] for k in ('length_km', 'r1', 'x1', 'r0', 'x0', 'kv')}


def events_to_text(E):
    return '\n'.join(f"{pd.Timestamp(r['Timestamp']).strftime('%d/%m/%Y %H:%M:%S.%f')[:-3]}  {r['Event']}" for _, r in E.iterrows())


# ---------------------------------------------------------------- AI model training data
def _saturate(x, rng):
    """Crude CT saturation: the secondary current collapses towards zero once the flux (integral) passes a knee."""
    flux = np.cumsum(x); knee = np.abs(flux).max() * rng.uniform(0.25, 0.6)
    y = x.copy(); y[np.abs(flux) > knee] *= rng.uniform(0.1, 0.5)
    return y


def training_records(n, seed=0, fs=2400.0, hard=True):
    """Simulated incidents through the same pipeline as real records: features and true fault type.
    hard=True adds weak infeed, high fault resistance, harmonics, CT saturation and more noise."""
    rng = np.random.default_rng(seed)
    X, y, loc_err, flags = [], [], [], []
    for k in range(n):
        ft = FAULT_TYPES[k % len(FAULT_TYPES)]
        sysp, line = random_system(rng)
        if hard and rng.random() < 0.25:                       # weak infeed at the relay end
            sysp['zs1'] *= rng.uniform(4, 12); sysp['zs0'] *= rng.uniform(4, 12)
        m = float(rng.uniform(0.02, 0.98))
        if ft.endswith('G'):
            Rf = float(rng.choice([rng.uniform(0, 3), rng.uniform(3, 60), rng.uniform(60, 150)], p=[0.5, 0.35, 0.15] if hard else [0.6, 0.4, 0.0]))
        else:
            Rf = float(rng.uniform(0, 8 if hard else 3))
        sim = network_fault(ft, m, Rf, sysp)
        t, I, IN, V, tm = waveforms(sim, fs=fs, dur=0.3, t_inc=0.06, op_ms=float(rng.uniform(15, 40)), cb_ms=float(rng.uniform(35, 60)),
                                    rng=rng, inc_angle=rng.uniform(0, 2 * np.pi), noise=float(rng.uniform(0.001, 0.02 if hard else 0.008)))
        sat = False
        if hard:
            w = 2 * np.pi * 50 * t
            for p in range(3):
                h = rng.uniform(0, 0.04)
                I[p] = I[p] * (1 + h * np.cos(3 * w + rng.uniform(0, 6.3))); V[p] = V[p] * (1 + rng.uniform(0, 0.02) * np.cos(5 * w))
            if rng.random() < 0.15:
                sat = True
                for p in range(3):
                    I[p] = _saturate(I[p], rng)
        A = np.column_stack([I[0], I[1], I[2], IN, V[0], V[1], V[2]])
        rec = {'t': t, 'freq': 50.0}
        ch = {'IA': 0, 'IB': 1, 'IC': 2, 'IN': 3, 'VA': 4, 'VB': 5, 'VC': 6}
        ev = detect(rec, A, ch)
        if ev['inception'] is None:
            continue
        P = fault_phasors(A, ch, ev, ev['N'])
        X.append(features_from_phasors(P)); y.append(ft); flags.append(rule_type(P))
        x, _ = locate(P, ft, line)
        loc_err.append((ft, m, Rf, sat, (x / line['length_km'] - m) * 100 if x is not None else np.nan))
    le = pd.DataFrame(loc_err, columns=['Type', 'm', 'Rf', 'CT saturation', 'Error (% of line)'])
    le['Rule type'] = flags
    return np.array(X), np.array(y), le


def train_model(n_train=4000, n_test=1000, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    Xtr, ytr, _ = training_records(n_train, seed)
    Xte, yte, le = training_records(n_test, seed + 99)
    m = RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    pr = m.predict(Xte)
    rule_acc = float((le['Rule type'].to_numpy() == yte).mean())
    hi = le['Rf'].to_numpy() > 60
    sat = le['CT saturation'].to_numpy()
    return {'model': m, 'features': FEATURE_NAMES, 'version': 1, 'n_train': int(len(ytr)), 'n_test': int(len(yte)),
            'accuracy': float(accuracy_score(yte, pr)), 'labels': FAULT_TYPES, 'rule_accuracy': rule_acc,
            'acc_high_rf': float((pr[hi] == yte[hi]).mean()) if hi.any() else None, 'acc_ct_sat': float((pr[sat] == yte[sat]).mean()) if sat.any() else None,
            'n_high_rf': int(hi.sum()), 'n_ct_sat': int(sat.sum()),
            'confusion': confusion_matrix(yte, pr, labels=FAULT_TYPES).tolist(),
            'loc_err': le.groupby('Type')['Error (% of line)'].apply(lambda s: float(np.nanmedian(np.abs(s)))).round(2).to_dict(),
            'loc_err_all': float(np.nanmedian(np.abs(le['Error (% of line)']))),
            'loc_err_p90': float(np.nanpercentile(np.abs(le['Error (% of line)']), 90)),
            'loc_err_lowrf': float(np.nanmedian(np.abs(le.loc[le['Rf'] < 10, 'Error (% of line)'])))}


def load_model(path='faultguard_model.pkl'):
    import pickle
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------- one incident from many relays
def _seq(P, kind, idx, which=1):
    ks = [kind + p for p in 'ABC']
    if not all(k in P for k in ks):
        return None
    return (AINV @ np.array([P[k][which] for k in ks]))[idx]


def two_ended_location(aS, aR, line):
    """Unsynchronised two-ended location from the negative-sequence (unbalanced faults) or superimposed positive-sequence
    phasors of the two line ends: |V_S - m Z I_S| = |V_R - (1 - m) Z I_R|. Independent of fault resistance and infeed."""
    PS, PR = aS.get('P') or {}, aR.get('P') or {}
    L = line['length_km']; Z = complex(line['r1'], line['x1']) * L
    if aS.get('type') == 'ABC':
        VS, IS = _seq(PS, 'V', 1) - _seq(PS, 'V', 1, 0), _seq(PS, 'I', 1) - _seq(PS, 'I', 1, 0)
        VR, IR = _seq(PR, 'V', 1) - _seq(PR, 'V', 1, 0), _seq(PR, 'I', 1) - _seq(PR, 'I', 1, 0)
    else:
        VS, IS, VR, IR = _seq(PS, 'V', 2), _seq(PS, 'I', 2), _seq(PR, 'V', 2), _seq(PR, 'I', 2)
    if any(x is None for x in (VS, IS, VR, IR)):
        return None
    A, B = VS, Z * IS                       # |A - m B|^2
    C, D = VR - Z * IR, Z * IR              # |C + m D|^2
    a2 = abs(B) ** 2 - abs(D) ** 2
    a1 = -2 * (A * B.conjugate()).real - 2 * (C * D.conjugate()).real
    a0 = abs(A) ** 2 - abs(C) ** 2
    roots = [a0 / -a1] if abs(a2) < 1e-12 else list(np.roots([a2, a1, a0]))
    ok = [float(r.real) for r in np.atleast_1d(roots) if abs(np.imag(r)) < 1e-6 and -0.1 <= np.real(r) <= 1.1]
    if not ok:
        return None
    m = min(ok, key=lambda r: abs(r - 0.5))
    return m * L


def record_summary(name, a):
    f0 = lambda v: None if v is None else round(float(v), 1)
    return {'Record': name, 'Station': (a.get('station') or '-'), 'Device': (a.get('device') or '-'),
            'Fault Type': a.get('type', '-') if a.get('fault') else 'No fault', 'AI Confidence (%)': round(a.get('type_conf', 0) * 100) if a.get('fault') else None,
            'Location (km)': f0(a.get('loc_km')), 'Fault Current (kA)': f0(a.get('I_fault_max_kA')), 'Trip (ms)': f0(a.get('operate_ms')),
            'Breaker (ms)': f0(a.get('breaker_ms')), 'Clearing (ms)': f0(a.get('clearing_ms')),
            'Voltages': 'yes' if all(a.get('channels', {}).get(k) is not None for k in ('VA', 'VB', 'VC')) else 'no',
            'Status': ('TEST' if a.get('likely_test') else a.get('Status'))}


def analyze_incident(records, events=None, settings=None, line=None, setting=None, model=None, scale='Auto', ct=1.0, vt=1.0, meta=None,
                     use_settings=True):
    """One incident from any number of disturbance records, event lists and settings files.
    records: [(name, rec)], events: [(name, DataFrame)], settings: [(name, parsed dict)].
    Returns (analysis of the main record with the merged view, main record)."""
    line = dict(line or {}); setting = dict(setting or {}); notes = []
    sts = [(n, s) for n, s in (settings or []) if s]
    used = best_settings(sts) if (sts and use_settings) else None
    if used:
        ls, ss, nn = settings_line(used[1], ct, vt)
        line.update(ls); notes += nn
        ctr, vtr = ss.pop('_ct', None), ss.pop('_vt', None)
        setting.update(ss)
        if scale == 'Secondary' and ctr and vtr:
            ct, vt = ctr, vtr
        notes.append(f"Line data and settings taken from {used[1].get('device') or used[0]} settings ({used[0]}).")
        # consistency between the relays' settings
        for k, lab in (('line_length', 'line length'), ('line_z', 'line impedance'), ('line_angle', 'line angle'), ('kzn', 'kZN')):
            vals = {(s.get('device') or n): s[k] for n, s in sts if s.get(k) is not None}
            if len(vals) > 1 and (max(vals.values()) - min(vals.values())) > 0.02 * max(abs(v) for v in vals.values()):
                notes.append(f"The relays have different {lab} settings: " + ', '.join(f"{d} {v:g}" for d, v in vals.items()) + '. Check which one is correct.')
    res = []
    for name, rec in records:
        a = analyze(rec, events, line, setting, model=model, scale=scale, ct=ct, vt=vt, meta=meta)
        a['station'] = (rec.get('station') or '').strip(); a['device'] = (rec.get('device') or '').strip(); a['record_name'] = name
        res.append((name, rec, a))
    if not res:
        raise ValueError('No disturbance record.')
    score = lambda x: (bool(x[2].get('fault')), not x[2].get('likely_test'),
                       sum(x[2]['channels'].get(k) is not None for k in ('IA', 'IB', 'IC', 'VA', 'VB', 'VC', 'IN')), len(x[1]['digital']))
    name, rec, a = max(res, key=score)
    a['records'] = [record_summary(n, x) for n, _, x in res]
    a['settings_all'] = sts; a['settings_used'] = used
    a['notes'] = notes + a.get('notes', [])
    # records from the other line end: two-ended location
    others = [x for n, _, x in res if x is not a and x.get('fault') and x.get('station') and x['station'].lower() != (a.get('station') or '').lower()]
    if others and a.get('fault'):
        d = two_ended_location(a, others[0], a['line'])
        if d is not None:
            a['loc_two_ended_km'] = d
            a['notes'].append(f"Two-ended location (records from {a['station']} and {others[0]['station']}): {d:.1f} km from {a['station']} "
                              f"({d / a['line']['length_km'] * 100:.0f}% of the line); not affected by fault resistance or infeed.")
    # differences between the records of the same incident
    types = {x['Fault Type'] for x in a['records'] if x['Fault Type'] not in ('No fault',)}
    if len(types) > 1:
        a['notes'].append('The records disagree on the fault type (' + ', '.join(sorted(types)) + '): check the channel mapping of each record.')
    return a, rec


def settings_table(sts):
    rows = []
    keys = ['device', 'description', 'line_length', 'line_z', 'line_angle', 'kzn', 'kzn_angle', 'kzm', 'z1_reach', 'z2_reach', 'z2_time',
            'ct_primary', 'ct_secondary', 'vt_primary', 'vt_secondary', 'values_in', 'auto_reclose']
    for k in keys:
        r = {'Setting': SET_LABEL.get(k, k)}
        for n, sv in sts:
            v = sv.get(k)
            if k in ('z1_reach', 'z2_reach') and v is not None:
                v = f"{v:g} {'%' if sv.get(k + '_unit') == '%' else 'Ω'}"
            r[sv.get('device') or n] = '-' if v is None else (f"{v:g}" if isinstance(v, float) else str(v))
        rows.append(r)
    return pd.DataFrame(rows)


def _cross(p1, d1, p2, d2):
    """Intersection of the lines p1 + t d1 and p2 + u d2 (complex numbers)."""
    den = (d1.real * d2.imag - d1.imag * d2.real)
    if abs(den) < 1e-12:
        return p1
    t = ((p2 - p1).real * d2.imag - (p2 - p1).imag * d2.real) / den
    return p1 + t * d1


def zone_shapes(a):
    """Distance zones on the R-X plane (primary ohms) as (label, colour, xs, ys, kind). Quadrilateral when the settings
    say so (reach line with tilt, resistive reach parallel to the line angle, directional line), otherwise mho circles."""
    line, st = a['line'], a['setting']
    Z = complex(line['r1'], line['x1']) * line['length_km']
    ang = np.angle(Z)
    kind = 'gnd' if str(a.get('type', '')).endswith('G') else 'ph'
    char = st.get('char_' + kind) or st.get('char_ph') or st.get('char_gnd') or 'mho'
    Rr = st.get('r_' + kind) or st.get('r_ph') or st.get('r_gnd') or [None, None, None]
    zones = [(st.get('z1_reach'), 'Zone 1', '#d63a3a'), (st.get('z2_reach'), 'Zone 2', '#d99a00'), (st.get('z3_reach'), 'Zone 3', '#2a78d6')]
    out = []
    for i, (reach, lab, col) in enumerate(zones):
        if not reach:
            continue
        if char == 'quad':
            R = Rr[i] if i < len(Rr) and Rr[i] else 2.0 * abs(Z)          # typical resistive reach when not set
            P = Z * reach
            top_d = np.exp(1j * np.radians(st.get('tilt_deg', -3.0)))
            side_d = np.exp(1j * ang)
            bot_d = np.exp(1j * np.radians(-30.0))
            TL = _cross(P, top_d, complex(-R / 4, 0), side_d)
            TR = _cross(P, top_d, complex(R, 0), side_d)
            BR = _cross(complex(R, 0), side_d, 0j, bot_d)
            BL = _cross(complex(-R / 4, 0), side_d, 0j, bot_d)
            pts = [TL, TR, BR, 0j, BL, TL]
            out.append((f"{lab} ({reach * 100:.0f}%, R {R:.0f} Ω)", col, [p.real for p in pts], [p.imag for p in pts], 'quad'))
        else:
            th = np.linspace(0, 2 * np.pi, 200)
            c = Z * reach / 2
            out.append((f"{lab} ({reach * 100:.0f}%)", col, list(c.real + abs(c) * np.cos(th)), list(c.imag + abs(c) * np.sin(th)), 'mho'))
    return out[::-1]                            # outer zones first, so zone 1 is drawn on top


def zone_seen(a):
    """Innermost distance zone (1, 2, 3) that contains the apparent impedance at the fault; 0 = outside all; None = unknown."""
    if a.get('Zapp') is None or not np.isfinite(a['Zapp']):
        return None
    from matplotlib.path import Path
    shapes = zone_shapes(a)[::-1]                # zone 1 first
    if not shapes:
        return None
    for lab, _, xs, ys, _ in shapes:
        if Path(np.column_stack([xs, ys])).contains_point((a['Zapp'].real, a['Zapp'].imag)):
            return int(re.search(r'Zone (\d)', lab).group(1))
    return 0


# ---------------------------------------------------------------- professional figures (shared by the app and the PDF)
import re as _re
from matplotlib.patches import Polygon as _Poly, FancyBboxPatch as _Box
PH_C = {'A': '#d62728', 'B': '#e3a400', 'C': '#1f5fbf', 'N': '#222222'}


def stations(a):
    """Names of the two line ends: from 'A - B' / 'A to B' in the line name, else the station and 'Remote end'."""
    m = a.get('meta', {}) or {}
    if m.get('End A') and m.get('End B'):
        return m['End A'], m['End B']
    ln = str(m.get('Line') or '')
    tidy = lambda x: _re.sub(r'\s{2,}', ' ', _re.sub(r'\b(\d+\s*kv|s/?s|substation|sub|bay|\d+[a-z]?|l\d+|line)\b', '', str(x), flags=_re.I)).strip(' -').title()
    sA = tidy(m.get('Station') or a.get('station') or '') or 'Station A'
    if not _re.search(r'\s-\s|\bto\b', ln, _re.I):
        used = a.get('settings_used')
        desc = (used[1].get('description') if used else '') or ''
        rb = tidy(desc)
        if rb and rb.lower() != sA.lower():
            return sA, rb
    parts = _re.split(r'\s+-\s+|\s*\bto\b\s*|\s*–\s*', ' ' + _re.sub(r'\b(L\d+|line\s*\d+)\b', '', ln, flags=_re.I) + ' ')
    parts = [p.strip(' -') for p in parts if p.strip(' -')]
    if len(parts) == 1 and _re.search(r'\bto\b', ln, _re.I):
        parts = [sA, parts[0]]
    if len(parts) >= 2:
        A_, B_ = parts[0], parts[1]
        if sA and A_.lower().split()[0] not in sA.lower() and B_.lower().split()[0] in sA.lower():
            A_, B_ = B_, A_
        return A_, B_
    return sA, 'Remote end'


def _bolt(ax, x, y, h, w, color='#d62728'):
    """Lightning bolt pointing down to (x, y); h in y units, w in x units."""
    pts = [(x, y), (x - w * 0.15, y + h * 0.45), (x + w * 0.35, y + h * 0.45), (x - w * 0.25, y + h),
           (x + w * 0.55, y + h), (x + w * 0.6, y + h * 0.55), (x + w * 0.1, y + h * 0.55), (x + w * 0.45, y + h * 0.1)]
    ax.add_patch(_Poly(pts, closed=True, fc=color, ec='#7a0000', lw=0.8, zorder=6))


def line_fig(a, width=10, height=3.3):
    """Single-line view: station A -- line -- station B, fault bolt at the located distance, zone reaches, relay estimates."""
    L = a['line']['length_km']; st = a['setting']
    A_, B_ = stations(a)
    loc = a.get('loc_km')
    xmax = max(L * 1.12, (loc or 0) * 1.06)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(-0.12 * L, xmax); ax.set_ylim(-2.6, 2.4); ax.axis('off')
    ax.plot([0, L], [0, 0], color='#3b3b3b', lw=4, solid_capstyle='butt', zorder=2)
    if loc is not None and loc > L:
        ax.plot([L, loc * 1.03], [0, 0], color='#9a9a9a', lw=2.5, ls='--', zorder=2)
        ax.text((L + loc) / 2, 0.25, 'adjacent line', ha='center', fontsize=7.5, color='#777')
    for x, name, ha in ((0, A_, 'right'), (L, B_, 'left')):
        ax.plot([x, x], [-0.75, 0.75], color='#0b2e59', lw=7, solid_capstyle='butt', zorder=3)
        ax.add_patch(_Box((x - 0.012 * L, -0.11), 0.024 * L, 0.22, boxstyle='square,pad=0', fc='white', ec='#0b2e59', lw=1.2, zorder=4))
        ax.text(x + (-0.02 * L if ha == 'right' else 0.02 * L), 0.95, name, ha=ha, va='bottom', fontsize=11, fontweight='bold', color='#0b2e59')
    ax.text(0, -0.95, 'relay', ha='center', va='top', fontsize=7, color='#555')
    # zone reaches under the line
    for i, (k, lab, col) in enumerate([('z1_reach', 'Zone 1', '#d62728'), ('z2_reach', 'Zone 2', '#e3a400')]):
        r = st.get(k)
        if r:
            y = -1.45 - 0.45 * i
            ax.annotate('', xy=(min(r * L, xmax * 0.995), y), xytext=(0, y), arrowprops=dict(arrowstyle='-|>', color=col, lw=2.2))
            ax.text(min(r * L, xmax * 0.995), y + 0.08, f" {lab} {r * 100:.0f}%", va='bottom', ha='right' if r * L > xmax * 0.9 else 'left',
                    fontsize=8, color=col, fontweight='bold')
    # relay estimates
    reps = a.get('relay_reports') or {}
    k = 0
    for dev, v in reps.items():
        m_ = _re.match(r'(-?\d+(?:\.\d+)?)\s*km', str(v.get('Relay Fault Location', '')))
        if m_:
            x = float(m_.group(1))
            ax.plot(x, -0.32, marker='^', ms=9, color='#1f5fbf', zorder=5)
            ax.text(x, -0.55 - 0.34 * (k % 2), f"{dev} {x:.1f} km", ha='right' if x > 0.9 * L else 'center', va='top', fontsize=7.5, color='#1f5fbf')
            k += 1
    if a.get('loc_two_ended_km') is not None:
        x = a['loc_two_ended_km']; ax.plot(x, 0.32, marker='v', ms=9, color='#1a9e5c', zorder=5)
    if loc is not None:
        _bolt(ax, loc, 0.12, 1.25, 0.05 * L)
        typ = a.get('type', '')
        ax.text(loc, 1.48, f"{typ} fault\n{loc:.1f} km ({loc / L * 100:.0f}%)", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#d62728',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d62728', lw=1))
        if loc <= L:
            ax.text(loc / 2, 0.2, f"{loc:.1f} km", ha='center', va='bottom', fontsize=8.5, color='#333')
            if L - loc > 0.06 * L:
                ax.text((loc + L) / 2, 0.2, f"{L - loc:.1f} km", ha='center', va='bottom', fontsize=8.5, color='#333')
    ax.text(L / 2, -0.42 if not reps else 2.25, f"Line length {L:.1f} km", ha='center', fontsize=8, color='#666')
    fig.tight_layout()
    return fig


def _window(rec, a):
    t = (rec['t'] - rec['t'][0]) * 1000
    inc = a.get('t_inception_ms') or 0
    ends = [v for v in (a.get('t_interruption_ms'), a.get('t_trip_ms'), a.get('t_reclose_ms'), a.get('t_refault_ms')) if v is not None]
    chg = [c[0] * 1000 for c in a.get('binary', [])]
    last = max(ends + [inc + 100] + [c for c in chg if c < (max(ends) if ends else inc) + 400])
    t0 = max(t[0], inc - 80); t1 = min(t[-1], last + 120)
    return t, t0, t1


def wave_fig(rec, a, width=10, height=None):
    """Currents (CT), voltages (VT) and binary contacts around the fault, with the key instants marked."""
    A = a.get('A')
    if A is None:
        A = scale_primary(rec)
    ch = a['channels']
    t, t0, t1 = _window(rec, a)
    m = (t >= t0) & (t <= t1)
    D = rec['D']
    act = [j for j in range(D.shape[1]) if D[m, j].any() and not _re.search(r'unused|spare', rec['digital'][j], _re.I)][:16]
    nb = len(act)
    height = height or (6.2 + 0.28 * nb)
    ratios = [3, 3] + ([max(1.2, 0.28 * nb)] if nb else [])
    fig, axs = plt.subplots(len(ratios), 1, figsize=(width, height), sharex=True, gridspec_kw={'height_ratios': ratios, 'hspace': 0.12})
    axI, axV = axs[0], axs[1]
    for p in 'ABC':
        if ch.get('I' + p) is not None:
            axI.plot(t[m], A[m, ch['I' + p]] / 1000, lw=1.1, color=PH_C[p], label='I' + p)
        if ch.get('V' + p) is not None:
            axV.plot(t[m], A[m, ch['V' + p]] / 1000, lw=1.1, color=PH_C[p], label='V' + p)
    if ch.get('IN') is not None:
        axI.plot(t[m], A[m, ch['IN']] / 1000, lw=1.0, color=PH_C['N'], ls='--', label='IN')
    axI.set_ylabel('Current (kA)', fontsize=9); axV.set_ylabel('Voltage (kV)', fontsize=9)
    axI.set_title('Currents (primary) - CT', loc='left', fontsize=9.5, fontweight='bold', color='#0b2e59')
    axV.set_title('Voltages (primary) - VT', loc='left', fontsize=9.5, fontweight='bold', color='#0b2e59')
    inc, intr = a.get('t_inception_ms'), a.get('t_interruption_ms')
    for ax in axs:
        ax.grid(alpha=.3, lw=0.6); ax.tick_params(labelsize=8)
        if inc is not None:
            ax.axvspan(inc, intr if intr is not None else t1, color='#d62728', alpha=0.06, lw=0)
    marks = [('t_inception_ms', '#d62728', 'Fault'), ('t_trip_ms', '#7b2cbf', 'Trip'), ('t_interruption_ms', '#1a9e5c', 'Cleared'),
             ('t_reclose_ms', '#0b2e59', 'Reclose'), ('t_refault_ms', '#d62728', 'Re-fault')]
    yl = axI.get_ylim()
    for i, (key, col, lab) in enumerate(marks):
        v = a.get(key)
        if v is None or not (t0 <= v <= t1):
            continue
        for ax in axs:
            ax.axvline(v, color=col, ls='--', lw=1.2, zorder=4)
        rel = f" +{v - inc:.0f} ms" if (inc is not None and key != 't_inception_ms') else ''
        axI.text(v, yl[1] - (yl[1] - yl[0]) * (0.02 + 0.11 * (i % 3)), f" {lab}{rel}", color='white', fontsize=7.8, fontweight='bold', va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.2', fc=col, ec='none'), zorder=6)
    axI.legend(fontsize=8, ncol=4, loc='lower right', frameon=True, framealpha=0.9)
    axV.legend(fontsize=8, ncol=3, loc='lower right', frameon=True, framealpha=0.9)
    if a.get('I_peak_kA'):
        axI.text(0.01, 0.04, f"Fault current {a.get('I_fault_max_kA', 0):.2f} kA rms, peak {a['I_peak_kA']:.2f} kA", transform=axI.transAxes,
                 fontsize=8, color='#333', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#ccc'))
    if nb:
        axB = axs[2]
        axB.set_title('Binary signals - relay contacts', loc='left', fontsize=9.5, fontweight='bold', color='#0b2e59')
        for k_, j in enumerate(act):
            nm = rec['digital'][j].strip()
            col = '#d62728' if _re.search(r'trip', nm, _re.I) else '#e3a400' if _re.search(r'start|pick', nm, _re.I) else \
                '#1f5fbf' if _re.search(r'aid|send|rec|carrier|pott|dcb', nm, _re.I) else '#1a9e5c' if _re.search(r'cb|52|close|open', nm, _re.I) else '#777'
            x = D[m, j]; tt = t[m]
            on = np.flatnonzero(np.diff(np.r_[0, x, 0]))
            for s_, e_ in zip(on[::2], on[1::2]):
                axB.broken_barh([(tt[s_], tt[min(e_, len(tt) - 1)] - tt[s_])], (nb - 1 - k_ - 0.32, 0.64), fc=col, ec='none')
        axB.set_yticks(range(nb)); axB.set_yticklabels([rec['digital'][j].strip()[:22] for j in act][::-1], fontsize=7.5)
        axB.set_ylim(-0.6, nb - 0.4); axB.grid(axis='y', alpha=0)
    axs[-1].set_xlabel('Time (ms from record start)', fontsize=9)
    axs[-1].set_xlim(t0, t1)
    return fig


def rx_fig(a, width=7.2, height=6.4):
    """R-X diagram: zones as set (quadrilateral or mho, filled), line A -> B, impedance trajectory and the fault point."""
    line = a['line']; st = a['setting']; Z = complex(line['r1'], line['x1']) * line['length_km']
    A_, B_ = stations(a)
    shapes = zone_shapes(a)
    za = a.get('Zapp')
    lim = max(1.45 * abs(Z) * max(st.get('z2_reach', 1.2), 1.0), (abs(za) * 1.35 if za is not None and np.isfinite(za) else 0))
    fig, ax = plt.subplots(figsize=(width, height))
    for lab, col, xs, ys, kind in shapes:
        ax.fill(xs, ys, fc=col, alpha=0.07, zorder=1)
        ax.plot(xs, ys, color=col, lw=1.8, zorder=2)
        top = max(range(len(ys)), key=lambda i: ys[i]) if kind == 'mho' else 0
        xl, yl = xs[top], ys[top]
        if kind == 'quad':                      # label on the reach line, inside the view
            x0_, y0_, x1_, y1_ = xs[0], ys[0], xs[1], ys[1]
            xl = min(max(0.62 * lim, x0_), x1_); yl = y0_ + (y1_ - y0_) * ((xl - x0_) / (x1_ - x0_) if x1_ != x0_ else 0)
        if -0.5 * lim < xl < lim and -0.28 * lim < yl < 1.06 * lim:
            ax.text(xl, yl, ' ' + lab.split(' (')[0] + ' ', fontsize=8.5, color=col, fontweight='bold', va='bottom', ha='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=col, lw=0.8), zorder=5)
        ax.plot([], [], color=col, lw=2, label=lab)
    ax.plot([0, Z.real], [0, Z.imag], color='#3b3b3b', lw=2.6, zorder=3)
    ax.plot([0], [0], 's', color='#0b2e59', ms=8, zorder=4); ax.plot([Z.real], [Z.imag], 's', color='#0b2e59', ms=8, zorder=4)
    ax.text(0, 0, f"  {A_}", fontsize=8.5, fontweight='bold', color='#0b2e59', va='top', ha='left')
    ax.annotate(B_, xy=(Z.real, Z.imag), xytext=(Z.real + 0.12 * lim, Z.imag - 0.1 * lim), fontsize=8.5, fontweight='bold', color='#0b2e59',
                arrowprops=dict(arrowstyle='-', color='#0b2e59', lw=0.8))
    z = a.get('traj')
    if z is not None and len(z):
        z = z[np.isfinite(z)]
        z = z[np.abs(z) < 4 * lim]
        if len(z):
            sc = ax.scatter(z.real, z.imag, c=np.arange(len(z)), cmap='Blues', s=10, vmin=-len(z) * 0.3, zorder=3, label='Impedance trajectory')
            if len(z) > 3:
                ax.annotate('', xy=(z[-1].real, z[-1].imag), xytext=(z[-3].real, z[-3].imag), arrowprops=dict(arrowstyle='-|>', color='#0b2e59', lw=1.2))
    if za is not None and np.isfinite(za):
        zs = a.get('zone_seen')
        ztxt = 'outside all zones' if zs == 0 else (f"inside Zone {zs}" if zs else '')
        ax.plot(za.real, za.imag, marker='*', ms=20, color='#d62728', mec='#7a0000', zorder=7)
        ax.annotate(f"Fault point\nR {za.real:.1f} Ω, X {za.imag:.1f} Ω\n{ztxt}", xy=(za.real, za.imag),
                    xytext=(za.real - 0.32 * lim, za.imag + 0.18 * lim), fontsize=8.8, fontweight='bold', color='#d62728',
                    arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2), bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#d62728'), zorder=8)
    ax.axhline(0, color='#999', lw=0.8); ax.axvline(0, color='#999', lw=0.8)
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.95)
    ax.set_xlim(-0.55 * lim, lim); ax.set_ylim(-0.3 * lim, 1.1 * lim); ax.set_aspect('equal')
    kind = shapes[0][4] if shapes else 'mho'
    src_ = 'from the relay settings' if a.get('settings_used') else 'as entered'
    ax.set_title(f"Impedance seen by the relay ({'quadrilateral' if kind == 'quad' else 'mho'} zones {src_})", fontsize=10, fontweight='bold', color='#0b2e59', loc='left')
    ax.set_xlabel('R (Ω primary)', fontsize=9); ax.set_ylabel('X (Ω primary)', fontsize=9); ax.grid(alpha=.3, lw=0.6); ax.tick_params(labelsize=8)
    return fig


# ---------------------------------------------------------------- the protected equipment (one trip report per fault)
EQUIP_TYPES = ['Line', 'Cable', 'Transformer', 'Reactor', 'Busbar']


def equipment_code(text):
    """Equipment number from a relay description such as '21A SILAA L12' -> 'L12' (also T1, R2, TR3, BB1)."""
    m = _re.search(r'\b((?:L|T|TR|R|SR|BB|C)\s*-?\d{1,3}[A-Z]?)\b', str(text or ''), _re.I)
    return m.group(1).upper().replace(' ', '') if m else ''


def name_equipment(a, typ=None, name=None, end_a=None, end_b=None):
    """Fill the incident's equipment identity: type, number, the two ends (stations) and the report title.
    Ends come from the user, else from the records of both ends, else from the record station and the relay description."""
    m = a.setdefault('meta', {})
    typ = typ or m.get('Equipment Type') or 'Line'
    used = a.get('settings_used')
    desc = (used[1].get('description') if used else '') or ''
    name = name or m.get('Equipment') or equipment_code(desc) or equipment_code(m.get('Line'))
    tidy = lambda x: _re.sub(r'\s{2,}', ' ', _re.sub(r'\b(\d+\s*kv|s/?s|substation|sub|bay|\d+[a-z]?|l\d+|line)\b', '', str(x), flags=_re.I)).strip(' -').title()
    rec_st = []
    for r in a.get('records') or []:
        t_ = tidy(r.get('Station') or '')
        if t_ and t_ not in rec_st:
            rec_st.append(t_)
    if not (end_a and end_b):
        if len(rec_st) >= 2:
            a0, b0 = rec_st[0], rec_st[1]
        else:
            a0, b0 = stations(a)
        end_a, end_b = end_a or a0, end_b or (b0 if b0 != 'Remote end' or typ in ('Line', 'Cable') else '')
    kv = a.get('line', {}).get('kv')
    kvs = f"{kv:g} kV " if kv else ''
    if typ in ('Line', 'Cable'):
        title = f"{end_a} - {end_b} {kvs}{typ} {name}".strip()
    else:
        title = f"{end_a} {kvs}{typ} {name}".strip()
    m.update({'Equipment Type': typ, 'Equipment': name, 'End A': end_a, 'End B': end_b if typ in ('Line', 'Cable') else '',
              'Line': _re.sub(r'\s{2,}', ' ', title), 'Station': end_a if typ not in ('Line', 'Cable') else f"{end_a} / {end_b}"})
    return m['Line']
