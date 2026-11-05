"""FaultGuard AI - dynamic trip report: one self-contained HTML file (no internet needed).

The disturbance records of every line end are replayed on one time line (0 = fault inception): waveforms with a moving cursor,
the rms current and voltage of each end, the phasors, the line with the fault and the breakers opening, the relay contacts
and the sequence of events, followed by the results by source and the assessment.
"""
import html as _h
import json

import numpy as np
import pandas as pd

import faultguard_core as fg

SLAB = {'EXCELLENT': 'NORMAL', 'WATCH': 'WATCH', 'ACTION': 'ACTION REQUIRED'}
SCOL = {'EXCELLENT': '#1a9e5c', 'WATCH': '#e6a100', 'ACTION': '#d62728'}


def _rec_data(rec, x, name):
    """One record as arrays (ms from fault inception), with the one-cycle rms of every channel."""
    A = x.get('A')
    if A is None:
        A = fg.scale_primary(rec)
    ch = x['channels']
    t, t0, t1 = fg._window(rec, x)
    inc = float(x.get('t_inception_ms') or 0.0)
    m = (t >= t0) & (t <= t1)
    tt = t[m]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.5
    n_cyc = max(int(round(20.0 / dt)), 4)
    step = max(1, int(round(0.25 / dt)))               # keep about 4 samples per ms
    out = {'name': name, 'station': fg.short_station(rec.get('station')) or name, 'source': fg.record_source(rec, x),
           't': [round(float(v - inc), 3) for v in tt[::step]], 'dt': dt * step, 'ncyc': max(int(round(20.0 / (dt * step))), 4), 'ch': {}}
    for k in ('IA', 'IB', 'IC', 'IN', 'VA', 'VB', 'VC'):
        j = ch.get(k)
        if j is None:
            continue
        y = A[:, j].astype(float)
        sq = np.convolve(y * y, np.ones(n_cyc) / n_cyc, mode='full')[:len(y)]
        rms = np.sqrt(np.maximum(sq, 0))
        sc = 1000.0
        out['ch'][k] = {'y': [round(float(v), 4) for v in (y[m] / sc)[::step]], 'rms': [round(float(v), 4) for v in (rms[m] / sc)[::step]]}
    key = {}
    for k, lab in (('t_trip_ms', 'Trip'), ('t_interruption_ms', 'Current interrupted'), ('t_reclose_ms', 'Re-energised')):
        if x.get(k) is not None:
            key[lab] = round(float(x[k] - inc), 1)
    out['key'] = key
    rows, srcs = fg.binary_rows(rec, x, t0, t1, max_rows=32, own_only=True)
    out['bin'] = [{'n': nm, 's': src, 'iv': [[round(a_ - inc, 1), round(b_ - inc, 1)] for a_, b_ in iv]} for nm, src, iv in rows]
    out['bin_src'] = srcs
    so = fg.soe(x)
    ev = []
    for _, r in so.iterrows():
        tm = float(r['Time (ms)']) - inc
        if t0 - inc - 50 <= tm <= t1 - inc + 50:
            ev.append({'t': round(tm, 1), 's': str(r['Source']), 'e': str(r['Event'])[:140], 'c': str(r.get('Clock Time', ''))})
    out['events'] = ev
    out['ifault'] = round(float(x.get('I_fault_max_kA') or 0), 2)
    out['type'] = str(x.get('type'))
    return out


def _table_html(df, cls='t'):
    head = ''.join(f'<th>{_h.escape(str(c))}</th>' for c in df.columns)
    body = ''.join('<tr>' + ''.join(f'<td>{_h.escape(str(v))}</td>' for v in r) + '</tr>' for r in df.astype(str).replace({'nan': '-', 'None': '-'}).values)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def trip_html(a, rec):
    """Dynamic trip report (str, a complete HTML page)."""
    meta = a.get('meta', {})
    eq = meta.get('Line') or 'Protected equipment'
    recs = [_rec_data(rec, a, 'record 1')]
    for k, (n_, r2, a2) in enumerate(a.get('other_records') or []):
        if a2.get('fault') and a2.get('t_inception_ms') is not None:
            recs.append(_rec_data(r2, a2, f'record {k + 2}'))
    L = a['line']['length_km']
    ends, line_recs = [], []
    for k, r in enumerate(recs):                      # one record per line end for the line view
        if r['station'] not in ends:
            ends.append(r['station']); line_recs.append(k)
    remote_i = None
    if len(ends) == 1:
        B_ = fg.short_station((a.get('meta') or {}).get('End B')) or 'Remote end'
        if B_ == ends[0]:
            B_ = fg.short_station((a.get('meta') or {}).get('End A')) or 'Remote end'
        ends.append(B_); line_recs.append(None)
        for dev, v in (a.get('relay_reports') or {}).items():      # the line differential relay reports the remote current
            xs = [v.get(f'Relay I{p} remote 1') for p in 'ABC']
            xs = [x for x in xs if x is not None]
            if xs:
                remote_i = {'i': round(max(xs) / 1000, 2), 'src': dev}
                break
    data = {'title': eq, 'incident': meta.get('Incident', ''), 'date': meta.get('Date', ''), 'relays': meta.get('Relay', ''),
            'status': a.get('Status'), 'slab': SLAB.get(a.get('Status'), a.get('Status')), 'scol': SCOL.get(a.get('Status'), '#555'),
            'headline': fg.fault_title(a), 'risk': fg.risk_title(a, SLAB.get(a.get('Status'), '')), 'risknote': fg.risk_note(a), 'summary': fg.summary_text(a), 'type': str(a.get('type')),
            'typetext': fg.TYPE_TEXT.get(a.get('type'), ''), 'L': L, 'loc': a.get('loc_km'), 'locm': a.get('loc_method') or 'single-ended',
            'ends': ends, 'recs': recs, 'lineRecs': line_recs, 'remoteI': remote_i, 'noLine': bool(a.get('no_line_data'))}
    TS = fg.source_tables(a)
    tables = ''
    if TS:
        for k, T_ in TS.items():
            if k == 'summary':
                continue
            tables += f"<h3>{_h.escape(k)} {'(disturbance records)' if k == 'FaultGuard AI' else '(relay fault records)'}</h3>" + _table_html(T_)
        tables += '<h3>Summary: average and spread of the sources</h3>' + _table_html(TS['summary'], 't sum')
    else:
        tables += '<h3>FaultGuard AI against the relays</h3>' + _table_html(fg.compare_table(a))
    find = ''.join(f'<li>{_h.escape(r)}</li>' for r in fg.findings(a))
    notes = ''.join(f'<li>{_h.escape(n)}</li>' for n in a.get('notes', []) if not n.startswith('Single-ended locations'))
    js = json.dumps(data, default=lambda o: o.item() if hasattr(o, 'item') else str(o)).replace('</', '<\\/')
    page = TEMPLATE.replace('__DATA__', js).replace('__TITLE__', _h.escape(eq)).replace('__TABLES__', tables) \
        .replace('__FINDINGS__', find).replace('__NOTES__', notes).replace('__ACTION__', _h.escape(str(a.get('Action', ''))))
    return page


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dynamic Trip Report - __TITLE__</title>
<style>
:root{--navy:#0b2e59;--ink:#1d2939;--mute:#667085;--line:#e4e7ec;--bg:#f4f6fa;--card:#fff;--A:#d62728;--B:#e6a100;--C:#1f5fbf;--N:#555}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{background:var(--navy);color:#fff;padding:14px 20px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
header h1{margin:0;font-size:20px}header .sub{font-size:12px;opacity:.8}
.wrap{max-width:1500px;margin:0 auto;padding:14px 16px 40px}
.banner{border:2px solid;border-radius:12px;padding:12px 16px;background:#fff;margin-bottom:12px}
.banner b{font-size:17px}.banner p{margin:6px 0 0;font-size:13.5px;line-height:1.45}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin-bottom:12px}
.card h2{margin:0 0 8px;font-size:15px;color:var(--navy)}
.ctrl{position:sticky;top:0;z-index:5;display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin-bottom:12px;box-shadow:0 2px 8px rgba(16,24,40,.06)}
.ctrl button{border:0;border-radius:8px;padding:8px 14px;font-weight:600;cursor:pointer;background:var(--navy);color:#fff}
.ctrl button.sec{background:#eef2f7;color:var(--navy)}
.ctrl input[type=range]{flex:1;min-width:200px}
.ctrl .tm{font:600 18px Consolas,monospace;min-width:150px;color:var(--navy)}
.ctrl select{padding:6px;border-radius:8px;border:1px solid var(--line)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:1000px){.grid2{grid-template-columns:1fr}}
canvas{width:100%;display:block}
.kv{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
.kv div{background:#f2f5fa;border-radius:8px;padding:6px 10px;font-size:12px}.kv b{font-size:15px;display:block}
.ev{max-height:330px;overflow:auto;font-size:12px;border-top:1px solid var(--line)}
.ev div{padding:4px 6px;border-bottom:1px solid #f0f2f5;display:grid;grid-template-columns:70px 120px 1fr;gap:6px;opacity:.45;transition:opacity .2s,background .2s}
.ev div.on{opacity:1}.ev div.now{background:#fff4d6;opacity:1}
.ev div span:first-child{font-family:Consolas,monospace}
table.t{border-collapse:collapse;width:100%;font-size:12.5px;margin:4px 0 12px}
table.t th{background:var(--navy);color:#fff;text-align:left;padding:6px 8px;font-weight:600}
table.t td{padding:5px 8px;border-bottom:1px solid var(--line)}table.t tr:nth-child(even) td{background:#f8fafc}
h3{font-size:13.5px;color:var(--navy);margin:10px 0 4px}
ul{margin:4px 0 8px 18px;font-size:13px;line-height:1.5}
.lg{font-size:11.5px;color:var(--mute)}
.pill{display:inline-block;border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;color:#fff}
footer{color:var(--mute);font-size:11px;text-align:center;margin-top:20px}
</style></head><body>
<header><div><h1>⚡ Dynamic Trip Report · <span id="ttl"></span></h1><div class="sub" id="sub"></div></div><div class="sub">FaultGuard AI · PowerGuard AI</div></header>
<div class="wrap">
<div class="banner" id="banner"></div>
<div class="ctrl"><button id="play">▶ Play</button><button class="sec" id="restart">⟲ Restart</button>
<select id="speed"><option value="0.02">0.02× (very slow)</option><option value="0.05" selected>0.05× (slow motion)</option><option value="0.1">0.1×</option><option value="0.25">0.25×</option><option value="1">1× (real time)</option></select>
<input type="range" id="slider" min="0" max="1000" value="0"><span class="tm" id="tm">t = 0.0 ms</span></div>
<div class="card"><h2>The line: fault, currents from each end and breakers</h2><canvas id="lineCv" height="230"></canvas>
<div class="lg">Arrow thickness = one-cycle rms current flowing into the line from that end. Breaker: red = closed, green = open. The values follow the cursor.</div></div>
<div id="ends" class="grid2"></div>
<div class="card"><h2>Results by source</h2>__TABLES__</div>
<div class="grid2"><div class="card"><h2>Assessment</h2><div id="risk"></div><ul>__FINDINGS__</ul><p style="font-size:13px"><b>Action:</b> __ACTION__</p></div>
<div class="card"><h2>Notes</h2><ul>__NOTES__</ul></div></div>
<footer>Generated by FaultGuard AI. Values are primary. Time 0 = fault inception in each record; the records of the two ends are aligned on the inception.</footer>
</div>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
$('ttl').textContent = D.title; document.title = 'Dynamic Trip Report - ' + D.title;
$('sub').textContent = [D.incident, D.date, D.relays].filter(Boolean).join('  ·  ');
$('banner').style.borderColor = '#0b2e59';
$('banner').innerHTML = `<b style="color:#0b2e59">${esc(D.headline)}</b><p>${esc(D.summary)}</p>`;
$('risk').innerHTML = `<b style="color:${D.scol};font-size:15px">${esc(D.risk)}</b><p class="lg">${esc(D.risknote)}</p>`;
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
const PC = {IA:'#d62728',IB:'#e6a100',IC:'#1f5fbf',IN:'#555',VA:'#d62728',VB:'#e6a100',VC:'#1f5fbf'};
// time range shared by all records
let T0 = Infinity, T1 = -Infinity;
D.recs.forEach(r => { T0 = Math.min(T0, r.t[0]); T1 = Math.max(T1, r.t[r.t.length - 1]); });
let tNow = T0, playing = false, last = null;
const idxAt = (r, t) => { let lo = 0, hi = r.t.length - 1; if (t <= r.t[0]) return 0; if (t >= r.t[hi]) return hi;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (r.t[m] <= t) lo = m; else hi = m; } return lo; };
// ---------- one card per line end
const ui = [];
D.recs.forEach((r, k) => {
  const div = document.createElement('div'); div.className = 'card';
  div.innerHTML = `<h2>${esc(r.station)} end · ${esc(r.source)}</h2>
   <div class="kv" id="kv${k}"></div>
   <canvas id="wi${k}" height="170"></canvas><canvas id="wv${k}" height="150"></canvas>
   <div style="display:grid;grid-template-columns:230px 1fr;gap:10px;align-items:start;margin-top:6px">
     <div><canvas id="ph${k}" width="230" height="230" style="width:230px"></canvas><div class="lg">Phasors (one cycle at the cursor): currents solid, voltages dashed.</div></div>
     <div><canvas id="bn${k}" height="${Math.max(80, 16 * r.bin.length + 26)}"></canvas><div class="lg">${esc(r.bin_src.join(' · '))}</div></div></div>
   <h3>Sequence of events (ms from fault inception)</h3><div class="ev" id="ev${k}"></div>`;
  $('ends').appendChild(div);
  const ev = $('ev' + k);
  r.events.forEach(e => { const d = document.createElement('div'); d.innerHTML = `<span>${e.t.toFixed(1)}</span><span>${esc(e.s)}</span><span>${esc(e.e)}</span>`; ev.appendChild(d); e.el = d; });
  ui.push({r, wi: $('wi' + k), wv: $('wv' + k), ph: $('ph' + k), bn: $('bn' + k), kv: $('kv' + k), ev});
});
function fit(cv){ const w = cv.clientWidth, h = cv.getAttribute('height'); const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + 'px'; }
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); return [c, w, +h]; }
function xOf(t, w, pad){ return pad + (t - T0) / (T1 - T0) * (w - pad - 10); }
function drawWave(cv, r, keys, unit, t){
  const [c, w, h] = fit(cv); c.clearRect(0, 0, w, h); const pad = 46;
  let mx = 0; keys.forEach(k => { if (r.ch[k]) r.ch[k].y.forEach(v => mx = Math.max(mx, Math.abs(v))); }); mx = mx * 1.08 || 1;
  const y = v => h / 2 - v / mx * (h / 2 - 14);
  c.fillStyle = '#fbfcfe'; c.fillRect(pad, 4, w - pad - 10, h - 8);
  // fault shading and key instants
  const kI = r.key['Current interrupted'];
  c.fillStyle = 'rgba(214,39,40,.07)'; c.fillRect(xOf(0, w, pad), 4, xOf(kI != null ? kI : T1, w, pad) - xOf(0, w, pad), h - 8);
  c.strokeStyle = '#e4e7ec'; c.lineWidth = 1; c.beginPath(); c.moveTo(pad, h / 2); c.lineTo(w - 10, h / 2); c.stroke();
  c.fillStyle = '#667085'; c.font = '11px Segoe UI'; c.fillText(`${mx.toFixed(1)} ${unit}`, 4, 14); c.fillText(`-${mx.toFixed(1)}`, 4, h - 6);
  keys.forEach(k => { const ch = r.ch[k]; if (!ch) return; c.strokeStyle = PC[k]; c.lineWidth = k === 'IN' ? 1 : 1.4; c.setLineDash(k === 'IN' ? [4, 3] : []);
    c.beginPath(); r.t.forEach((tt, i) => { const X = xOf(tt, w, pad), Y = y(ch.y[i]); i ? c.lineTo(X, Y) : c.moveTo(X, Y); }); c.stroke(); });
  c.setLineDash([]);
  const mark = (tt, col, lab, row) => { if (tt == null) return; const X = xOf(tt, w, pad); c.strokeStyle = col; c.setLineDash([5, 4]); c.beginPath(); c.moveTo(X, 4); c.lineTo(X, h - 4); c.stroke(); c.setLineDash([]);
    c.font = '600 10.5px Segoe UI'; const tw = c.measureText(lab).width; c.fillStyle = 'rgba(255,255,255,.85)'; c.fillRect(X + 2, 6 + row * 13, tw + 4, 12);
    c.fillStyle = col; c.fillText(lab, X + 4, 16 + row * 13); };
  mark(0, '#d62728', 'Fault 0 ms', 0); mark(r.key['Trip'], '#7b3fbf', 'Trip ' + (r.key['Trip'] != null ? r.key['Trip'].toFixed(0) + ' ms' : ''), 1); mark(kI, '#1a9e5c', 'Cleared ' + (kI != null ? kI.toFixed(0) + ' ms' : ''), 2);
  // cursor
  const X = xOf(t, w, pad); c.strokeStyle = '#0b2e59'; c.lineWidth = 2; c.beginPath(); c.moveTo(X, 2); c.lineTo(X, h - 2); c.stroke();
  // legend
  let lx = w - 10 - keys.length * 42; keys.forEach(k => { if (!r.ch[k]) return; c.fillStyle = PC[k]; c.fillRect(lx, h - 16, 14, 3); c.fillStyle = '#344054'; c.fillText(k, lx + 17, h - 11); lx += 42; });
}
function phasors(r, t){ const i = idxAt(r, t), N = r.ncyc, out = {};
  Object.keys(r.ch).forEach(k => { let re = 0, im = 0; const s = Math.max(0, i - N + 1);
    for (let j = s; j <= i; j++) { const a = 2 * Math.PI * (j - s) / N; re += r.ch[k].y[j] * Math.cos(a); im -= r.ch[k].y[j] * Math.sin(a); }
    const n = (i - s + 1) || 1; out[k] = [re * Math.SQRT2 / n, im * Math.SQRT2 / n]; });
  return out; }
function drawPhasor(cv, r, t){ const [c, w, h] = fit(cv); c.clearRect(0, 0, w, h); const P = phasors(r, t), cx = w / 2, cy = h / 2, R = w / 2 - 14;
  c.strokeStyle = '#e4e7ec'; c.beginPath(); c.arc(cx, cy, R, 0, 2 * Math.PI); c.stroke(); c.beginPath(); c.moveTo(cx - R, cy); c.lineTo(cx + R, cy); c.moveTo(cx, cy - R); c.lineTo(cx, cy + R); c.stroke();
  const ref = P.VA || P.IA || [1, 0], ang = Math.atan2(ref[1], ref[0]);
  const draw = (keys, dash) => { let mx = 1e-9; keys.forEach(k => { if (P[k]) mx = Math.max(mx, Math.hypot(P[k][0], P[k][1])); });
    keys.forEach(k => { if (!P[k]) return; const m = Math.hypot(P[k][0], P[k][1]) / mx * R * 0.95, a = Math.atan2(P[k][1], P[k][0]) - ang;
      const X = cx + m * Math.cos(a), Y = cy - m * Math.sin(a); c.strokeStyle = PC[k]; c.lineWidth = 2.4; c.setLineDash(dash); c.beginPath(); c.moveTo(cx, cy); c.lineTo(X, Y); c.stroke();
      c.setLineDash([]); c.fillStyle = PC[k]; c.beginPath(); c.arc(X, Y, 3.5, 0, 2 * Math.PI); c.fill(); c.font = '600 10px Segoe UI'; c.fillText(k, X + 4, Y - 4); }); };
  draw(['VA', 'VB', 'VC'], [5, 3]); draw(['IA', 'IB', 'IC'], []); }
function drawBin(cv, r, t){ const [c, w, h] = fit(cv); c.clearRect(0, 0, w, h); const pad = 170, rh = 16;
  r.bin.forEach((b, k) => { const y = 6 + k * rh; const multi = new Set(r.bin.map(x => x.s)).size > 1;
    const col = /trip/i.test(b.n) ? '#d62728' : /start|pick/i.test(b.n) ? '#e3a400' : /aid|send|rec|carrier/i.test(b.n) ? '#1f5fbf' : /cb|pole|close|open/i.test(b.n) ? '#1a9e5c' : '#888';
    const active = b.iv.some(([a, z]) => t >= a && t <= z);
    c.fillStyle = active ? '#0b2e59' : '#667085'; c.font = (active ? '600 ' : '') + '10.5px Segoe UI';
    c.fillText(((multi ? b.s.split(' ')[0] + ' · ' : '') + b.n).slice(0, 30), 4, y + 11);
    b.iv.forEach(([a, z]) => { const x0 = xOf(Math.max(a, T0), w, pad), x1 = xOf(Math.min(z, T1), w, pad); if (x1 < pad) return;
      c.fillStyle = col; c.globalAlpha = t >= a ? 0.95 : 0.18; c.fillRect(x0, y + 2, Math.max(2, x1 - x0), rh - 5); c.globalAlpha = 1; }); });
  const X = xOf(t, w, pad); c.strokeStyle = '#0b2e59'; c.lineWidth = 2; c.beginPath(); c.moveTo(X, 0); c.lineTo(X, h); c.stroke(); }
function rmsAt(r, k, t){ const ch = r.ch[k]; return ch ? ch.rms[idxAt(r, t)] : null; }
function endState(r, t){ const I = ['IA', 'IB', 'IC'].map(k => rmsAt(r, k, t)).filter(v => v != null), V = ['VA', 'VB', 'VC'].map(k => rmsAt(r, k, t)).filter(v => v != null);
  return {I: I.length ? Math.max(...I) : 0, V: V.length ? Math.min(...V) : null, open: r.key['Current interrupted'] != null && t >= r.key['Current interrupted'],
          trip: r.key['Trip'] != null && t >= r.key['Trip']}; }
function drawLine(t){ const cv = $('lineCv'); const [c, w, h] = fit(cv); c.clearRect(0, 0, w, h);
  const xa = 130, xb = w - 130, yl = 120, L = D.L;
  // stations
  [[xa, D.ends[0]], [xb, D.ends[1]]].forEach(([x, n]) => { c.fillStyle = '#0b2e59'; c.fillRect(x - 7, yl - 45, 14, 90); c.font = '700 14px Segoe UI'; c.textAlign = 'center'; c.fillText(n, x, yl - 55); });
  c.textAlign = 'left';
  // line, energised or not
  const loc0 = D.recs[D.lineRecs[0]];
  const st = D.lineRecs.map(k => k == null ? null : endState(D.recs[k], t));
  if (st[1] == null) { const s0 = st[0]; st[1] = {I: (D.remoteI && t >= 0 && !s0.open) ? D.remoteI.i : 0, V: null, open: s0.open, trip: s0.trip, remote: true}; }
  const dead = st.every(s => s.open);
  c.strokeStyle = dead ? '#98a2b3' : '#0b2e59'; c.lineWidth = 5; c.beginPath(); c.moveTo(xa + 28, yl); c.lineTo(xb - 28, yl); c.stroke();
  c.fillStyle = '#667085'; c.font = '12px Segoe UI'; c.textAlign = 'center'; c.fillText(D.noLine ? 'line length not known (no settings file)' : `${L.toFixed(1)} km`, (xa + xb) / 2, yl + 62); c.textAlign = 'left';
  // breakers
  st.forEach((s, k) => { const x = k === 0 ? xa + 20 : xb - 20; c.fillStyle = s.open ? '#1a9e5c' : '#d62728'; c.fillRect(x - 9, yl - 9, 18, 18);
    c.fillStyle = '#344054'; c.font = '600 11px Segoe UI'; c.textAlign = k === 0 ? 'left' : 'right';
    c.fillText('breaker ' + (s.open ? 'OPEN' : (s.trip ? 'tripping' : 'closed')), k === 0 ? x + 14 : x - 14, yl + 26); c.textAlign = 'left'; });
  // fault
  const xf = D.loc != null ? xa + 28 + (xb - xa - 56) * Math.min(Math.max(D.loc / L, 0), 1) : (xa + xb) / 2;
  const on = t >= 0 && !dead;
  if (t >= 0) { c.save(); c.translate(xf, yl); c.fillStyle = on ? '#ffd400' : '#f3c98b'; c.strokeStyle = '#b42318'; c.lineWidth = 2;
    c.beginPath(); c.moveTo(-6, -62); c.lineTo(10, -62); c.lineTo(1, -30); c.lineTo(12, -30); c.lineTo(-8, 4); c.lineTo(-2, -22); c.lineTo(-12, -22); c.closePath(); c.fill(); c.stroke(); c.restore();
    if (on) { c.fillStyle = 'rgba(255,90,30,' + (0.25 + 0.2 * Math.sin(t * 0.6)) + ')'; c.beginPath(); c.arc(xf, yl, 20, 0, 2 * Math.PI); c.fill(); }
    c.fillStyle = '#b42318'; c.font = '700 12.5px Segoe UI'; c.textAlign = 'center';
    c.fillText(`${D.type} fault` + (D.loc != null ? ` · ${D.loc.toFixed(1)} km from ${D.ends[0]}` : (D.noLine ? ' · location not known (no line data)' : '')), xf, yl - 72); c.textAlign = 'left'; }
  // current arrows into the fault
  const Imax = Math.max(...D.recs.map(r => Math.max(...['IA', 'IB', 'IC'].map(k => r.ch[k] ? Math.max(...r.ch[k].rms) : 0))), D.remoteI ? D.remoteI.i : 0, 0.1);
  st.forEach((s, k) => { const x0 = k === 0 ? xa + 34 : xb - 34, x1 = k === 0 ? xf - 26 : xf + 26, dir = k === 0 ? 1 : -1;
    const th = 2 + 16 * s.I / Imax; if (s.I < 0.02) return;
    const col = t >= 0 && !s.open ? '#d62728' : '#1f5fbf';
    c.strokeStyle = col; c.fillStyle = col; c.globalAlpha = 0.85; c.lineWidth = th; c.beginPath(); c.moveTo(x0, yl - 30); c.lineTo(x1 - dir * 14, yl - 30); c.stroke();
    c.beginPath(); c.moveTo(x1, yl - 30); c.lineTo(x1 - dir * (14 + th), yl - 30 - th - 4); c.lineTo(x1 - dir * (14 + th), yl - 30 + th + 4); c.closePath(); c.fill(); c.globalAlpha = 1;
    c.font = '700 14px Segoe UI'; c.textAlign = k === 0 ? 'left' : 'right'; c.fillStyle = col;
    c.fillText(`${s.I.toFixed(2)} kA`, k === 0 ? x0 : x0, yl - 44 - th / 2);
    c.font = '12px Segoe UI'; c.fillStyle = '#344054'; if (s.V != null) c.fillText(`lowest phase voltage ${s.V.toFixed(0)} kV`, x0, yl + 45);
    if (s.remote && D.remoteI) c.fillText(`remote current from the ${D.remoteI.src} fault record (no record at this end)`, x0, yl + 45); c.textAlign = 'left'; });
  if (dead && t >= 0) { c.fillStyle = '#1a9e5c'; c.font = '700 13px Segoe UI'; c.textAlign = 'center'; c.fillText('Fault cleared from both ends', (xa + xb) / 2, 22); c.textAlign = 'left'; }
  else if (t < 0) { c.fillStyle = '#667085'; c.font = '600 13px Segoe UI'; c.textAlign = 'center'; c.fillText('Before the fault: load current', (xa + xb) / 2, 22); c.textAlign = 'left'; }
}
function render(){
  $('tm').textContent = `t = ${tNow.toFixed(1)} ms`;
  $('slider').value = Math.round((tNow - T0) / (T1 - T0) * 1000);
  drawLine(tNow);
  ui.forEach(u => { const r = u.r;
    drawWave(u.wi, r, ['IA', 'IB', 'IC', 'IN'], 'kA', tNow); drawWave(u.wv, r, ['VA', 'VB', 'VC'], 'kV', tNow); drawPhasor(u.ph, r, tNow); drawBin(u.bn, r, tNow);
    const s = endState(r, tNow);
    const I = ['IA', 'IB', 'IC', 'IN'].map(k => [k, rmsAt(r, k, tNow)]).filter(x => x[1] != null), V = ['VA', 'VB', 'VC'].map(k => [k, rmsAt(r, k, tNow)]).filter(x => x[1] != null);
    u.kv.innerHTML = I.map(([k, v]) => `<div style="border-left:3px solid ${PC[k]}">${k}<b>${v.toFixed(2)} kA</b></div>`).join('') +
                     V.map(([k, v]) => `<div style="border-left:3px solid ${PC[k]}">${k}<b>${v.toFixed(0)} kV</b></div>`).join('') +
                     `<div>Breaker<b style="color:${s.open ? '#1a9e5c' : '#d62728'}">${s.open ? 'OPEN' : 'CLOSED'}</b></div>`;
    let nowEl = null; r.events.forEach(e => { const on = e.t <= tNow; e.el.classList.toggle('on', on); e.el.classList.remove('now'); if (on) nowEl = e.el; });
    if (nowEl && playing) { nowEl.classList.add('now'); const box = u.ev; if (nowEl.offsetTop > box.scrollTop + box.clientHeight - 30 || nowEl.offsetTop < box.scrollTop) box.scrollTop = nowEl.offsetTop - 60; }
  });
}
function loop(ts){ if (!playing) { last = null; return; } if (last != null) { tNow += (ts - last) * +$('speed').value; if (tNow >= T1) { tNow = T1; playing = false; $('play').textContent = '▶ Play'; } }
  last = ts; render(); if (playing) requestAnimationFrame(loop); }
$('play').onclick = () => { playing = !playing; $('play').textContent = playing ? '⏸ Pause' : '▶ Play'; if (playing) { if (tNow >= T1) tNow = T0; requestAnimationFrame(loop); } };
$('restart').onclick = () => { tNow = T0; render(); };
$('slider').oninput = e => { tNow = T0 + (T1 - T0) * e.target.value / 1000; render(); };
window.addEventListener('resize', render);
tNow = Math.max(T0, -20); render();
</script></body></html>
'''
