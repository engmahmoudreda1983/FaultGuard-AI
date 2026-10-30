"""FaultGuard AI - PDF reports (incident report, combined report)."""
import io
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

NAVY = colors.HexColor('#0b2e59'); LINE = colors.HexColor('#d5dce6'); MUTED = colors.HexColor('#5a6575')
SCOL = {'EXCELLENT': colors.HexColor('#1a9e5c'), 'WATCH': colors.HexColor('#d99a00'), 'ACTION': colors.HexColor('#d63a3a')}
SLAB = {'EXCELLENT': 'NORMAL', 'WATCH': 'WATCH', 'ACTION': 'ACTION REQUIRED'}
W = A4[0] - 30 * mm

_ss = getSampleStyleSheet()
H1 = ParagraphStyle('h1', parent=_ss['Heading1'], fontName='Helvetica-Bold', fontSize=17, textColor=NAVY, spaceAfter=4)
H2 = ParagraphStyle('h2', parent=_ss['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=NAVY, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle('b', parent=_ss['BodyText'], fontName='Helvetica', fontSize=9.2, leading=12.5)
SMALL = ParagraphStyle('s', parent=BODY, fontSize=7.8, leading=10, textColor=MUTED)
CELL = ParagraphStyle('c', parent=BODY, fontSize=8, leading=10)
CELLB = ParagraphStyle('cb', parent=CELL, fontName='Helvetica-Bold', textColor=colors.white)

_MAP = {'Ω': 'ohm', '≥': '>=', '≤': '<=', '−': '-', '…': '...', '→': '->', '×': 'x', '÷': '/', '²': '2', '·': '-', '’': "'", '‘': "'",
        '“': '"', '”': '"', '–': '-', '—': '-', '±': '+/-'}


def clean(x):
    s = '' if x is None else str(x)
    for k, v in _MAP.items():
        s = s.replace(k, v)
    s = s.encode('latin-1', 'replace').decode('latin-1')
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def P(x, st=BODY):
    return Paragraph(clean(x), st)


def table(rows, widths, head=True, zebra=True, align_right=()):
    data = [[Paragraph(clean(c), CELLB if (head and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [('GRID', (0, 0), (-1, -1), 0.4, LINE), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]
    if head:
        st.append(('BACKGROUND', (0, 0), (-1, 0), NAVY))
    if zebra:
        for i in range(1 if head else 0, len(rows)):
            if i % 2 == 0:
                st.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f5f7fa')))
    t.setStyle(TableStyle(st))
    return t


def banner(text, sub, status):
    c = SCOL.get(status, NAVY)
    t = Table([[Paragraph(f"<font color='{c.hexval()}' size='14'><b>{clean(text)}</b></font>", BODY)], [P(sub)]], colWidths=[W])
    t.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 1.4, c), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                           ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    return t


def fig_image(fig, width=W):
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=160, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    from PIL import Image as PI
    w, h = PI.open(buf).size; buf.seek(0)
    return Image(buf, width=width, height=width * h / w)


def _doc(title):
    buf = io.BytesIO()

    def deco(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY); canvas.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white); canvas.setFont('Helvetica-Bold', 10)
        canvas.drawString(15 * mm, A4[1] - 8 * mm, 'FaultGuard AI'); canvas.setFont('Helvetica', 8.5)
        canvas.drawRightString(A4[0] - 15 * mm, A4[1] - 8 * mm, clean(title).replace('&amp;', '&'))
        canvas.setFillColor(MUTED); canvas.setFont('Helvetica', 7.5)
        canvas.drawString(15 * mm, 9 * mm, f"Generated {datetime.now():%Y-%m-%d %H:%M}  -  PowerGuard AI  -  Confidential")
        canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=18 * mm, bottomMargin=16 * mm,
                            title=title, author='FaultGuard AI')
    return buf, doc, deco


def _fmt(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{d}f}" if abs(v) >= 1 else f"{v:.4g}"
    return str(v)

import pandas as pd
import faultguard_core as fg
GREEN, AMBER, RED, NAVYH = '#1a9e5c', '#d99a00', '#d63a3a', '#0b2e59'
PH_COL = {'A': '#d63a3a', 'B': '#d99a00', 'C': '#2a78d6', 'N': '#555555'}


def wave_fig(rec, a, width=10, height=5.4):
    A = a.get('A'); ch = a['channels']; t = (rec['t'] - rec['t'][0]) * 1000
    fig, axs = plt.subplots(2, 1, figsize=(width, height), sharex=True)
    for p in 'ABC':
        if ch.get('I' + p) is not None:
            axs[0].plot(t, A[:, ch['I' + p]] / 1000, lw=0.7, color=PH_COL[p], label='I' + p)
        if ch.get('V' + p) is not None:
            axs[1].plot(t, A[:, ch['V' + p]] / 1000, lw=0.7, color=PH_COL[p], label='V' + p)
    if ch.get('IN') is not None:
        axs[0].plot(t, A[:, ch['IN']] / 1000, lw=0.6, color=PH_COL['N'], label='IN')
    for k_i, (key, col, lab) in enumerate([('t_inception_ms', AMBER, 'Fault'), ('t_trip_ms', RED, 'Trip'), ('t_interruption_ms', GREEN, 'Cleared'), ('t_reclose_ms', NAVYH, 'Reclose')]):
        v = a.get(key)
        if v is not None:
            for ax in axs:
                ax.axvline(v, color=col, ls='--', lw=0.9)
            yl = axs[0].get_ylim(); axs[0].text(v, yl[1] - (yl[1] - yl[0]) * 0.09 * k_i, f' {lab}', color=col, fontsize=8, va='top')
    axs[0].set_ylabel('Current (kA)'); axs[1].set_ylabel('Voltage (kV)'); axs[1].set_xlabel('ms from record start')
    for ax in axs:
        ax.legend(fontsize=7, ncol=4, loc='upper right', frameon=False); ax.grid(alpha=.25)
    return fig


def rx_fig(a, width=5.2, height=4.4):
    line = a['line']; st = a['setting']; Z = complex(line['r1'], line['x1']) * line['length_km']
    fig, ax = plt.subplots(figsize=(width, height))
    th = np.linspace(0, 2 * np.pi, 200)
    for reach, col, lab in [(st['z1_reach'], RED, 'Zone 1'), (st['z2_reach'], AMBER, 'Zone 2')]:
        c = Z * reach / 2
        ax.plot(c.real + abs(c) * np.cos(th), c.imag + abs(c) * np.sin(th), color=col, lw=1, label=f"{lab} ({reach * 100:.0f}%)")
    ax.plot([0, Z.real], [0, Z.imag], color='#555', lw=1.5, label='Line impedance')
    z = a.get('traj')
    if z is not None:
        z = z[np.isfinite(z)]
        z = z[np.abs(z) < 5 * abs(Z)]
        ax.plot(z.real, z.imag, '.', ms=2.5, color=NAVYH, label='Apparent impedance')
    if a.get('Zapp') is not None:
        ax.plot(a['Zapp'].real, a['Zapp'].imag, 'o', color=RED, ms=6, label='At the fault')
    lim = 1.5 * abs(Z) * st['z2_reach']
    ax.set_xlim(-0.5 * lim, lim); ax.set_ylim(-0.3 * lim, 1.1 * lim); ax.set_aspect('equal')
    ax.set_xlabel('R (ohm primary)'); ax.set_ylabel('X (ohm primary)'); ax.grid(alpha=.25); ax.legend(fontsize=7, loc='lower right', frameon=False)
    return fig


def incident_pdf(a, rec):
    meta = a.get('meta', {})
    buf, doc, deco = _doc('Incident and Trip Analysis Report')
    S = [P('Incident and Trip Analysis Report', H1),
         P(f"{meta.get('Incident', '')}  -  {meta.get('Station', '')}  -  {meta.get('Line', '')}  -  {meta.get('Date', '')}", SMALL), Spacer(1, 4)]
    title = 'TEST RECORD' if a.get('likely_test') else f"{SLAB[a['Status']]}  -  incident risk {a['Overall Risk (%)']:.0f}%"
    S.append(banner(title, fg.summary_text(a), 'EXCELLENT' if a.get('likely_test') else a['Status']))
    S.append(P('Incident', H2))
    S.append(table([['Item', 'Value', 'Item', 'Value'],
                    ['Station', meta.get('Station', '-') or '-', 'Line / bay', meta.get('Line', '-') or '-'],
                    ['Relay', meta.get('Relay', '-') or '-', 'Record start', str(a.get('record_start') or '-')[:23]],
                    ['Line length', f"{a['line']['length_km']:.1f} km", 'Line impedance (per km)', f"Z1 {a['line']['r1']:.3f}+j{a['line']['x1']:.3f}, Z0 {a['line']['r0']:.3f}+j{a['line']['x0']:.3f} ohm"]],
                   [W * 0.16, W * 0.34, W * 0.18, W * 0.32]))
    if a.get('fault'):
        conf = f"{a['type_conf'] * 100:.0f}%"
        S.append(P('Fault', H2))
        C = fg.compare_table(a)
        cols = list(C.columns)
        wq = 0.17; wr = (1 - wq) / (len(cols) - 1)
        S.append(table([cols] + C.astype(str).values.tolist(), [W * wq] + [W * wr] * (len(cols) - 1)))
        if a.get('restored'):
            rs = a['restored']
            S.append(P(f"Line re-energised at {pd.Timestamp(rs['time']):%Y-%m-%d %H:%M:%S}, {fg.fmt_dur(rs['after_s'])} after the fault ({rs['device']}: {rs['event']}).", SMALL))
        for n in a.get('notes', []):
            S.append(P(n, SMALL))
        if len(a.get('records') or []) > 1:
            R_ = pd.DataFrame(a['records'])[['Record', 'Station', 'Fault Type', 'Location (km)', 'Fault Current (kA)', 'Trip (ms)', 'Breaker (ms)', 'Clearing (ms)', 'Voltages']]
            S.append(KeepTogether([P('Disturbance records of this incident', H2),
                                   table([list(R_.columns)] + R_.astype(str).replace({'None': '-', 'nan': '-'}).values.tolist(), [W / len(R_.columns)] * len(R_.columns))]))
        if a.get('settings_all'):
            T_ = fg.settings_table(a['settings_all'])
            T_ = T_[(T_.iloc[:, 1:] != '-').any(axis=1)]
            S.append(KeepTogether([P('Relay settings used', H2),
                                   table([list(T_.columns)] + T_.astype(str).values.tolist(), [W * 0.3] + [W * 0.7 / (len(T_.columns) - 1)] * (len(T_.columns) - 1))]))
        rows = [['Channel', 'Pre-fault (rms)', 'Fault (rms)', 'Fault angle']]
        for k in ('IA', 'IB', 'IC', 'IN', 'VA', 'VB', 'VC'):
            if k in a['P']:
                pre, flt = a['P'][k]; u = 'kA' if k.startswith('I') else 'kV'
                rows.append([k, f"{abs(pre) / 1000:.3f} {u}", f"{abs(flt) / 1000:.3f} {u}", f"{np.degrees(np.angle(flt)):.0f} deg"])
        S.append(KeepTogether([P('Measured values (primary)', H2), table(rows, [W * 0.2, W * 0.27, W * 0.27, W * 0.26])]))
        S.append(KeepTogether([P('Waveforms', H2), fig_image(wave_fig(rec, a))]))
        if a.get('traj') is not None:
            S.append(KeepTogether([P('Impedance seen by the relay (R-X)', H2), fig_image(rx_fig(a), W * 0.6)]))
    if not a.get('fault'):
        for n in a.get('notes', []):
            S.append(P(n))
    S.append(P('Assessment', H2))
    for r in fg.findings(a):
        S.append(P(f"- {r}"))
    S.append(Paragraph(f"<b>Action:</b> {clean(a['Action'])}", BODY))
    S.append(P('FMEA', H2))
    S.append(table([['Aspect', 'Risk %', 'S', 'O', 'D', 'RPN', 'Level']] + [[m['Failure Mode'], f"{m['Risk (%)']:.0f}", m['S'], m['O'], m['D'], m['RPN'], m['RPN Level']] for m in a['FMEA']],
                   [W * 0.34, W * 0.12, W * 0.08, W * 0.08, W * 0.08, W * 0.12, W * 0.18]))
    so = fg.soe(a)
    if len(so):
        rows = [['Clock time' if 'Clock Time' in so else '#', 'ms', 'Source', 'Event']] + [[r.get('Clock Time', i + 1), f"{r['Time (ms)']:.1f}", r['Source'], r['Event']] for i, r in so.head(60).iterrows()]
        S.append(P('Sequence of events', H2)); S.append(table(rows, [W * 0.25, W * 0.1, W * 0.15, W * 0.5]))
    S.append(Spacer(1, 6))
    S.append(P('Fault type by the FaultGuard AI model (Random Forest on superimposed-current and voltage features) checked by the classic phase selection; '
               'location by the Takagi method with the line impedance; phasors by one-cycle Fourier filtering. RPN = S x O x D.', SMALL))
    doc.build(S, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()


def combined_pdf(R, summary, source, n_a, n_w):
    buf, doc, deco = _doc('Incident Summary Report')
    S = [P('Incident Summary Report', H1), P(f"{len(R)} incidents  -  source: {source}", SMALL), Spacer(1, 4)]
    st = 'ACTION' if n_a else 'WATCH' if n_w else 'EXCELLENT'
    S.append(banner(f"{n_a} action  -  {n_w} watch  -  {len(R) - n_a - n_w} normal", summary, st))
    rows = [['Incident', 'Date', 'Line', 'Type', 'Location', 'Clearing', 'Reclose', 'Governing aspect', 'Risk', 'Status']]
    for _, r in R.iterrows():
        rows.append([r['Incident'], str(r['Date'])[:16], r['Line'], r['Fault Type'], r['Location'], r['Clearing (ms)'], str(r['Auto-Reclose'])[:22],
                     r['Governing Aspect'], f"{r['Overall Risk (%)']:.0f}", r['Status']])
    t = table(rows, [W * x for x in [0.08, 0.12, 0.14, 0.06, 0.1, 0.07, 0.13, 0.15, 0.06, 0.09]])
    for i, (_, r) in enumerate(R.iterrows(), 1):
        t.setStyle(TableStyle([('TEXTCOLOR', (9, i), (9, i), SCOL[r['Status']])]))
    S.append(P('All incidents', H2)); S.append(t)
    above = R[R['Status'] != 'EXCELLENT']
    if len(above):
        S.append(P('Incidents needing attention', H2))
        for _, r in above.iterrows():
            block = [Paragraph(f"<b>{clean(r['Incident'])}</b>  -  {clean(r['Line'])}  -  {clean(r['Governing Aspect'])}  -  {r['Overall Risk (%)']:.0f}% {r['Status']}", BODY),
                     P(r['Summary'])] + [P(f"- {f}") for f in r['Findings']]
            block += [P(f"Action: {r['Action']}", SMALL), Spacer(1, 5)]
            S.append(KeepTogether(block))
    doc.build(S, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()
