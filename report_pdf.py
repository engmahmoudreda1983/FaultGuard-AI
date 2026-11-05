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
H3 = ParagraphStyle('h3', parent=_ss['Heading3'], fontName='Helvetica-Bold', fontSize=10, textColor=NAVY, spaceBefore=6, spaceAfter=3)
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


wave_fig, rx_fig, line_fig = fg.wave_fig, fg.rx_fig, fg.line_fig


def incident_pdf(a, rec):
    meta = a.get('meta', {})
    eq = meta.get('Line') or 'Protected equipment'
    is_line = meta.get('Equipment Type', 'Line') in ('Line', 'Cable')
    buf, doc, deco = _doc(f"Trip Report - {eq}")
    S = [P('Trip Report', H1), Paragraph(f"<font size='13' color='#0b2e59'><b>{clean(eq)}</b></font>", BODY), Spacer(1, 3),
         P(f"Incident {meta.get('Incident', '')}  -  {meta.get('Date', '')}  -  relays: {meta.get('Relay', '')}", SMALL), Spacer(1, 4)]
    S.append(banner(fg.fault_title(a), fg.summary_text(a), 'INFO'))
    if a.get('fault') and is_line and not a.get('no_line_data'):
        S.append(P('Fault location on the line', H2))
        S.append(fig_image(line_fig(a)))
        LE = fg.location_explanation(a)
        if LE:
            rows_, txt_ = LE
            S.append(P('How the fault location was found', H3))
            S.append(table(rows_, [W * 0.24, W * 0.18, W * 0.18, W * 0.16, W * 0.24]))
            for t_ in txt_:
                S.append(P(t_, SMALL))
    two_ends = len(fg.end_views(a)) > 1 if a.get('fault') else False
    if a.get('fault'):
        tl = fg.timeline_fig(a, rec)
        if tl is not None and not two_ends:
            S.append(fig_image(tl))
        S.append(PageBreak())
        TS = fg.source_tables(a)
        if TS:
            S.append(P('Results by source', H2))
            S.append(P('Each source is shown on its own: FaultGuard AI from the disturbance records, and each relay from its own fault record. '
                       'Every location is also given from the same line end so that they can be compared.', SMALL))
            for k_, T_ in TS.items():
                if k_ == 'summary':
                    continue
                cols = list(T_.columns)
                S.append(KeepTogether([P(k_ + (' (disturbance records)' if k_ == 'FaultGuard AI' else ' (relay fault records)'), H3),
                                       table([cols] + T_.astype(str).values.tolist(), [W * 0.4] + [W * 0.6 / (len(cols) - 1)] * (len(cols) - 1))]))
            T_ = TS['summary']; cols = list(T_.columns)
            S.append(KeepTogether([P('Summary: average and spread of the sources', H3),
                                   table([cols] + T_.astype(str).values.tolist(), [W * 0.27] + [W * 0.11] * (len(cols) - 2) + [W * (0.73 - 0.11 * (len(cols) - 2))])]))
            S.append(P('Relay values are as recorded by each relay in its own fault record: its trip time counts from its own start (0 = it tripped as it started) '
                       'and its breaker time from the trip command to pole dead. FaultGuard counts from the fault inception in the record and the breaker time from '
                       'the trip to current zero. Relay locations in the summary are the average of the two ends of the same relay type. '
                       'Spread = half the difference between the highest and lowest value.', SMALL))
        PR = fg.protection_by_relay(a) if fg.end_views(a) else None
        if PR is not None and len(PR):
            cols = list(PR.columns)
            S.append(KeepTogether([P('Protection operation by relay (ms from the fault inception at each end)', H3),
                                   table([cols] + PR.astype(str).replace({'nan': '-', 'None': '-'}).values.tolist(),
                                         [W * x for x in (0.08, 0.07, 0.12, 0.08, 0.09, 0.27, 0.19, 0.1)])]))
            S.append(P("'Record' = the relay's own disturbance record and event list; 'event list (aligned)' = its event list placed on the record of the same "
                       'station (its clock is not synchronised, so its times are within about one cycle).', SMALL))
            if two_ends and tl is not None:
                S.append(P('Sequence of events at both ends (ms from the fault inception)', H3))
                S.append(fig_image(tl))
        else:
            S.append(P('Fault analysis: FaultGuard AI against the relays', H2))
            C = fg.compare_table(a)
            cols = list(C.columns)
            wq = 0.17; wr = (1 - wq) / (len(cols) - 1)
            S.append(table([cols] + C.astype(str).values.tolist(), [W * wq] + [W * wr] * (len(cols) - 1)))
        if a.get('restored'):
            rs = a['restored']
            S.append(P(f"Line re-energised at {pd.Timestamp(rs['time']):%Y-%m-%d %H:%M:%S}, {fg.fmt_dur(rs['after_s'])} after the fault ({rs['device']}: {rs['event']}).", SMALL))
        for k_, (r_, a_) in enumerate([(rec, a)] + [(r2, a2) for _, r2, a2 in (a.get('other_records') or [])]):
            S.append(PageBreak())
            S.append(P(f"Disturbance record {k_ + 1}: {fg.record_source(r_, a_)} - currents, voltages and relay contacts", H2))
            S.append(fig_image(wave_fig(r_, a_), W if k_ == 0 else W))
        done_st = set()
        for r_, x_ in [(rec, a)] + [(r2, a2) for _, r2, a2 in (a.get('other_records') or [])]:
            st_ = fg.short_station(x_.get('station')) or ''
            if x_.get('traj') is None or not is_line or a.get('no_line_data') or st_ in done_st:
                continue                       # one R-X page per line end
            done_st.add(st_)
            xx = dict(x_); mm = dict(meta)
            if mm.get('End A') and mm.get('End B') and st_ and fg.short_station(mm['End A']) != st_ and fg.short_station(mm['End B']) == st_:
                mm['End A'], mm['End B'] = mm['End B'], mm['End A']        # the relay's own station first
            xx['meta'] = mm
            dev = (x_.get('record_device') or '').split(' ')[0] or 'relay'
            sdev = x_.get('settings_dev') or dev
            S.append(PageBreak())
            S.append(P(f"Impedance seen by the relay at {st_ or 'the local end'} ({dev}) - R-X", H2))
            S.append(P(f"Measured from the {st_} disturbance record; zones from the {sdev} {st_} settings. The line is drawn from {st_} towards "
                       f"{fg.short_station(mm.get('End B')) or 'the remote end'}.", SMALL))
            S.append(fig_image(rx_fig(xx), W * 0.92))
            zs = x_.get('zone_seen')
            if zs is not None:
                S.append(P('The star marks the impedance one cycle into the fault; the blue dots show its path from the load point into the fault. '
                           + ('It lies outside every zone.' if zs == 0 else f"It lies inside zone {zs}."), SMALL))
    else:
        for n in a.get('notes', []):
            S.append(P(n))
    S.append(P('Incident', H2))
    S.append(table([['Item', 'Value', 'Item', 'Value'],
                    ['Protected equipment', eq, 'Type / number', f"{meta.get('Equipment Type', '-')} {meta.get('Equipment', '')}"],
                    ['Station A (local)', meta.get('End A', '-') or '-', 'Station B (remote)', meta.get('End B', '-') or '-'],
                    ['Relays', meta.get('Relay', '-') or '-', 'Record start (relay clock)',
                     ' / '.join(f"{st_} {str(x_.get('record_start') or '-')[11:23]}" for st_, _, x_ in fg.end_views(a)) if two_ends else str(a.get('record_start') or '-')[:23]],
                    (['Line length', 'not known (no settings file)', 'Line impedance (per km)', 'not known (no settings file)'] if a.get('no_line_data') else
                     ['Line length', f"{a['line']['length_km']:.1f} km", 'Line impedance (per km)', f"Z1 {a['line']['r1']:.3f}+j{a['line']['x1']:.3f}, Z0 {a['line']['r0']:.3f}+j{a['line']['x0']:.3f} ohm"])],
                   [W * 0.16, W * 0.34, W * 0.18, W * 0.32]))
    if a.get('fault'):
        V_ = fg.end_views(a) or [(fg.short_station(a.get('station')) or 'record', rec, a)]
        hdr = ['Channel']
        for st_, _, _ in V_:
            hdr += [f'{st_} pre-fault', f'{st_} fault', f'{st_} angle']
        rows = [hdr]
        for k in ('IA', 'IB', 'IC', 'IN', 'VA', 'VB', 'VC'):
            r_ = [k]; u = 'kA' if k.startswith('I') else 'kV'
            for _, _, x_ in V_:
                if k in x_['P']:
                    pre, flt = x_['P'][k]
                    r_ += [f"{abs(pre) / 1000:.3f} {u}", f"{abs(flt) / 1000:.3f} {u}", f"{np.degrees(np.angle(flt)):.0f} deg"]
                else:
                    r_ += ['-', '-', '-']
            rows.append(r_)
        wc = (W * 0.1, ) + tuple([W * 0.9 / (3 * len(V_))] * (3 * len(V_)))
        S.append(KeepTogether([P('Measured values (primary rms, one cycle before and one cycle into the fault)', H2), table(rows, list(wc))]))
        if len(V_) > 1:
            S.append(P('Fault angles are relative to each record\'s own time reference: the two records are not time-synchronised, so angles are '
                       'not comparable between the two ends.', SMALL))
        if len(a.get('records') or []) > 1:
            allx = [(rec, a)] + [(r2, a2) for _, r2, a2 in (a.get('other_records') or [])]
            rr = []
            for r_, x_ in allx:
                rr.append([fg.record_source(r_, x_).replace(' disturbance record', ''), str(x_.get('record_start') or '-')[:23],
                           str(x_.get('type') if x_.get('fault') else 'no fault'),
                           '-' if x_.get('loc_single_km', x_.get('loc_km')) is None else f"{(x_.get('loc_single_km') if x_ is a and x_.get('loc_single_km') is not None else x_.get('loc_km')):.1f}",
                           f"{x_.get('I_fault_max_kA', 0):.2f}", '-' if x_.get('operate_ms') is None else f"{x_['operate_ms']:.1f}",
                           '-' if x_.get('breaker_ms') is None else f"{x_['breaker_ms']:.1f}", '-' if x_.get('clearing_ms') is None else f"{x_['clearing_ms']:.1f}"])
            S.append(KeepTogether([P('Disturbance records of this incident', H2),
                                   table([['Record (relay)', 'Record start (relay clock)', 'Fault type', 'Location from its own end (km)', 'Fault current (kA)',
                                           'Trip (ms)', 'Breaker (ms)', 'Clearing (ms)']] + rr,
                                         [W * x for x in (0.17, 0.2, 0.08, 0.13, 0.1, 0.1, 0.11, 0.11)]),
                                   P('Each record start is shown as written by its relay clock. Trip, breaker and clearing times count from the fault inception '
                                     'in that record.', SMALL)]))
        if a.get('settings_all'):
            T_ = fg.settings_table(a['settings_all'])
            T_ = T_[(T_.iloc[:, 1:] != '-').any(axis=1)]
            S.append(KeepTogether([P('Relay settings used', H2),
                                   table([list(T_.columns)] + T_.astype(str).values.tolist(), [W * 0.3] + [W * 0.7 / (len(T_.columns) - 1)] * (len(T_.columns) - 1))]))
    S.append(P('Assessment', H2))
    if not a.get('likely_test'):
        c_ = SCOL.get(a['Status'], NAVY).hexval()
        S.append(Paragraph(f"<font color='{c_}' size='12'><b>{clean(fg.risk_title(a, SLAB[a['Status']]))}</b></font>", BODY))
        S.append(P(fg.risk_note(a), SMALL))
    for r in fg.findings(a):
        S.append(P(f"- {r}"))
    S.append(Paragraph(f"<b>Action:</b> {clean(a['Action'])}", BODY))
    nts = [n for n in a.get('notes', []) if not (n.startswith('Single-ended locations') and fg.source_tables(a))] if a.get('fault') else []
    if nts:
        S.append(P('Notes', H3))
        for n in nts:
            S.append(P(f"- {n}", SMALL))
    if (a.get('engineer_comments') or '').strip():
        S.append(P("Engineer's comments", H2))
        for ln in a['engineer_comments'].strip().splitlines():
            if ln.strip():
                S.append(P(ln.strip()))
    S.append(P('FMEA', H2))
    S.append(table([['Aspect', 'Risk %', 'S', 'O', 'D', 'RPN', 'Level']] + [[m['Failure Mode'], f"{m['Risk (%)']:.0f}", m['S'], m['O'], m['D'], m['RPN'], m['RPN Level']] for m in a['FMEA']],
                   [W * 0.34, W * 0.12, W * 0.08, W * 0.08, W * 0.08, W * 0.12, W * 0.18]))
    V_ = fg.end_views(a) if a.get('fault') else []
    for st_, r_, x_ in (V_ if len(V_) > 1 else [(None, rec, a)]):
        so = fg.soe(x_)
        if len(so):
            inc = float(x_.get('t_inception_ms') or 0)
            rows = [['Clock time (record)' if 'Clock Time' in so else '#', 'ms from fault', 'Source', 'Event']] + \
                   [[r.get('Clock Time', i + 1), f"{r['Time (ms)'] - inc:.1f}", r['Source'], r['Event']] for i, r in so.head(70).iterrows()]
            S.append(P('Sequence of events' + (f' - {st_} end' if st_ else ''), H2))
            S.append(P('Clock time as written by this record; ms counted from the fault inception in this record. Event lists of relays with another clock '
                       'are placed on this record.', SMALL))
            S.append(table(rows, [W * 0.24, W * 0.11, W * 0.15, W * 0.5]))
    S.append(PageBreak())
    S.append(P('Appendix: FMEA reference - how S, O, D and the risks are set', H2))
    S.append(P(fg.FMEA_BASIS, SMALL)); S.append(Spacer(1, 4))
    F = fg.fmea_reference(a)
    hdr = ['Failure mode', 'Definition', 'S', 'Risk (%) criteria', 'D'] + (['This incident'] if 'This incident' in F else [])
    wd = [0.13, 0.17, 0.04, 0.4, 0.04] + ([0.22] if 'This incident' in F else [])
    S.append(table([hdr] + F[hdr].astype(str).values.tolist(), [W * x for x in wd]))
    S.append(Spacer(1, 6))
    S.append(table([['Scale', 'Rule']] + [list(r) for r in fg.FMEA_SCALES], [W * 0.22, W * 0.78]))
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
    rows = [['Incident', 'Date', 'Line', 'Type', 'Location', 'Clearing', 'Reclose', 'Main reason', 'Risk', 'Status']]
    for _, r in R.iterrows():
        rows.append([r['Incident'], str(r['Date'])[:16], r['Line'], r['Fault Type'], r['Location'], r['Clearing (ms)'], str(r['Auto-Reclose'])[:22],
                     f"{r['Highest-Risk Mode']}: {r.get('Main Reason', '')}", f"{r['Overall Risk (%)']:.0f}", r['Status']])
    t = table(rows, [W * x for x in [0.08, 0.12, 0.14, 0.06, 0.1, 0.07, 0.13, 0.15, 0.06, 0.09]])
    for i, (_, r) in enumerate(R.iterrows(), 1):
        t.setStyle(TableStyle([('TEXTCOLOR', (9, i), (9, i), SCOL[r['Status']])]))
    S.append(P('All incidents', H2)); S.append(t)
    above = R[R['Status'] != 'EXCELLENT']
    if len(above):
        S.append(P('Incidents needing attention', H2))
        for _, r in above.iterrows():
            block = [Paragraph(f"<b>{clean(r['Incident'])}</b>  -  {clean(r['Line'])}  -  {r['Overall Risk (%)']:.0f}% {r['Status']}  -  main reason: {clean(r['Highest-Risk Mode'])} ({clean(r.get('Main Reason', ''))})", BODY),
                     P(r['Summary'])] + [P(f"- {f}") for f in r['Findings']]
            block += [P(f"Action: {r['Action']}", SMALL), Spacer(1, 5)]
            S.append(KeepTogether(block))
    doc.build(S, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()
