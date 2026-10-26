# FaultGuard AI: Trip and Incident Analysis

A PowerGuard AI module. It reads a relay disturbance record (COMTRADE .cfg/.dat or CSV) and the relay event lists
(CSV, TXT or PDF exports such as MiCOM S1 Agile event reports, one or more relays) and writes the incident report:
fault type (AI model checked by the classic phase selection), fault location, protection and breaker times, the elements
that tripped, auto-reclose outcome and restoration, relay clock alignment, an incident risk with FMEA and actions.

## Pages
- **Incidents Overview**: results from the analysis page, or the demo substation (7 simulated incidents).
- **Incident Analysis**: demo incident or uploaded files, incident data, line data (per-km sequence impedances or the
  relay settings: line impedance, angle, kZN), protection settings and scaling. PDF report and CSV downloads.
- **AI Fault Classifier & Accuracy**: accuracy on unseen simulated incidents, confusion matrix, location error.
- **Combined Report**: PDF, text, table and FMEA CSV.
- **Incident Statistics**: faults by line, type and month, auto-reclose success, repeated faults at the same place.

## Files
- `faultguard_core.py`: engine (record readers, phasors, fault detection, AI type, Takagi location, event parsing and
  clock alignment, assessment, FMEA, demo incidents, model training).
- `train_faultguard_model.py`: trains and tests the fault-type model. `faultguard_model.pkl`: trained model.
- `report_pdf.py`: PDF reports. `FaultGuard_AI.app.py`: Streamlit app. `comtrade_features.py`: COMTRADE reader.

Login credentials are read from the Streamlit secrets `APP_USERNAME` and `APP_PASSWORD`. Real relay records and event
lists must not be committed to this repository.

© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. Confidential and proprietary.
