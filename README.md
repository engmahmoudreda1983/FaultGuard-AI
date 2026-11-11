# FaultGuard AI: Trip and Incident Analysis

A PowerGuard AI module. It reads the files of every relay involved in a trip and merges them into one incident report:
- disturbance records: COMTRADE .cfg + .dat (or .zip, or CSV), any number of relays (two line ends give a two-ended fault location);
- relay settings: the MiCOM settings file itself (.set from S1 Agile / S1 Studio: P44x, P54x and other Courier relays, read
  cell by cell from the active setting group: line data, CT/VT, distance zones, zone times, resistive reaches, characteristics),
  or a settings export as PDF, scanned PDF, picture (OCR), CSV, TXT or Excel, any number of relays;
- event lists: PDF, scanned PDF, picture (read by OCR), CSV, TXT or Excel, any number of relays.

The report gives:
fault type (AI model checked by the classic phase selection), fault location, protection and breaker times, the elements
that tripped, the distance zone the impedance entered (quadrilateral or mho zones from the settings), restoration time, relay clock alignment, settings used, an incident risk with FMEA and actions. The external
auto-recloser is normally out of service, so auto-reclose is reported but not assessed unless it is marked in service.

## Pages
- **Incidents Overview**: results from the analysis page, or the demo substation (7 simulated incidents).
- **Incident Analysis**: demo incident or uploaded files, incident data, line data (per-km sequence impedances or the
  relay settings: line impedance, angle, kZN), protection settings and scaling. PDF report and CSV downloads.
- **Combined Report**: PDF, text, table and FMEA CSV.
- **Incident Statistics**: faults by line, type and month, repeated faults at the same place, and FaultGuard against the relays' own
  fault records (fault type agreement and location difference) over all analyzed incidents.

## Files
- `faultguard_core.py`: engine (record readers, phasors, fault detection, AI type, Takagi location, event parsing and
  clock alignment, assessment, FMEA, demo incidents, model training).
- `train_faultguard_model.py`: trains and tests the fault-type model. `faultguard_model.pkl`: trained model.
- `packages.txt`: system package for OCR (`tesseract-ocr`) on Streamlit Cloud.
- `micom_set.py`: reader for MiCOM .set settings files (read only, the same reader as the platform's `pg_platform/readers`).
- `report_pdf.py`: PDF reports. `FaultGuard_AI.app.py`: Streamlit app. `comtrade_features.py`: COMTRADE reader.

Login credentials are read from the Streamlit secrets `APP_USERNAME` and `APP_PASSWORD`. Real relay records and event
lists must not be committed to this repository.

© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. Confidential and proprietary.
