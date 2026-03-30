@echo off
setlocal

cd /d "%~dp0"

if not exist integration\integration_output.csv (
  echo [DEMO] integration/integration_output.csv not found. Running full pipeline...
  python -u run_pipeline.py
) else (
  echo [DEMO] Found integration/integration_output.csv. Skipping pipeline run.
)

echo.
echo [DEMO] Integrated dataset size and severity breakdown:
python -c "import pandas as pd; df=pd.read_csv('integration/integration_output.csv'); print('rows', len(df)); print('severity', df['Severity'].value_counts().to_dict())"

echo.
echo [DEMO] Now run the Step 5 dashboard:
echo streamlit run integration\dashboard.py

endlocal

