# DiabVox — Streamlit Edition

AI-powered diabetes intelligence app, fully ported from Flask + Vanilla JS to Streamlit.

## Features

- **5-step onboarding** with two pathways (A: voice risk test / B: known diagnosis)
- **Real voice analysis** via `st.audio_input()` → librosa → SVM + MLP + RF ensemble
- **Glucose prediction** via Random Forest on 8 PPG-correlated features
- **10-tab dashboard**: Voice, Glucose, Meal Plan, Medications, Meal Log, Activity Log, Medical Records, Wellness, Care Team, Risk Analytics
- **10-year risk projection** & complication risk modelling
- **Data export** (JSON + CSV)
- All state stored in `st.session_state` (no database required)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
streamlit run diabvox_app.py
```

App opens at **http://localhost:8501**

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io → **New app**
3. Select your repo, set **Main file path** to `diabvox_app.py`
4. Click **Deploy**

No secrets or environment variables required.

## Notes

- Voice recording requires microphone permission in the browser
- ML models are trained on synthetic data at startup (~5 sec, cached)
- Session data is in-memory only — refreshing the page resets the session
  (add a database backend for persistence in production)

## Stack

| Component | Technology |
|-----------|-----------|
| Framework | Streamlit 1.46+ |
| Voice ML  | SVM + MLP + RandomForest (scikit-learn) |
| Voice features | librosa |
| Glucose ML | Random Forest Regressor |
| Charts | Plotly |
| State | `st.session_state` |
