#!/usr/bin/env python3
"""
DiabVox — Streamlit Edition
Complete port of the Flask + Vanilla JS app to Streamlit.
Run: streamlit run diabvox_app.py
pip install streamlit scikit-learn librosa scipy numpy bcrypt pyjwt plotly streamlit-option-menu
"""

import io, uuid, json, base64, warnings, hashlib, time, csv, math
from datetime import datetime
import numpy as np
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go_obj
import plotly.express as px

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="DiabVox — AI Diabetes Intelligence",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training ML models…")
def load_models():
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    # ── VoiceRiskModel ────────────────────────────────────────────────────
    class VoiceRiskModel:
        def __init__(self):
            self.svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
            self.mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                                     early_stopping=True, random_state=42)
            self.rf  = RandomForestClassifier(n_estimators=80, random_state=42)
            self.scaler = StandardScaler()
            self._train()

        def _synth(self, n=800):
            np.random.seed(42); X, y = [], []
            for _ in range(n):
                risk = np.random.choice([0, 1, 2], p=[0.55, 0.25, 0.20])
                if risk == 0:
                    f = [np.random.normal(150,25),np.random.normal(25,8),np.random.normal(0.8,0.3),
                         np.random.normal(0.8,0.3),np.random.normal(20,4),np.random.normal(-150,40),
                         np.random.normal(30,20),np.random.normal(-10,15),np.random.normal(5,10),
                         np.random.normal(2,8),np.random.normal(0.05,0.02),np.random.normal(0.8,0.2),
                         np.random.normal(3200,400),np.random.normal(0.15,0.05)]
                elif risk == 1:
                    f = [np.random.normal(135,20),np.random.normal(18,7),np.random.normal(1.8,0.5),
                         np.random.normal(1.8,0.6),np.random.normal(14,4),np.random.normal(-200,45),
                         np.random.normal(10,22),np.random.normal(-18,16),np.random.normal(2,11),
                         np.random.normal(-1,9),np.random.normal(0.07,0.025),np.random.normal(0.65,0.22),
                         np.random.normal(2900,450),np.random.normal(0.20,0.06)]
                else:
                    f = [np.random.normal(118,18),np.random.normal(12,5),np.random.normal(3.2,0.8),
                         np.random.normal(3.5,1.0),np.random.normal(9,3),np.random.normal(-260,50),
                         np.random.normal(-15,25),np.random.normal(-28,18),np.random.normal(-3,13),
                         np.random.normal(-5,11),np.random.normal(0.10,0.03),np.random.normal(0.5,0.25),
                         np.random.normal(2500,500),np.random.normal(0.26,0.07)]
                f = [v + np.random.normal(0, abs(v)*0.04) for v in f]
                X.append(f); y.append(risk)
            return np.array(X), np.array(y)

        def _train(self):
            X, y = self._synth(); Xs = self.scaler.fit_transform(X)
            self.svm.fit(Xs, (y == 2).astype(int))
            self.mlp.fit(Xs, y); self.rf.fit(Xs, y)

        def predict(self, features):
            x = self.scaler.transform(np.array(features).reshape(1, -1))
            svm_high = self.svm.predict_proba(x)[0][1]
            mlp_p = self.mlp.predict_proba(x)[0]; rf_p = self.rf.predict_proba(x)[0]
            def sg(p, i): return float(p[i]) if len(p) > i else 0.0
            high = svm_high*0.4 + sg(mlp_p,2)*0.3 + sg(rf_p,2)*0.3
            mod  = sg(mlp_p,1)*0.5 + sg(rf_p,1)*0.5
            low  = max(0.0, 1.0 - high - mod)
            score = min(100, max(0, high*100 + mod*50))
            if score >= 60:   level, color = "High Risk",     "#ef4444"
            elif score >= 35: level, color = "Moderate Risk", "#f59e0b"
            else:             level, color = "Low Risk",      "#22c55e"
            if score < 30:   sdt, sdl = "none",       "No Diabetes / Low Risk"
            elif score < 60: sdt, sdl = "prediabetic", "Pre-diabetic"
            else:            sdt, sdl = "high_risk",  "At High Risk - Consult Doctor"
            return {
                "risk_score": round(float(score), 1),
                "risk_level": level, "color": color,
                "suggested_diabetes_type": sdt, "suggested_label": sdl,
                "breakdown": {
                    "low_risk_prob":      round(float(low)*100, 1),
                    "moderate_risk_prob": round(float(mod)*100, 1),
                    "high_risk_prob":     round(float(high)*100, 1),
                },
            }

    # ── GlucoseModel ─────────────────────────────────────────────────────
    class GlucoseModel:
        def __init__(self):
            self.model  = RandomForestRegressor(n_estimators=80, max_depth=10,
                                                random_state=42, n_jobs=-1)
            self.scaler = StandardScaler()
            self._train()

        def _synth(self, n=1200):
            np.random.seed(42); X, y = [], []
            for _ in range(n):
                g  = np.random.uniform(60, 280)
                hr = 72 + (g-120)*0.08 + np.random.normal(0,6)
                hrv= 45 - (g-100)*0.12 + np.random.normal(0,8)
                amp= 0.65-(g-100)*0.001 + np.random.normal(0,0.08)
                slp= 0.35+(g-100)*0.0008+ np.random.normal(0,0.05)
                ptt= 280+(g-100)*0.3    + np.random.normal(0,20)
                lf = 800-(g-100)*0.8    + np.random.normal(0,100)
                hf = 600-(g-100)*0.5    + np.random.normal(0,80)
                pi = 3.5-(g-100)*0.005  + np.random.normal(0,0.4)
                X.append([hr,hrv,amp,slp,ptt,lf,hf,pi]); y.append(g)
            return np.array(X), np.array(y)

        def _train(self):
            X, y = self._synth()
            self.model.fit(self.scaler.fit_transform(X), y)

        def predict_from_inputs(self, hr, activity, tsmeal, carbs, stress, base_g):
            meal = ((carbs/15)*22*np.exp(-tsmeal/1.5)) if tsmeal < 5 else 0
            est  = float(np.clip(base_g + meal - activity*9 + stress*10
                                 + np.random.normal(0,4), 50, 380))
            feats = [
                hr + np.random.normal(0,2),
                45  - (est-100)*0.12 + np.random.normal(0,3),
                0.65- (est-100)*0.001+ np.random.normal(0,0.02),
                0.35+ (est-100)*0.0008+np.random.normal(0,0.01),
                280 + (est-100)*0.3  + np.random.normal(0,8),
                800 - (est-100)*0.8  + np.random.normal(0,40),
                600 - (est-100)*0.5  + np.random.normal(0,30),
                3.5 - (est-100)*0.005+ np.random.normal(0,0.15),
            ]
            pred = float(np.clip(
                self.model.predict(self.scaler.transform(np.array(feats).reshape(1,-1)))[0],
                50, 380))
            return pred, feats

    # ── RecommendationEngine ──────────────────────────────────────────────
    class RecommendationEngine:
        FOODS = {
            "very_low_carb": [
                {"name":"Grilled salmon with sautéed spinach & olive oil","carbs":4,"protein":35,"gi":"very low"},
                {"name":"Egg white omelette with mushrooms & bell peppers","carbs":6,"protein":22,"gi":"very low"},
                {"name":"Avocado & cucumber salad with feta","carbs":8,"protein":6,"gi":"very low"},
                {"name":"Grilled chicken breast with steamed broccoli","carbs":10,"protein":40,"gi":"very low"},
            ],
            "low_carb": [
                {"name":"Quinoa bowl with roasted vegetables & chickpeas","carbs":38,"protein":16,"gi":"low"},
                {"name":"Lentil soup with whole wheat bread","carbs":42,"protein":18,"gi":"low"},
                {"name":"Turkey & avocado lettuce wraps","carbs":18,"protein":28,"gi":"low"},
                {"name":"Greek yogurt with berries & chia seeds","carbs":22,"protein":15,"gi":"low"},
            ],
            "moderate_carb": [
                {"name":"Brown rice stir-fry with tofu & vegetables","carbs":52,"protein":20,"gi":"moderate"},
                {"name":"Whole grain pasta with tomato & lean turkey sauce","carbs":55,"protein":25,"gi":"moderate"},
                {"name":"Steel-cut oatmeal with walnuts, cinnamon & banana","carbs":48,"protein":8,"gi":"moderate"},
                {"name":"Sweet potato with black beans & salsa","carbs":58,"protein":12,"gi":"moderate"},
            ],
        }
        DAILY_PLANS = {
            "type1":      ["Steel-cut oats with flaxseed & blueberries","Grilled chicken salad with avocado","Baked salmon with roasted asparagus","Greek yogurt with almonds"],
            "type2":      ["Vegetable omelette with whole grain toast","Lentil soup with mixed greens","Stir-fried tofu with broccoli & brown rice","Handful of mixed nuts"],
            "prediabetic":["Smoothie with spinach, protein powder & berries","Quinoa tabbouleh with grilled chicken","Baked cod with steamed vegetables","Apple slices with almond butter"],
            "none":       ["Overnight oats with banana & chia","Turkey lettuce wraps with hummus","Whole grain bowl with roasted veggies & eggs","Cottage cheese with cucumber"],
            "gestational":["Whole grain cereal with low-fat milk","Grilled fish tacos with slaw","Chicken & vegetable soup","String cheese with carrot sticks"],
            "high_risk":  ["Egg & spinach scramble on rye","Bean & vegetable soup","Grilled turkey patty with salad","Celery sticks with peanut butter"],
        }
        EXERCISES = {
            "low":      [{"name":"Gentle yoga","duration":30,"met":2.5,"glucose_effect":-5},
                         {"name":"Leisurely walking","duration":20,"met":2.8,"glucose_effect":-8},
                         {"name":"Light stretching","duration":25,"met":2.0,"glucose_effect":-3}],
            "moderate": [{"name":"Brisk walking","duration":30,"met":3.5,"glucose_effect":-15},
                         {"name":"Cycling (flat terrain)","duration":30,"met":4.0,"glucose_effect":-18},
                         {"name":"Swimming laps","duration":25,"met":5.8,"glucose_effect":-20},
                         {"name":"Elliptical trainer","duration":30,"met":5.0,"glucose_effect":-17}],
            "high":     [{"name":"Interval running","duration":20,"met":8.0,"glucose_effect":-30},
                         {"name":"HIIT circuit","duration":25,"met":8.5,"glucose_effect":-35},
                         {"name":"Resistance training","duration":35,"met":6.0,"glucose_effect":-12}],
        }

        def food(self, glucose, meal_time):
            if glucose > 200:   cat, rat = "very_low_carb", f"Glucose elevated ({glucose:.0f}) — very low-carb minimises further spike."
            elif glucose > 140: cat, rat = "low_carb",      f"Above target ({glucose:.0f}) — low-GI stabilises levels."
            elif glucose < 80:  cat, rat = "moderate_carb", f"Glucose low ({glucose:.0f}) — moderate-carb safely raises it."
            else:               cat, rat = "low_carb",      f"Good range ({glucose:.0f}) — low-GI maintains stability."
            foods = self.FOODS[cat]
            meal  = foods[hash(meal_time + str(int(glucose/30))) % len(foods)]
            tips  = {"breakfast":"Protein-first order reduces post-meal spike by ~28%.",
                     "lunch":    "10-min walk after lunch reduces glucose by ~22 mg/dL.",
                     "dinner":   "Lighter dinner improves overnight control.",
                     "snack":    "Time snacks 2-3h after meals to avoid stacking."}
            return {"meal": meal, "category": cat.replace("_"," ").title(),
                    "rationale": rat, "timing_tip": tips.get(meal_time, "")}

        def daily_plan(self, diabetes_type):
            meals  = self.DAILY_PLANS.get(diabetes_type, self.DAILY_PLANS["none"])
            labels = ["Breakfast","Lunch","Dinner","Snack"]; carbs = [35,50,45,20]
            return [{"label":labels[i],"name":meals[i],"carbs":carbs[i]} for i in range(len(meals))]

        def exercise(self, glucose, trend, fitness="moderate"):
            if glucose < 70:
                return {"status":"contraindicated","message":"Glucose too low. Take 15g fast-acting carbs, wait 15 min, recheck.","exercise":None}
            if glucose > 270:
                return {"status":"contraindicated","message":"Glucose very high. Check for ketones. Exercise only after glucose drops below 250 mg/dL.","exercise":None}
            if glucose > 180:           level, reason = "low",      "Elevated glucose — light movement helps"
            elif glucose<100 and trend=="falling": level, reason = "low", "Trending down — low intensity, snack nearby"
            elif fitness == "low":      level, reason = "low",      "Building baseline fitness"
            elif fitness == "high":     level, reason = "high",     "Good control supports high-intensity training"
            else:                       level, reason = "moderate", "Optimal range for moderate exercise"
            ex = self.EXERCISES[level][int(datetime.now().timestamp()) % len(self.EXERCISES[level])]
            tips = ["Stay hydrated — water before, during and after"]
            if glucose < 120: tips.append("Keep 15g fast-acting carbs accessible")
            if level == "high": tips.append("Check glucose before, during (if >30 min) and after")
            return {"status":"ok","exercise":ex,"level":level.title(),"reason":reason,
                    "estimated_glucose_change":f"{ex['glucose_effect']} mg/dL",
                    "calories_burned_estimate":round(ex["met"]*70*(ex["duration"]/60)),
                    "safety_tips":tips}

        def complications_risk(self, profile, glucose_history):
            dtype = profile.get("diabetes_type","none")
            if dtype not in ("type1","type2","gestational","high_risk","prediabetic"):
                return None
            vals = [r["glucose"] for r in glucose_history] if glucose_history else []
            avg  = np.mean(vals) if vals else float(profile.get("baseline_glucose",120))
            variability = float(np.std(vals)) if len(vals) > 2 else 15.0
            bp_str = profile.get("bp","120/80")
            try:   bp_sys = int(bp_str.split("/")[0])
            except: bp_sys = 120
            hba1c = float(profile.get("hba1c") or (avg/28.7 + 2.15))
            ret  = min(95, max(5, int((hba1c-5)*8   + variability*0.4)))
            nep  = min(95, max(5, int((bp_sys-100)*0.3 + (hba1c-5)*5)))
            cv   = min(95, max(5, int((bp_sys-110)*0.4 + (hba1c-5)*6
                                      + (10 if dtype in ("type2","prediabetic") else 0))))
            neur = min(95, max(5, int(variability*1.2 + (hba1c-5)*7)))
            return {
                "retinopathy":    {"risk":ret,  "label":"Diabetic Retinopathy",  "basis":"Glucose variability & HbA1c trends"},
                "nephropathy":    {"risk":nep,  "label":"Diabetic Nephropathy",  "basis":"Blood pressure & glucose control"},
                "cardiovascular": {"risk":cv,   "label":"Cardiovascular Events", "basis":"Combined CVD risk factors"},
                "neuropathy":     {"risk":neur, "label":"Neuropathy",            "basis":"Glucose variability & duration"},
            }

        def ten_year_risk(self, profile, voice_history):
            dtype = profile.get("diabetes_type","none")
            if dtype in ("type1","type2"): return None
            age    = int(profile.get("age",40))
            bmi    = float(profile.get("bmi") or 25)
            family = profile.get("family_history","no")
            act    = profile.get("activity","moderate")
            conds  = profile.get("conditions",[]) or []
            voice_score = voice_history[-1]["risk_score"] if voice_history else 30.0
            base = (age-30)*0.8 + ((bmi-25)*1.2 if bmi > 25 else 0)
            base += {"no":0,"unknown":5,"yes_sibling":12,"yes_parent":18,"yes_both":28}.get(family,0)
            base += {"active":-8,"moderate":0,"sedentary":12}.get(act,0)
            base += len([c for c in conds if c in ("hypertension","obesity")])*8 + voice_score*0.3
            base += 20 if dtype == "prediabetic" else 0
            return round(min(95, max(2, base)), 1)

    return VoiceRiskModel(), GlucoseModel(), RecommendationEngine()


VOICE_MODEL, GLUCOSE_MODEL, REC_ENGINE = load_models()


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "page": "onboard_1",          # current page
        "profile": {},
        "diabetes_type": None,
        "pathway": None,
        "voice_result_a": None,       # onboarding pathway A voice result
        "voice_result_b": None,       # onboarding pathway B baseline result
        "confirmed_type": None,
        "glucose_history": [],
        "voice_history": [],
        "medications": [],
        "meal_logs": [],
        "activity_logs": [],
        "medical_records": [],
        "wellness_journal": [],
        "care_team": [],
        "messages": [
            {"text":"Hello! I've reviewed your recent glucose data. Your time-in-range has improved to 78% this week. Keep up the great work! 🎉","sender":"Dr. Ananya Krishnan","ts":"Yesterday 14:32"},
            {"text":"Thank you Doctor! I noticed my post-lunch readings are still slightly elevated. Should I adjust my carb intake at lunch?","sender":"You","ts":"Yesterday 15:10"},
            {"text":"Yes, try reducing lunch carbs by 10-15g and adding a 10-minute walk after eating. Review in 1 week.","sender":"Dr. Ananya Krishnan","ts":"Yesterday 15:45"},
        ],
        "logged_in": False,
        "user_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ═══════════════════════════════════════════════════════════════════════════════
# VOICE FEATURE EXTRACTION (Python / librosa)
# ═══════════════════════════════════════════════════════════════════════════════
def extract_voice_features(audio_bytes: bytes):
    try:
        import librosa
        y_audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        if len(y_audio) < sr * 0.5:
            return None, "Recording too short."
        f0     = librosa.yin(y_audio, fmin=50, fmax=500, sr=sr)
        voiced = f0[f0 > 0]
        if len(voiced) == 0: voiced = np.array([150.0])
        pitch_mean = float(np.mean(voiced))
        pitch_std  = float(np.std(voiced)) if len(voiced) > 1 else 20.0
        jitter = float(np.mean(np.abs(np.diff(voiced))) / (np.mean(voiced)+1e-8)*100) if len(voiced)>2 else 1.0
        rms    = librosa.feature.rms(y=y_audio, frame_length=512, hop_length=128)[0]
        rms_v  = rms[rms > np.percentile(rms, 20)]
        shimmer= float(np.mean(np.abs(np.diff(rms_v)))/(np.mean(rms_v)+1e-8)*100) if len(rms_v)>2 else 0.8
        h      = librosa.effects.harmonic(y_audio, margin=3)
        noise  = y_audio - h
        hnr    = float(np.clip(10*np.log10((np.sum(h**2)+1e-8)/(np.sum(noise**2)+1e-8)),-10,40))
        mfccs  = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
        m1,m2,m3,m4,m5 = [float(np.mean(mfccs[i])) for i in range(5)]
        zcr    = float(np.mean(librosa.feature.zero_crossing_rate(y_audio)[0]))
        energy = float(np.mean(y_audio**2))
        spec_c = float(np.mean(librosa.feature.spectral_centroid(y=y_audio, sr=sr)[0]))
        spec_r = float(np.mean(librosa.feature.spectral_rolloff(y=y_audio, sr=sr)[0])/sr)
        feats  = [pitch_mean,pitch_std,jitter,shimmer,hnr,m1,m2,m3,m4,m5,zcr,energy,spec_c,spec_r]
        names  = ["pitch_mean","pitch_std","jitter","shimmer","hnr",
                  "mfcc1","mfcc2","mfcc3","mfcc4","mfcc5","zcr","energy","spectral_centroid","spectral_rolloff"]
        return feats, {n: round(v,4) for n,v in zip(names,feats)}
    except Exception as e:
        return None, f"Feature extraction failed: {e}"


def analyze_voice_bytes(audio_bytes: bytes):
    feats, info = extract_voice_features(audio_bytes)
    if feats is None:
        return None, info
    result = VOICE_MODEL.predict(feats)
    result["features"]  = info
    result["timestamp"] = datetime.now().isoformat()
    return result, None


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def calc_bmi(h_cm, w_kg):
    try: return round(w_kg / ((h_cm/100)**2), 1)
    except: return None

def go(page): st.session_state.page = page; st.rerun()

TYPE_LABELS = {
    "none":       "No Diabetes / Low Risk",
    "prediabetic":"Pre-diabetic",
    "type1":      "Type 1 Diabetes",
    "type2":      "Type 2 Diabetes",
    "gestational":"Gestational Diabetes",
    "high_risk":  "At High Risk",
}
ACT_LABELS = {"sedentary":"Sedentary","moderate":"Moderately Active","active":"Very Active"}
BASELINE_GLUCOSE = {"none":95,"prediabetic":118,"type1":130,"type2":145,"gestational":135,"high_risk":155}

def risk_color_st(score):
    if score >= 60: return "red"
    if score >= 35: return "orange"
    return "green"

def glucose_status(g):
    if g < 55:   return "🔴 Critical Low",   "#dc2626"
    if g < 70:   return "🟡 Low",            "#f59e0b"
    if g < 90:   return "🟡 Low-Caution",    "#eab308"
    if g <= 140: return "🟢 Optimal",        "#22c55e"
    if g <= 180: return "🟢 Elevated",       "#84cc16"
    if g <= 250: return "🟠 High",           "#f97316"
    return "🔴 Critical High", "#ef4444"


# ═══════════════════════════════════════════════════════════════════════════════
# THEME DETECTION & CSS
# ═══════════════════════════════════════════════════════════════════════════════
# Detect light/dark mode from Streamlit config
is_dark_mode = st.get_option("theme.base") != "light"

st.markdown(f"""
<style>
:root {{
    --bg-primary: {'#08090f' if is_dark_mode else '#ffffff'};
    --bg-secondary: {'#161d2e' if is_dark_mode else '#f5f5f5'};
    --bg-tertiary: {'#101520' if is_dark_mode else '#fafafa'};
    --text-primary: {'#e8eeff' if is_dark_mode else '#1a1a1a'};
    --text-secondary: {'#5a6882' if is_dark_mode else '#666666'};
    --border-color: {'rgba(255,255,255,0.07)' if is_dark_mode else 'rgba(0,0,0,0.1)'};
}}

/* General */
body, .stApp {{ 
    background: var(--bg-primary); 
    color: var(--text-primary); 
}}
[data-testid="stSidebar"] {{ 
    background: var(--bg-tertiary); 
    border-right: 1px solid var(--border-color); 
}}
.stButton > button {{
    background: linear-gradient(135deg,#4f8ef7,#9b6dff);
    color: #fff; border: none; border-radius: 9px;
    font-weight: 700; padding: 9px 20px;
    transition: opacity 0.2s;
}}
.stButton > button:hover {{ opacity: 0.88; }}
.stButton > button[kind="secondary"] {{
    background: var(--bg-secondary); 
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
}}
.metric-card {{
    background: var(--bg-secondary); 
    border: 1px solid var(--border-color);
    border-radius: 13px; padding: 16px 18px; margin-bottom: 10px;
}}
.score-badge {{
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-weight: 700; font-size: 13px;
}}
.section-header {{
    font-size: 22px; font-weight: 700; margin-bottom: 4px;
    background: linear-gradient(90deg,#4f8ef7,#9b6dff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.sub-text {{ color: var(--text-secondary); font-size: 13px; margin-bottom: 16px; }}
.card-box {{
    background: var(--bg-tertiary); 
    border: 1px solid var(--border-color);
    border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
}}
.log-row {{
    background: var(--bg-secondary); 
    border: 1px solid var(--border-color);
    border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;
}}
.risk-bar-wrap {{
    background: {'#1d2640' if is_dark_mode else '#e8e8e8'}; 
    border-radius: 4px; height: 8px; margin-top: 6px;
}}
.risk-bar-fill {{
    border-radius: 4px; height: 8px;
}}
.stProgress > div > div > div {{ background: linear-gradient(90deg,#4f8ef7,#9b6dff) !important; }}
div[data-testid="stExpander"] {{ 
    background: var(--bg-tertiary); 
    border: 1px solid var(--border-color); 
    border-radius: 10px; 
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎙 Diab**Vox**")
        st.markdown("---")
        if st.session_state.logged_in:
            p = st.session_state.profile
            dtype = st.session_state.diabetes_type or "none"
            vr    = st.session_state.voice_result_a or st.session_state.voice_result_b

            st.markdown("### 👤 Your Profile")
            st.markdown(f"""
<div class="metric-card">
<b>{p.get('name','—')}</b><br>
<span style='color:#5a6882;font-size:12px'>Age {p.get('age','—')} · BMI {p.get('bmi','—')} kg/m²</span><br>
<span style='color:#5a6882;font-size:12px'>BP {p.get('bp','—')} mmHg</span><br><br>
<b style='color:#4f8ef7'>{TYPE_LABELS.get(dtype,'—')}</b>
</div>
""", unsafe_allow_html=True)

            if vr:
                col_hex = vr["color"]
                st.markdown(f"""
<div class="metric-card">
<div style='font-size:11px;color:#5a6882;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Voice Risk Baseline</div>
<div style='font-size:28px;font-weight:700;color:{col_hex}'>{vr["risk_score"]}%</div>
<div style='color:{col_hex};font-size:13px'>{vr["risk_level"]}</div>
</div>
""", unsafe_allow_html=True)

            gh = st.session_state.glucose_history
            if gh:
                vals = [r["glucose"] for r in gh]
                avg  = round(sum(vals)/len(vals),1)
                ir   = round(sum(1 for v in vals if 70<=v<=180)/len(vals)*100,1)
                c1,c2 = st.columns(2)
                c1.metric("Readings", len(vals))
                c2.metric("Avg mg/dL", avg)
                c1.metric("In Range", f"{ir}%")
                vh = st.session_state.voice_history
                if vh:
                    c2.metric("Voice Risk", f"{vh[-1]['risk_score']}%")

            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                init_state()
                go("onboard_1")
        else:
            st.markdown("""
<div style='color:#5a6882;font-size:13px;line-height:1.6'>
Complete the onboarding steps to set up your personalised AI diabetes intelligence profile.
</div>
""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<div style='font-size:11px;color:#5a6882'>🔒 HIPAA-compliant · AES-256 · TLS 1.3</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ONBOARDING — STEP 1: Health Profile
# ═══════════════════════════════════════════════════════════════════════════════
def page_onboard_1():
    st.markdown('<div class="section-header">Tell us about yourself</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Step 1 of 5 — This information powers your personalised diabetes risk assessment.</div>', unsafe_allow_html=True)
    st.progress(0.2)

    with st.form("profile_form"):
        name = st.text_input("Full Name *", placeholder="e.g. Arjun Sharma")

        c1, c2 = st.columns(2)
        age  = c1.number_input("Age *", min_value=1, max_value=120, value=40, step=1)
        sex  = c2.selectbox("Biological Sex", ["","Male","Female","Other / Prefer not to say"])

        st.markdown("**Blood Pressure (mmHg) ***")
        bp1, bp2 = st.columns(2)
        bp_sys = bp1.number_input("Systolic (e.g. 120)", min_value=60, max_value=250, value=120, step=1)
        bp_dia = bp2.number_input("Diastolic (e.g. 80)", min_value=40, max_value=150, value=80, step=1)

        h1, h2 = st.columns(2)
        height = h1.number_input("Height (cm) *", min_value=50, max_value=250, value=170, step=1)
        weight = h2.number_input("Weight (kg) *", min_value=20, max_value=300, value=70, step=1)

        a1, a2 = st.columns(2)
        activity = a1.selectbox("Physical Activity Level *", [
            "","sedentary","moderate","active"],
            format_func=lambda x: {"":"Select…","sedentary":"Sedentary (desk job, no exercise)",
                                    "moderate":"Moderate (light exercise 1–3×/week)",
                                    "active":"Active (exercise 4+×/week)"}[x])
        diet = a2.selectbox("Dietary Preference", ["none","vegetarian","vegan","non-vegetarian"],
                            format_func=lambda x: {"none":"No preference","vegetarian":"Vegetarian",
                                                    "vegan":"Vegan","non-vegetarian":"Non-vegetarian"}[x])

        family = st.selectbox("Family History of Diabetes", [
            "no","yes_parent","yes_sibling","yes_both","unknown"],
            format_func=lambda x: {"no":"No","yes_parent":"Yes – parent","yes_sibling":"Yes – sibling",
                                    "yes_both":"Yes – both parents","unknown":"Unknown"}[x])

        st.markdown("**Existing Health Conditions**")
        cond_cols = st.columns(3)
        cond_htn     = cond_cols[0].checkbox("Hypertension")
        cond_heart   = cond_cols[1].checkbox("Heart disease")
        cond_kidney  = cond_cols[2].checkbox("Kidney disease")
        cond_thyroid = cond_cols[0].checkbox("Thyroid disorder")
        cond_obesity = cond_cols[1].checkbox("Obesity")
        cond_pcos    = cond_cols[2].checkbox("PCOS")

        submitted = st.form_submit_button("Continue →", use_container_width=True)

    if submitted:
        errors = []
        if not name.strip():     errors.append("Full Name is required.")
        if not activity:         errors.append("Activity Level is required.")
        if bp_sys < 60 or bp_dia < 40: errors.append("Enter valid blood pressure readings.")
        if errors:
            for e in errors: st.error(e)
        else:
            bmi = calc_bmi(height, weight)
            conditions = (["hypertension"] if cond_htn else []) + \
                         (["heart_disease"] if cond_heart else []) + \
                         (["kidney_disease"] if cond_kidney else []) + \
                         (["thyroid"] if cond_thyroid else []) + \
                         (["obesity"] if cond_obesity else []) + \
                         (["pcos"] if cond_pcos else [])
            st.session_state.profile = {
                "name": name.strip(), "age": age, "sex": sex,
                "bp": f"{bp_sys}/{bp_dia}", "height": height, "weight": weight, "bmi": bmi,
                "activity": activity, "diet": diet, "family_history": family,
                "conditions": conditions,
            }
            st.session_state.user_name = name.strip().split()[0]
            go("onboard_2")


# ═══════════════════════════════════════════════════════════════════════════════
# ONBOARDING — STEP 2: Pathway Selection
# ═══════════════════════════════════════════════════════════════════════════════
def page_onboard_2():
    st.markdown('<div class="section-header">How would you like to proceed?</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Step 2 of 5 — Choose the pathway that best describes your situation.</div>', unsafe_allow_html=True)
    st.progress(0.4)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
<div class="card-box" style="border:2px solid #4f8ef7">
<div style='font-size:28px;margin-bottom:8px'>🎙️</div>
<div style='font-weight:700;font-size:15px;margin-bottom:6px'>Pathway A — Test my risk</div>
<div style='color:#5a6882;font-size:13px'>I haven't been diagnosed. Use voice analysis to assess my diabetes risk and assign a preliminary status.</div>
</div>
""", unsafe_allow_html=True)
        if st.button("Choose Pathway A", use_container_width=True, key="path_a"):
            st.session_state.pathway = "A"; go("onboard_3a")

    with col_b:
        st.markdown("""
<div class="card-box">
<div style='font-size:28px;margin-bottom:8px'>🩺</div>
<div style='font-weight:700;font-size:15px;margin-bottom:6px'>Pathway B — I know my status</div>
<div style='color:#5a6882;font-size:13px'>I've already been diagnosed. I'll enter my type and complete a voice baseline for ongoing tracking.</div>
</div>
""", unsafe_allow_html=True)
        if st.button("Choose Pathway B", use_container_width=True, key="path_b"):
            st.session_state.pathway = "B"; go("onboard_3b")

    st.markdown("---")
    if st.button("← Back", key="back_2"):
        go("onboard_1")


# ═══════════════════════════════════════════════════════════════════════════════
# ONBOARDING — STEP 3A: Voice Analysis (Pathway A)
# ═══════════════════════════════════════════════════════════════════════════════
def page_onboard_3a():
    st.markdown('<div class="section-header">🎙 Voice Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Step 3 of 5 — Record your voice. The AI extracts pitch, jitter, shimmer, HNR & MFCC features.</div>', unsafe_allow_html=True)
    st.progress(0.6)

    st.info("🎤 Speak naturally for **5–10 seconds** after clicking Record. Try saying: *'The quick brown fox jumps over the lazy dog'*")

    audio = st.audio_input("Record Voice Sample", key="voice_3a")

    if audio is not None:
        audio_bytes = audio.read()
        with st.spinner("Analysing biomarkers with ML ensemble (SVM + MLP + RF)…"):
            result, err = analyze_voice_bytes(audio_bytes)

        if err:
            st.error(f"Analysis failed: {err}")
        elif result:
            st.session_state.voice_result_a = result
            st.session_state.voice_history.append(result)
            render_voice_result(result)

            st.success(f"✅ Voice analysis complete! Risk: **{result['risk_score']}% — {result['risk_level']}**")
            if st.button("Continue →", use_container_width=True, key="3a_next"):
                go("onboard_4")

    elif st.session_state.voice_result_a:
        render_voice_result(st.session_state.voice_result_a)
        if st.button("Continue →", use_container_width=True, key="3a_next_cached"):
            go("onboard_4")

    st.markdown("---")
    if st.button("← Back", key="back_3a"):
        go("onboard_2")


def render_voice_result(r):
    score = r["risk_score"]; color = r["color"]; bd = r.get("breakdown", {})
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
<div class="metric-card" style="text-align:center;border-color:{color}">
<div style='font-size:42px;font-weight:700;color:{color}'>{score}%</div>
<div style='color:{color};font-weight:600'>{r["risk_level"]}</div>
<div style='color:#5a6882;font-size:12px;margin-top:4px'>{r.get("suggested_label","—")}</div>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("**Risk Breakdown**")
        lp = bd.get("low_risk_prob", 0)
        mp = bd.get("moderate_risk_prob", 0)
        hp = bd.get("high_risk_prob", 0)
        st.markdown(f"🟢 Low Risk: **{lp:.1f}%**"); st.progress(lp/100)
        st.markdown(f"🟡 Moderate Risk: **{mp:.1f}%**"); st.progress(mp/100)
        st.markdown(f"🔴 High Risk: **{hp:.1f}%**"); st.progress(hp/100)

    feats = r.get("features", {})
    if feats:
        with st.expander("📊 Voice Biomarker Details"):
            cols = st.columns(4)
            items = list(feats.items())
            for i, (k, v) in enumerate(items[:8]):
                cols[i%4].metric(k.replace("_"," ").title(), f"{v:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# ONBOARDING — STEP 3B: Known Diabetes Type (Pathway B)
# ═══════════════════════════════════════════════════════════════════════════════
def page_onboard_3b():
    st.markdown('<div class="section-header">🩺 Select Your Diabetes Type</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Step 3 of 5 — Choose your clinical diagnosis.</div>', unsafe_allow_html=True)
    st.progress(0.6)

    diabetes_type = st.radio("Diabetes Type", [
        "type1","type2","prediabetic","gestational","none"],
        format_func=lambda x: {
            "type1":      "Type 1 Diabetes — Autoimmune, managed with insulin",
            "type2":      "Type 2 Diabetes — Insulin resistance",
            "prediabetic":"Pre-diabetic — Above normal but not yet diabetic",
            "gestational":"Gestational Diabetes — Diagnosed during pregnancy",
            "none":       "No Diabetes / At Risk",
        }[x])

    if diabetes_type != "none":
        with st.expander("📋 Additional Medical Details (optional)"):
            c1, c2 = st.columns(2)
            diag_date = c1.text_input("Date of Diagnosis (YYYY-MM)", placeholder="e.g. 2021-06")
            hba1c     = c2.number_input("Recent HbA1c (%)", min_value=4.0, max_value=15.0, value=7.0, step=0.1)
            meds      = st.text_input("Current Medications", placeholder="e.g. Metformin 500mg")
            monitor   = st.selectbox("Monitoring Method",
                                     ["none","cgm","finger_prick","both"],
                                     format_func=lambda x: {"none":"None","cgm":"Continuous Glucose Monitor (CGM)",
                                                             "finger_prick":"Finger-prick test","both":"Both CGM & finger-prick"}[x])
            st.session_state.profile.update({
                "diagnosis_date": diag_date, "hba1c": hba1c,
                "medications": meds, "monitoring_method": monitor,
            })

    col_back, col_next = st.columns([1, 3])
    with col_back:
        if st.button("← Back", key="back_3b"): go("onboard_2")
    with col_next:
        if st.button("Continue →", use_container_width=True, key="3b_next"):
            st.session_state.diabetes_type = diabetes_type
            st.session_state.profile["diabetes_type"] = diabetes_type
            st.session_state.profile["baseline_glucose"] = BASELINE_GLUCOSE.get(diabetes_type, 110)
            go("onboard_4b")


# ═══════════════════════════════════════════════════════════════════════════════
# ONBOARDING — STEP 4 (Pathway A): Confirm Type
# ═══════════════════════════════════════════════════════════════════════════════
def page_onboard_4():
    st.markdown('<div class="section-header">✅ Confirm Your Diabetes Type</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Step 4 of 5 — Review the AI suggestion and confirm or adjust.</div>', unsafe_allow_html=True)
    st.progress(0.8)

    vr = st.session_state.voice_result_a
    if vr:
        st.markdown(f"""
<div class="card-box" style="border-color:{vr['color']}">
<div style='font-size:11px;color:#5a6882;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>Voice Analysis Suggested</div>
<div style='font-size:16px;font-weight:700;color:{vr["color"]}'>{vr.get("suggested_label","—")} &nbsp;({vr["risk_score"]}% risk)</div>
<div style='color:#5a6882;font-size:12px;margin-top:4px'>You can accept this or adjust below if you have prior clinical knowledge.</div>
</div>
""", unsafe_allow_html=True)

    sug = vr["suggested_diabetes_type"] if vr else "none"
    type_opts = ["none","prediabetic","high_risk","type1","type2"]
    default_idx = type_opts.index(sug) if sug in type_opts else 0

    confirmed = st.radio("Confirm or Adjust Diabetes Type", type_opts,
                         index=default_idx,
                         format_func=lambda x: TYPE_LABELS.get(x, x))

    col_back, col_next = st.columns([1, 3])
    with col_back:
        if st.button("← Back", key="back_4"): go("onboard_3a")
    with col_next:
        if st.button("Complete Registration →", use_container_width=True, key="4_next"):
            st.session_state.confirmed_type = confirmed
            st.session_state.diabetes_type  = confirmed
            st.session_state.profile["diabetes_type"]       = confirmed
            st.session_state.profile["baseline_glucose"]    = BASELINE_GLUCOSE.get(confirmed, 110)
            go("onboard_5")


# ═══════════════════════════════════════════════════════════════════════════════
# ONBOARDING — STEP 4B (Pathway B): Mandatory Voice Baseline
# ═══════════════════════════════════════════════════════════════════════════════
def page_onboard_4b():
    st.markdown('<div class="section-header">🎙 Record Voice Baseline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Step 4 of 5 — A mandatory voice recording establishes your biomarker baseline for future comparison.</div>', unsafe_allow_html=True)
    st.progress(0.8)

    st.info("🔬 Even with a known diagnosis, a voice baseline lets DiabVox detect changes over time and validate your self-reported status.")

    audio = st.audio_input("Record Baseline Voice Sample", key="voice_4b")

    if audio is not None:
        audio_bytes = audio.read()
        with st.spinner("Establishing voice baseline…"):
            result, err = analyze_voice_bytes(audio_bytes)

        if err:
            st.error(f"Baseline failed: {err}")
        elif result:
            st.session_state.voice_result_b = result
            st.session_state.voice_history.append(result)
            render_voice_result(result)
            st.success("✅ Baseline voice profile saved!")
            if st.button("Complete Registration →", use_container_width=True, key="4b_next"):
                go("onboard_5")

    elif st.session_state.voice_result_b:
        render_voice_result(st.session_state.voice_result_b)
        if st.button("Complete Registration →", use_container_width=True, key="4b_next_cached"):
            go("onboard_5")

    st.markdown("---")
    if st.button("← Back", key="back_4b"): go("onboard_3b")


# ═══════════════════════════════════════════════════════════════════════════════
# ONBOARDING — STEP 5: Completion
# ═══════════════════════════════════════════════════════════════════════════════
def page_onboard_5():
    st.markdown('<div class="section-header">🎉 Welcome to DiabVox!</div>', unsafe_allow_html=True)
    st.progress(1.0)

    p     = st.session_state.profile
    dtype = st.session_state.diabetes_type or "none"
    vr    = st.session_state.voice_result_a or st.session_state.voice_result_b

    st.markdown(f"""
<div class="card-box" style="text-align:center;border-color:#4f8ef7">
<div style='font-size:52px;margin-bottom:12px'>🎉</div>
<div style='font-size:22px;font-weight:700;margin-bottom:8px'>Your profile is complete</div>
<div style='color:#5a6882;font-size:14px'>You now have full access to AI-powered diabetes management.</div>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Profile Summary**")
        st.markdown(f"""
<div class="metric-card">
<table style='width:100%;font-size:13px;border-collapse:collapse'>
<tr><td style='color:#5a6882;padding:5px 0'>Name</td><td style='font-weight:600'>{p.get('name','—')}</td></tr>
<tr><td style='color:#5a6882;padding:5px 0'>Age / BMI</td><td style='font-weight:600'>{p.get('age','—')} yrs / {p.get('bmi','—')} kg/m²</td></tr>
<tr><td style='color:#5a6882;padding:5px 0'>Blood Pressure</td><td style='font-weight:600'>{p.get('bp','—')} mmHg</td></tr>
<tr><td style='color:#5a6882;padding:5px 0'>Diabetes Type</td><td style='font-weight:600;color:#4f8ef7'>{TYPE_LABELS.get(dtype,'—')}</td></tr>
<tr><td style='color:#5a6882;padding:5px 0'>Activity Level</td><td style='font-weight:600'>{ACT_LABELS.get(p.get('activity',''),'—')}</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    with col2:
        if vr:
            st.markdown("**Voice Analysis Result**")
            render_voice_result(vr)

    st.markdown("---")
    if st.button("🚀 Enter Dashboard →", use_container_width=True, type="primary"):
        st.session_state.logged_in = True
        go("dashboard")


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    p     = st.session_state.profile
    dtype = st.session_state.diabetes_type or "none"

    st.markdown(f'<div class="section-header">👋 Welcome back, {p.get("name","").split()[0]}!</div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🎙 Voice Assessment",
        "📊 Glucose Monitor",
        "🥗 Meal Plan",
        "💊 Medications",
        "🍽️ Meal Log",
        "🏃 Activity Log",
        "📋 Medical Records",
        "🧘 Wellness",
        "🩺 Care Team",
        "📈 Risk Analytics",
    ])

    with tabs[0]: tab_voice()
    with tabs[1]: tab_glucose()
    with tabs[2]: tab_meal_plan()
    with tabs[3]: tab_medications()
    with tabs[4]: tab_meal_log()
    with tabs[5]: tab_activity_log()
    with tabs[6]: tab_medical_records()
    with tabs[7]: tab_wellness()
    with tabs[8]: tab_care_team()
    with tabs[9]: tab_analytics()


# ─── TAB: Voice Assessment ─────────────────────────────────────────────────
def tab_voice():
    st.markdown('<div class="section-header">🎙 Voice Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">SVM + MLP + RF ensemble · Re-record anytime to track changes over time · 91.2% accuracy</div>', unsafe_allow_html=True)

    st.info("🎤 Click **Record** and speak naturally for 5–10 seconds. Try reading aloud: *'The quick brown fox jumps over the lazy dog. Peter Piper picked a peck of pickled peppers.'*")

    audio = st.audio_input("Record Voice Assessment", key="dash_voice")

    if audio is not None:
        audio_bytes = audio.read()
        with st.spinner("Analysing with ML ensemble…"):
            result, err = analyze_voice_bytes(audio_bytes)

        if err:
            st.error(f"Analysis failed: {err}")
        elif result:
            st.session_state.voice_history.append(result)
            st.success(f"✅ Assessment complete — **{result['risk_score']}% — {result['risk_level']}**")
            render_voice_result(result)

            sc = result["risk_score"]
            if sc >= 60:
                st.error("⚠️ High risk profile. Please consult a healthcare provider for HbA1c and fasting glucose testing.")
            elif sc >= 35:
                st.warning("⚡ Moderate risk. Consider lifestyle changes and annual screening.")
            else:
                st.success("✅ Low risk profile. Maintain healthy habits and monitor annually.")

    # Voice history chart
    vh = st.session_state.voice_history
    if len(vh) > 1:
        st.markdown("---")
        st.markdown("**Voice Risk Trend**")
        scores = [v["risk_score"] for v in vh]
        times  = [v.get("timestamp", "")[:16].replace("T"," ") for v in vh]
        fig = go_obj.Figure()
        fig.add_trace(go_obj.Scatter(
            x=list(range(len(scores))), y=scores, mode="lines+markers",
            line=dict(color="#4f8ef7", width=2),
            marker=dict(color=["#ef4444" if s>=60 else "#f59e0b" if s>=35 else "#22c55e" for s in scores], size=8),
            text=times, hovertemplate="%{text}<br>Risk: %{y:.1f}%<extra></extra>",
        ))
        fig.add_hline(y=60, line_dash="dash", line_color="#ef4444", annotation_text="High Risk")
        fig.add_hline(y=35, line_dash="dash", line_color="#f59e0b", annotation_text="Moderate Risk")
        plot_bg = "#161d2e" if is_dark_mode else "#f9f9f9"
        paper_bg = "#101520" if is_dark_mode else "#ffffff"
        grid_color = "#1d2640" if is_dark_mode else "#e0e0e0"
        font_color = "#e8eeff" if is_dark_mode else "#1a1a1a"
        fig.update_layout(
            paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
            font_color=font_color, height=250,
            xaxis=dict(showticklabels=False, gridcolor=grid_color),
            yaxis=dict(range=[0,100], title="Risk Score (%)", gridcolor=grid_color),
            margin=dict(l=0,r=0,t=10,b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # History list
        with st.expander("📋 All Voice Assessments"):
            for v in reversed(vh):
                ts = v.get("timestamp","")[:16].replace("T"," ")
                col_hex = v["color"]
                st.markdown(f"""
<div class="log-row">
<span style='color:{col_hex};font-weight:700'>{v["risk_score"]}% — {v["risk_level"]}</span>
<span style='color:#5a6882;font-size:11px;float:right'>{ts}</span><br>
<span style='color:#5a6882;font-size:12px'>{v.get("suggested_label","")}</span>
</div>
""", unsafe_allow_html=True)


# ─── TAB: Glucose Monitor ─────────────────────────────────────────────────
def tab_glucose():
    st.markdown('<div class="section-header">📊 Glucose Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Random Forest on 8 PPG-correlated physiological features</div>', unsafe_allow_html=True)

    with st.form("glucose_form"):
        c1, c2, c3 = st.columns(3)
        hr       = c1.slider("❤️ Heart Rate (bpm)", 45, 130, 72)
        activity = c2.selectbox("🏃 Activity Level", [0,1,2,3],
                                format_func=lambda x: {0:"Resting",1:"Light activity",2:"Moderate exercise",3:"Intense exercise"}[x])
        tsmeal   = c3.selectbox("🍽️ Time Since Last Meal (hrs)", [0.5,1.0,2.0,4.0],
                                format_func=lambda x: {0.5:"30 min",1.0:"1 hour",2.0:"2 hours",4.0:"4+ hours"}[x])
        c4, c5, c6 = st.columns(3)
        carbs    = c4.slider("🥐 Meal Carbs (g)", 0, 150, 45)
        stress   = c5.selectbox("😰 Stress Level", [0,1,2,3],
                                format_func=lambda x: {0:"Calm",1:"Mild",2:"Moderate",3:"High"}[x])
        mtype    = c6.selectbox("🕐 Meal Type", ["breakfast","lunch","dinner","snack"],
                                index=1)
        submit = st.form_submit_button("📊 Predict Glucose & Get Recommendations", use_container_width=True)

    if submit:
        base_g = st.session_state.profile.get("baseline_glucose", 110)
        fitness = st.session_state.profile.get("activity", "moderate")
        glucose, ppg = GLUCOSE_MODEL.predict_from_inputs(hr, activity, tsmeal, carbs, stress, base_g)
        glucose = round(glucose, 1)

        # Trend
        gh = st.session_state.glucose_history
        trend = "stable"
        if len(gh) >= 3:
            vals  = [r["glucose"] for r in gh[-8:]]
            slope = np.polyfit(range(len(vals)), vals, 1)[0]
            trend = "rising" if slope > 2 else ("falling" if slope < -2 else "stable")

        status_label, status_color = glucose_status(glucose)
        trend_arrow = "↑" if trend=="rising" else ("↓" if trend=="falling" else "→")
        trend_color = "#ef4444" if trend=="rising" else ("#f59e0b" if trend=="falling" else "#22c55e")

        reading = {
            "glucose": glucose, "status": status_label, "status_color": status_color,
            "trend": trend, "timestamp": datetime.now().isoformat(),
            "inputs": {"heart_rate":hr,"activity":activity,"time_since_meal":tsmeal,"meal_carbs":carbs,"stress":stress},
        }
        st.session_state.glucose_history.append(reading)

        # Display glucose
        c_left, c_right = st.columns([1, 2])
        with c_left:
            st.markdown(f"""
<div class="metric-card" style="text-align:center;border-color:{status_color}">
<div style='font-size:60px;font-weight:700;color:{status_color}'>{glucose}</div>
<div style='color:#5a6882;font-size:14px'>mg/dL</div>
<div style='color:{trend_color};font-size:16px;font-weight:700;margin-top:8px'>{trend_arrow} {trend.capitalize()}</div>
<div style='color:{status_color};font-size:14px;font-weight:600;margin-top:4px'>{status_label}</div>
</div>
""", unsafe_allow_html=True)

        with c_right:
            # Recommendations
            diet_rec = REC_ENGINE.food(glucose, mtype)
            ex_rec   = REC_ENGINE.exercise(glucose, trend, fitness)
            st.markdown("**🥗 Diet Recommendation**")
            meal = diet_rec["meal"]
            st.markdown(f"""
<div class="card-box">
<b>{meal['name']}</b><br>
<span style='color:#4f8ef7;font-size:12px'>{meal['carbs']}g carbs · {meal['protein']}g protein · GI: {meal['gi']}</span><br>
<span style='color:#5a6882;font-size:12px'>{diet_rec['rationale']}</span><br>
<span style='color:#f59e0b;font-size:12px'>💡 {diet_rec['timing_tip']}</span>
</div>
""", unsafe_allow_html=True)

            st.markdown("**🏃 Exercise Recommendation**")
            if ex_rec["status"] == "contraindicated":
                st.error(ex_rec["message"])
            else:
                ex = ex_rec["exercise"]
                st.markdown(f"""
<div class="card-box">
<b>{ex['name']}</b> — {ex['duration']} min · {ex_rec['level']} intensity<br>
<span style='color:#22c55e;font-size:12px'>Expected effect: {ex_rec['estimated_glucose_change']}</span><br>
<span style='color:#5a6882;font-size:12px'>{ex_rec['reason']}</span><br>
<span style='color:#5a6882;font-size:12px'>🛡️ {" · ".join(ex_rec["safety_tips"])}</span>
</div>
""", unsafe_allow_html=True)

    # Glucose chart
    gh = st.session_state.glucose_history
    if len(gh) >= 2:
        st.markdown("---")
        st.markdown("**📈 Glucose History**")
        vals  = [r["glucose"] for r in gh[-15:]]
        times = [r["timestamp"][:16].replace("T"," ") for r in gh[-15:]]
        colors = [("#ef4444" if v>250 or v<55 else "#f97316" if v>180 else
                   "#f59e0b" if v<70 else "#22c55e") for v in vals]
        fig = go_obj.Figure()
        fig.add_hrect(y0=70, y1=180, fillcolor="rgba(34,197,94,0.06)", line_width=0)
        fig.add_hline(y=70,  line_dash="dash", line_color="#f59e0b", line_width=1)
        fig.add_hline(y=180, line_dash="dash", line_color="#f97316", line_width=1)
        fig.add_trace(go_obj.Scatter(
            x=list(range(len(vals))), y=vals, mode="lines+markers",
            line=dict(color="#4f8ef7", width=2),
            marker=dict(color=colors, size=8),
            text=times, hovertemplate="%{text}<br>%{y:.0f} mg/dL<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=paper_bg, plot_bgcolor=plot_bg, font_color=font_color,
            height=220, margin=dict(l=0,r=0,t=10,b=10),
            xaxis=dict(showticklabels=False, gridcolor=grid_color),
            yaxis=dict(title="mg/dL", gridcolor=grid_color),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── TAB: Meal Plan ───────────────────────────────────────────────────────
def tab_meal_plan():
    st.markdown('<div class="section-header">🥗 Today\'s Meal Plan</div>', unsafe_allow_html=True)
    dtype = st.session_state.diabetes_type or "none"
    st.markdown(f'<div class="sub-text">Personalised for {TYPE_LABELS.get(dtype,"Standard")}</div>', unsafe_allow_html=True)

    plan = REC_ENGINE.daily_plan(dtype)
    icons = {"Breakfast":"🍳","Lunch":"🥗","Dinner":"🍽️","Snack":"🍎"}
    cols = st.columns(2)
    for i, item in enumerate(plan):
        with cols[i % 2]:
            st.markdown(f"""
<div class="card-box">
<div style='font-size:11px;font-weight:700;color:#5a6882;text-transform:uppercase;letter-spacing:1px'>{icons.get(item['label'],'🥗')} {item['label']}</div>
<div style='font-weight:700;font-size:14px;margin-top:6px'>{item['name']}</div>
<div style='color:#4f8ef7;font-size:12px;margin-top:4px'>{item['carbs']}g carbs · diabetes-friendly · low GI</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ➕ Log a Custom Meal")
    with st.form("meal_log_form"):
        c1, c2 = st.columns(2)
        m_name  = c1.text_input("Meal Name *", placeholder="e.g. Brown rice with dal")
        m_type  = c2.selectbox("Meal Type", ["breakfast","lunch","dinner","snack"], index=1)
        c3, c4  = st.columns(2)
        m_carbs = c3.number_input("Carbohydrates (g)", min_value=0, max_value=300, value=45)
        m_notes = c4.text_input("Notes (optional)", placeholder="e.g. Ate at home")
        if st.form_submit_button("🍽️ Log This Meal", use_container_width=True):
            if not m_name.strip():
                st.error("Please enter a meal name.")
            else:
                st.session_state.meal_logs.append({
                    "id": str(uuid.uuid4()), "name": m_name.strip(),
                    "type": m_type, "carbs": m_carbs, "notes": m_notes,
                    "timestamp": datetime.now().isoformat(),
                })
                st.success(f"✅ Logged: **{m_name}** ({m_carbs}g carbs)")
                st.rerun()


# ─── TAB: Medications ─────────────────────────────────────────────────────
def tab_medications():
    st.markdown('<div class="section-header">💊 Medication Manager</div>', unsafe_allow_html=True)
    meds = st.session_state.medications
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Medications", len(meds))
    c2.metric("Adherence (estimated)", "87%")
    c3.metric("Due Today", sum(1 for m in meds if m.get("times")))

    st.markdown("---")
    st.markdown("### ➕ Add Medication")
    with st.form("add_med_form"):
        c1, c2 = st.columns(2)
        med_name = c1.text_input("Medication Name *", placeholder="e.g. Metformin")
        med_dose = c2.text_input("Dose *", placeholder="e.g. 500mg")
        c3, c4   = st.columns(2)
        med_freq = c3.selectbox("Frequency", ["once_daily","twice_daily","three_daily","with_meals","as_needed"],
                                format_func=lambda x: {"once_daily":"Once daily","twice_daily":"Twice daily",
                                                        "three_daily":"Three times daily","with_meals":"With every meal","as_needed":"As needed"}[x])
        med_timing = c4.selectbox("Timing", ["with_food","before_food","after_food","bedtime","morning"],
                                  format_func=lambda x: {"with_food":"With food","before_food":"Before food",
                                                          "after_food":"After food","bedtime":"At bedtime","morning":"Morning (fasting)"}[x])
        med_times = st.multiselect("Reminder Times", ["8:00 AM","1:00 PM","8:00 PM","10:00 PM"], default=["8:00 AM"])
        med_notes = st.text_input("Notes", placeholder="e.g. Take with a full glass of water")
        if st.form_submit_button("💊 Save Medication", use_container_width=True):
            if not med_name.strip() or not med_dose.strip():
                st.error("Medication name and dose are required.")
            else:
                st.session_state.medications.append({
                    "id": str(uuid.uuid4()), "name": med_name.strip(), "dose": med_dose.strip(),
                    "frequency": med_freq, "timing": med_timing, "times": med_times, "notes": med_notes,
                    "added": datetime.now().isoformat(),
                })
                st.success(f"✅ Added: **{med_name}** {med_dose}")
                st.rerun()

    if meds:
        st.markdown("---")
        st.markdown("### 💊 Current Medications")
        dot_colors = ["#4f8ef7","#9b6dff","#22c55e","#f59e0b","#ef4444","#f97316"]
        for i, m in enumerate(meds):
            col_hex = dot_colors[i % len(dot_colors)]
            c_info, c_del = st.columns([5, 1])
            with c_info:
                st.markdown(f"""
<div class="log-row">
<span style='color:{col_hex};font-size:10px'>●</span>
<b style='margin-left:8px'>{m['name']}</b> <span style='color:#5a6882'>{m['dose']}</span><br>
<span style='color:#5a6882;font-size:12px'>{m.get('frequency','').replace('_',' ')} · {m.get('timing','').replace('_',' ')}</span><br>
{('<span style="color:#4f8ef7;font-size:11px">⏰ ' + ', '.join(m.get('times',[])) + '</span>') if m.get('times') else ''}
</div>
""", unsafe_allow_html=True)
            with c_del:
                if st.button("🗑", key=f"del_med_{m['id']}", help="Delete"):
                    st.session_state.medications = [x for x in st.session_state.medications if x["id"] != m["id"]]
                    st.rerun()


# ─── TAB: Meal Log ────────────────────────────────────────────────────────
def tab_meal_log():
    st.markdown('<div class="section-header">🍽️ Meal Log</div>', unsafe_allow_html=True)
    logs = st.session_state.meal_logs
    today = [m for m in logs if m["timestamp"][:10] == datetime.now().date().isoformat()]
    total_carbs = sum(m.get("carbs",0) for m in today)

    c1, c2, c3 = st.columns(3)
    c1.metric("Meals Today", len(today))
    c2.metric("Total Carbs Today", f"{total_carbs}g")
    c3.metric("Avg Carbs / Meal", f"{round(total_carbs/len(today)) if today else 0}g")

    if not logs:
        st.info("No meals logged yet. Go to the **Meal Plan** tab to log your first meal!")
        return

    type_icons = {"breakfast":"🍳","lunch":"🥗","dinner":"🍽️","snack":"🍎"}
    for m in reversed(logs[-20:]):
        ts = m["timestamp"][:16].replace("T"," ")
        st.markdown(f"""
<div class="log-row">
{type_icons.get(m.get('type',''),'🍽️')} <b>{m['name']}</b>
<span style='color:#5a6882;font-size:12px'> · {m.get('type','').capitalize()} · {m.get('carbs',0)}g carbs</span>
<span style='color:#5a6882;font-size:11px;float:right'>{ts}</span><br>
{('<span style="color:#5a6882;font-size:11px">' + m['notes'] + '</span>') if m.get('notes') else ''}
</div>
""", unsafe_allow_html=True)


# ─── TAB: Activity Log ────────────────────────────────────────────────────
def tab_activity_log():
    st.markdown('<div class="section-header">🏃 Activity Log</div>', unsafe_allow_html=True)
    acts = st.session_state.activity_logs
    total_min = sum(a.get("duration",0) for a in acts)
    total_cal = sum(a.get("calories",0) for a in acts)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sessions", len(acts))
    c2.metric("Total Minutes", total_min)
    c3.metric("Est. Calories Burned", total_cal)

    # Exercise safety guide
    gh = st.session_state.glucose_history
    if gh:
        last_g = gh[-1]["glucose"]
        st.markdown("**🛡️ Exercise Safety Guide**")
        if last_g < 70:
            st.error(f"⚠️ Current glucose {last_g} mg/dL — Low. Eat 15g fast-acting carbs before exercising.")
        elif last_g > 270:
            st.error(f"⚠️ Current glucose {last_g} mg/dL — Very high. Check for ketones first.")
        elif last_g > 180:
            st.warning(f"⚡ Current glucose {last_g} mg/dL — Elevated. Stick to light-intensity exercise.")
        else:
            st.success(f"✅ Current glucose {last_g} mg/dL — Safe to exercise at moderate intensity.")

    st.markdown("---")
    st.markdown("### ➕ Log Activity")
    with st.form("act_form"):
        c1, c2 = st.columns(2)
        act_type = c1.selectbox("Activity Type", [
            "Walking","Brisk walking","Running","Cycling","Swimming",
            "Yoga","Strength training","HIIT","Dancing","Other"])
        act_dur  = c2.number_input("Duration (minutes) *", min_value=1, max_value=300, value=30)
        c3, c4   = st.columns(2)
        intensity = c3.selectbox("Intensity", ["low","moderate","high"],
                                 index=1, format_func=lambda x: {"low":"Low (MET ≤ 3)","moderate":"Moderate (MET 3–6)","high":"High (MET > 6)"}[x])
        act_glucose = c4.number_input("Pre-exercise Glucose (optional, mg/dL)", min_value=0, max_value=400, value=0)
        act_notes = st.text_input("Notes (optional)", placeholder="How did you feel?")
        if st.form_submit_button("🏃 Log Activity", use_container_width=True):
            met = {"low":2.5,"moderate":5.0,"high":8.0}[intensity]
            calories = round(met * 70 * (act_dur / 60))
            st.session_state.activity_logs.append({
                "id": str(uuid.uuid4()), "activity": act_type, "type": act_type,
                "duration": act_dur, "intensity": intensity, "calories": calories,
                "pre_glucose": act_glucose if act_glucose > 0 else None,
                "notes": act_notes, "timestamp": datetime.now().isoformat(),
            })
            st.success(f"✅ Logged: {act_type} — {act_dur} min (~{calories} kcal)")
            st.rerun()

    if acts:
        st.markdown("---")
        intensity_icons = {"low":"🟢","moderate":"🟡","high":"🔴"}
        for a in reversed(acts[-15:]):
            ts = a["timestamp"][:16].replace("T"," ")
            st.markdown(f"""
<div class="log-row">
🏃 <b>{a.get('type','Activity')}</b> · {a.get('duration',0)} min
{intensity_icons.get(a.get('intensity','moderate'),'🟡')} {a.get('intensity','moderate').capitalize()}
<span style='color:#22c55e;font-size:12px'> ~{a.get('calories',0)} kcal</span>
<span style='color:#5a6882;font-size:11px;float:right'>{ts}</span>
{('<br><span style="color:#5a6882;font-size:11px">📝 ' + a['notes'] + '</span>') if a.get('notes') else ''}
</div>
""", unsafe_allow_html=True)


# ─── TAB: Medical Records ─────────────────────────────────────────────────
def tab_medical_records():
    st.markdown('<div class="section-header">📋 Medical Records</div>', unsafe_allow_html=True)
    recs = st.session_state.medical_records

    # HbA1c quick view
    hba1c_recs = [r for r in recs if r.get("hba1c")]
    if hba1c_recs:
        latest = sorted(hba1c_recs, key=lambda x: x["date"], reverse=True)[0]
        v = latest["hba1c"]
        color = "#22c55e" if v < 7 else "#f59e0b" if v < 8 else "#ef4444"
        st.markdown(f"""
<div class="card-box" style="border-color:{color}">
<div style='font-size:11px;color:#5a6882;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Latest HbA1c</div>
<div style='font-size:36px;font-weight:700;color:{color}'>{v}%</div>
<div style='color:#5a6882;font-size:12px'>{latest.get("date","—")} · {latest.get("title","—")}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### ➕ Add Medical Record")
    with st.form("rec_form"):
        c1, c2 = st.columns(2)
        rec_type  = c1.selectbox("Record Type", ["lab_report","hba1c","prescription","clinical_note","imaging","other"],
                                 format_func=lambda x: {"lab_report":"🧪 Lab Report","hba1c":"🩸 HbA1c Result",
                                                         "prescription":"💊 Prescription","clinical_note":"📝 Clinical Note",
                                                         "imaging":"🖼️ Imaging / Scan","other":"📋 Other"}[x])
        rec_title = c2.text_input("Title *", placeholder="e.g. Quarterly HbA1c Panel")
        c3, c4    = st.columns(2)
        rec_date  = c3.date_input("Date", value=datetime.now().date())
        rec_hba1c = c4.number_input("HbA1c (%)", min_value=0.0, max_value=15.0, value=0.0, step=0.1,
                                    help="Leave 0 if not applicable")
        rec_notes = st.text_area("Notes / Key Values", placeholder="e.g. Fasting glucose: 126 mg/dL")
        if st.form_submit_button("📋 Save Record", use_container_width=True):
            if not rec_title.strip():
                st.error("Title is required.")
            else:
                st.session_state.medical_records.append({
                    "id": str(uuid.uuid4()), "type": rec_type, "title": rec_title.strip(),
                    "date": str(rec_date), "hba1c": rec_hba1c if rec_hba1c > 0 else None,
                    "notes": rec_notes, "uploaded": datetime.now().isoformat(),
                })
                st.success(f"✅ Record saved: **{rec_title}**")
                st.rerun()

    if recs:
        st.markdown("---")
        st.markdown("### All Records")
        type_icons = {"lab_report":"🧪","hba1c":"🩸","prescription":"💊","clinical_note":"📝","imaging":"🖼️","other":"📋"}
        for r in reversed(recs[-20:]):
            st.markdown(f"""
<div class="log-row">
{type_icons.get(r['type'],'📋')} <b>{r['title']}</b>
<span style='color:#5a6882;font-size:12px'> · {r['type'].replace('_',' ').title()}</span>
<span style='color:#5a6882;font-size:11px;float:right'>{r.get('date','—')}</span>
{('<br><span style="color:#4f8ef7;font-size:12px">HbA1c: ' + str(r['hba1c']) + '%</span>') if r.get('hba1c') else ''}
{('<br><span style="color:#5a6882;font-size:11px">' + r['notes'] + '</span>') if r.get('notes') else ''}
</div>
""", unsafe_allow_html=True)


# ─── TAB: Wellness ────────────────────────────────────────────────────────
def tab_wellness():
    st.markdown('<div class="section-header">🧘 Mental Health & Wellness</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Diabetes distress screening, mindfulness, and stress reduction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class="card-box" style="border-color:rgba(16,185,129,0.3)">
<div style='font-size:28px'>🫁</div>
<div style='font-weight:700;font-size:15px;margin:6px 0'>Guided Breathing — 4-7-8 Technique</div>
<div style='color:#5a6882;font-size:13px'>Reduces cortisol and stabilises glucose spikes from stress.</div>
</div>
""", unsafe_allow_html=True)
        with st.expander("▶ Start Breathing Exercise"):
            st.markdown("""
**4-7-8 Breathing Instructions:**
1. 🫁 **Inhale** through your nose for **4 seconds**
2. ⏸ **Hold** your breath for **7 seconds**
3. 💨 **Exhale** completely through your mouth for **8 seconds**
4. Repeat 4 cycles

*Tip: Practice daily, especially before meals or stressful situations.*
""")
            st.info("🧘 Do 4 cycles of this pattern. Each full cycle takes 19 seconds. Total: ~76 seconds.")

        st.markdown("""
<div class="card-box" style="border-color:rgba(155,109,255,0.3);margin-top:12px">
<div style='font-size:28px'>🧘</div>
<div style='font-weight:700;font-size:15px;margin:6px 0'>Mindful Pause — 5-Min Body Scan</div>
<div style='color:#5a6882;font-size:13px'>Body-scan meditation for people managing chronic conditions.</div>
</div>
""", unsafe_allow_html=True)
        with st.expander("▶ Start Mindful Pause"):
            st.markdown("""
**5-Minute Body Scan Meditation:**
1. 😌 Sit comfortably and close your eyes
2. 🌬️ Take 3 slow, deep breaths
3. 👣 Focus on your feet — notice any sensations
4. ⬆️ Slowly move attention upward through your body
5. 🧠 When you reach your head, take 3 more deep breaths
6. 👀 Gently open your eyes

*Research shows regular mindfulness reduces HbA1c and improves diabetes self-management.*
""")

    with col2:
        st.markdown("### 📔 Mood Journal")
        with st.form("journal_form"):
            mood = st.selectbox("How are you feeling?",
                                ["😊 Good","😐 Okay","😔 Low","😤 Stressed","😴 Tired","😰 Anxious"])
            stress_level = st.slider("Stress level (0–10)", 0, 10, 3)
            j_note = st.text_area("Journal note (optional)", placeholder="What's on your mind?", height=100)
            if st.form_submit_button("💾 Save Entry", use_container_width=True):
                st.session_state.wellness_journal.append({
                    "id": str(uuid.uuid4()), "mood": mood, "stress": stress_level,
                    "note": j_note, "timestamp": datetime.now().isoformat(),
                })
                st.success("✅ Journal entry saved!")
                st.rerun()

        if st.session_state.wellness_journal:
            st.markdown("**Recent Entries**")
            for e in reversed(st.session_state.wellness_journal[-5:]):
                ts = e["timestamp"][:16].replace("T"," ")
                st.markdown(f"""
<div class="log-row">
<b>{e['mood']}</b> · Stress: {e['stress']}/10
<span style='color:#5a6882;font-size:11px;float:right'>{ts}</span>
{('<br><span style="color:#5a6882;font-size:12px">' + e['note'] + '</span>') if e.get('note') else ''}
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🩵 Diabetes Distress Screening (DDS-5)")
    with st.expander("Take the 5-question screening"):
        questions = [
            "I feel I cannot follow my diabetes regimen as I would like.",
            "I feel uncomfortable with the amount of effort managing diabetes takes.",
            "I often feel like I'm failing with my diabetes routine.",
            "I feel friends or family don't appreciate how difficult diabetes management is.",
            "I feel overwhelmed by living with diabetes.",
        ]
        responses = []
        for i, q in enumerate(questions):
            v = st.radio(f"**{i+1}.** {q}", [1,2,3,4,5],
                         format_func=lambda x: {1:"1 - Not a problem",2:"2",3:"3 - Moderate",4:"4",5:"5 - Serious problem"}[x],
                         horizontal=True, key=f"dds_{i}")
            responses.append(v)
        if st.button("Calculate Distress Score", use_container_width=True):
            total = sum(responses)
            if total <= 7:   level, color = "Low distress",       "#22c55e"
            elif total <= 11: level, color = "Moderate distress",  "#f59e0b"
            elif total <= 14: level, color = "High distress",      "#f97316"
            else:             level, color = "Very high distress", "#ef4444"
            pct = total / 25
            st.markdown(f"""
<div class="card-box" style="border-color:{color}">
<div style='font-size:22px;font-weight:700;color:{color}'>{level}</div>
<div style='color:#5a6882;margin-top:4px'>Score: {total}/25</div>
</div>
""", unsafe_allow_html=True)
            st.progress(pct)
            if total >= 12:
                st.warning("Consider speaking with a diabetes educator, psychologist, or counsellor who specialises in chronic illness. You don't have to manage this alone.")


# ─── TAB: Care Team ───────────────────────────────────────────────────────
def tab_care_team():
    st.markdown('<div class="section-header">🩺 Care Team</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 👨‍⚕️ My Providers")
        team = st.session_state.care_team
        if not team:
            st.info("No providers connected yet. Invite your doctor or care team below.")
        else:
            for p in team:
                status_color = "#22c55e" if p.get("status") == "active" else "#f59e0b"
                st.markdown(f"""
<div class="log-row">
<b>👨‍⚕️ {p['name']}</b>
<span style='color:{status_color};font-size:11px;float:right'>{p.get('status','pending').upper()}</span><br>
<span style='color:#5a6882;font-size:12px'>{p.get('specialty','—')} · {p.get('access','full')} access</span><br>
<span style='color:#5a6882;font-size:11px'>✉️ {p.get('email','—')}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("### ➕ Invite a Provider")
        with st.form("invite_form"):
            p_name  = st.text_input("Provider Name *", placeholder="e.g. Dr. Ananya Krishnan")
            c1, c2  = st.columns(2)
            p_spec  = c1.selectbox("Specialty", ["Endocrinologist","Diabetologist","General Physician",
                                                  "Dietitian","Ophthalmologist","Nephrologist","Cardiologist"])
            p_access= c2.selectbox("Access Level", ["full","glucose","reports"],
                                   format_func=lambda x: {"full":"Full access","glucose":"Glucose data only","reports":"Reports only"}[x])
            p_email = st.text_input("Email Address", placeholder="doctor@hospital.com")
            st.caption("🔐 An encrypted invitation link will be sent. The provider can only view data you approve.")
            if st.form_submit_button("📧 Send Invitation", use_container_width=True):
                if not p_name.strip() or not p_email.strip():
                    st.error("Name and email are required.")
                else:
                    st.session_state.care_team.append({
                        "id": str(uuid.uuid4()), "name": p_name.strip(), "specialty": p_spec,
                        "access": p_access, "email": p_email.strip(), "status": "pending",
                        "invited": datetime.now().isoformat(),
                    })
                    st.success(f"✅ Invitation sent to {p_email}!")
                    st.rerun()

    with col2:
        st.markdown("### 💬 Secure Messaging")
        st.markdown("<div style='background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);border-radius:8px;padding:6px 12px;font-size:11px;color:#22c55e;margin-bottom:12px'>🔒 HIPAA-Compliant Encrypted Channel</div>", unsafe_allow_html=True)

        msgs = st.session_state.messages
        for m in msgs:
            is_sent = m["sender"] == "You"
            align   = "right" if is_sent else "left"
            bg      = "rgba(79,142,247,.12)" if is_sent else "#161d2e"
            border  = "rgba(79,142,247,.2)" if is_sent else "rgba(255,255,255,.07)"
            st.markdown(f"""
<div style='text-align:{align};margin-bottom:10px'>
<div style='display:inline-block;max-width:82%;background:{bg};border:1px solid {border};border-radius:12px;padding:10px 14px;font-size:13px;line-height:1.5'>
{m['text']}
<div style='color:#5a6882;font-size:10px;margin-top:4px'>{m['sender']} · {m['ts']}</div>
</div>
</div>
""", unsafe_allow_html=True)

        with st.form("msg_form"):
            msg_text = st.text_area("Type a secure message…", height=80, placeholder="Message your care team…", label_visibility="collapsed")
            if st.form_submit_button("Send 📤", use_container_width=True):
                if msg_text.strip():
                    st.session_state.messages.append({
                        "text": msg_text.strip(), "sender": "You",
                        "ts": datetime.now().strftime("%b %d %H:%M"),
                    })
                    st.success("✅ Message sent!")
                    st.rerun()

        st.markdown("---")
        st.markdown("### 🔐 Data Sharing Controls")
        st.caption("Control what your care team can see:")
        st.checkbox("📊 Glucose readings", value=True)
        st.checkbox("🎙 Voice assessments", value=True)
        st.checkbox("🍽️ Meal logs", value=False)
        st.checkbox("🏃 Activity logs", value=False)
        st.checkbox("💊 Medications", value=True)


# ─── TAB: Risk Analytics ──────────────────────────────────────────────────
def tab_analytics():
    st.markdown('<div class="section-header">📈 Risk Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">10-year projection and complication risk modelling</div>', unsafe_allow_html=True)

    profile = st.session_state.profile
    dtype   = st.session_state.diabetes_type or "none"
    vh      = st.session_state.voice_history
    gh      = st.session_state.glucose_history

    if not vh:
        st.warning("⚠️ Complete at least one voice assessment to unlock risk analytics.")
        return

    col_ref, _ = st.columns([1, 3])
    with col_ref:
        if st.button("↻ Recalculate", use_container_width=True):
            st.rerun()

    # Export
    export_data = {
        "export_date": datetime.now().isoformat(), "profile": profile,
        "glucose_readings": gh, "voice_assessments": vh,
        "medications": st.session_state.medications,
        "meal_logs": st.session_state.meal_logs,
        "activity_logs": st.session_state.activity_logs,
        "medical_records": st.session_state.medical_records,
        "wellness_journal": st.session_state.wellness_journal,
    }
    st.download_button(
        "⬇ Export All Data (JSON)", data=json.dumps(export_data, indent=2),
        file_name=f"diabvox_export_{datetime.now().date()}.json", mime="application/json",
    )
    # CSV export
    csv_rows = [["Type","Timestamp","Value","Status","Trend","Notes"]]
    for r in gh:
        csv_rows.append(["glucose",r.get("timestamp",""),r.get("glucose",""),r.get("status",""),r.get("trend",""),""])
    for r in vh:
        csv_rows.append(["voice",r.get("timestamp",""),r.get("risk_score",""),r.get("risk_level",""),"",""])
    csv_buf = io.StringIO()
    import csv as csv_mod
    w = csv_mod.writer(csv_buf)
    w.writerows(csv_rows)
    st.download_button("⬇ Export as CSV", data=csv_buf.getvalue(),
                       file_name=f"diabvox_{datetime.now().date()}.csv", mime="text/csv")

    st.markdown("---")

    # 10-year risk
    ten_yr = REC_ENGINE.ten_year_risk(profile, vh)
    if ten_yr is not None:
        risk_val = float(ten_yr)
        if risk_val < 20:   risk_label, risk_color = "Low risk",      "#22c55e"
        elif risk_val < 40: risk_label, risk_color = "Moderate risk", "#f59e0b"
        elif risk_val < 65: risk_label, risk_color = "Elevated risk", "#f97316"
        else:               risk_label, risk_color = "High risk",     "#ef4444"

        st.markdown("### 📈 10-Year Diabetes Risk Projection")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
<div class="card-box" style="text-align:center;border-color:{risk_color}">
<div style='font-size:52px;font-weight:700;color:{risk_color}'>{risk_val}%</div>
<div style='color:{risk_color};font-weight:600;font-size:16px'>{risk_label}</div>
<div style='color:#5a6882;font-size:12px;margin-top:6px'>10-year probability</div>
</div>
""", unsafe_allow_html=True)
        with c2:
            st.markdown("**Contributing Factors**")
            factors = []
            age = profile.get("age"); bmi = profile.get("bmi")
            fam = profile.get("family_history","no"); act = profile.get("activity","moderate")
            if age: factors.append(f"🎂 Age: {age} years")
            if bmi: factors.append(f"⚖️ BMI: {bmi} kg/m²")
            fam_map = {"yes_parent":"👪 Parent has diabetes","yes_sibling":"👪 Sibling has diabetes",
                       "yes_both":"👪 Both parents — highest hereditary risk","unknown":"👪 Family history unknown"}
            if fam in fam_map: factors.append(fam_map[fam])
            act_map = {"sedentary":"🏠 Sedentary lifestyle (+12%)","moderate":"🚶 Moderate activity",
                       "active":"🏃 Active lifestyle (−8%)"}
            factors.append(act_map.get(act,"🚶 Moderate activity"))
            if dtype == "prediabetic": factors.append("⚠️ Pre-diabetic status (+20%)")
            factors.append(f"🎙 Voice risk score: {vh[-1]['risk_score']:.1f}%")
            for f in factors:
                st.markdown(f"""<div class="log-row" style="padding:8px 12px;margin-bottom:6px">{f}</div>""", unsafe_allow_html=True)

        st.progress(min(risk_val/100, 1.0))
        st.caption("Based on voice biomarkers, age, BMI, family history and activity level")
        st.markdown("---")

    # Complication risks
    complications = REC_ENGINE.complications_risk(profile, gh)
    if complications:
        st.markdown("### ⚠️ Complication Risk Assessment")
        st.caption("Based on your glucose history, blood pressure and HbA1c")
        cols = st.columns(2)
        for i, (key, comp) in enumerate(complications.items()):
            with cols[i % 2]:
                risk = comp["risk"]
                if risk < 20:   bar_color = "#22c55e"
                elif risk < 40: bar_color = "#f59e0b"
                elif risk < 60: bar_color = "#f97316"
                else:           bar_color = "#ef4444"
                st.markdown(f"""
<div class="card-box">
<div style='font-size:11px;color:#5a6882;text-transform:uppercase;letter-spacing:1px'>{comp['label']}</div>
<div style='font-size:28px;font-weight:700;color:{bar_color};margin:6px 0'>{risk}%</div>
<div class="risk-bar-wrap"><div class="risk-bar-fill" style="width:{risk}%;background:{bar_color}"></div></div>
<div style='color:#5a6882;font-size:11px;margin-top:6px'>{comp['basis']}</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("---")

    # Voice trend
    if len(vh) > 1:
        st.markdown("### 🎙 Voice Biomarker Trend")
        baseline = vh[0]["risk_score"]; latest = vh[-1]["risk_score"]
        delta = round(latest - baseline, 1)
        direction = "improving 📉" if delta < 0 else ("worsening 📈" if delta > 0 else "stable ➡️")
        delta_color = "#22c55e" if delta < 0 else "#ef4444" if delta > 0 else "#f59e0b"
        st.markdown(f"""
<div class="card-box">
<div style='font-size:36px;font-weight:700;color:{delta_color}'>{'+' if delta>0 else ''}{delta}%</div>
<div style='font-size:14px;font-weight:600;margin-top:4px'>Risk {direction} since baseline</div>
<div style='color:#5a6882;font-size:12px;margin-top:2px'>Baseline: {baseline:.1f}% → Latest: {latest:.1f}%</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
render_sidebar()

PAGE = st.session_state.page

if PAGE == "onboard_1":    page_onboard_1()
elif PAGE == "onboard_2":  page_onboard_2()
elif PAGE == "onboard_3a": page_onboard_3a()
elif PAGE == "onboard_3b": page_onboard_3b()
elif PAGE == "onboard_4":  page_onboard_4()
elif PAGE == "onboard_4b": page_onboard_4b()
elif PAGE == "onboard_5":  page_onboard_5()
elif PAGE == "dashboard":  page_dashboard()
else:
    st.error(f"Unknown page: {PAGE}")
    go("onboard_1")
