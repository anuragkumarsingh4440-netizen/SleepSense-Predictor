# file: app/streamlit_app.py
# encoding: utf-8
"""
Streamlit UI for SleepSense — Bold dark theme, robust image loading, localized suggestions, yoga media added.
Save/overwrite as app/streamlit_app.py and run:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
from pathlib import Path
import sys
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tempfile

# ensure app folder in path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# import helper functions (predict_helper.py must exist)
from predict_helper import (
    FEATURE_COLS, MODEL_PATH,
    load_model, create_dummy_model,
    predict_sleep_quality, display_prediction
)

# requests for robust image fallback
try:
    import requests
    from requests.exceptions import RequestException
except Exception:
    requests = None
    RequestException = Exception

# ---------- Page config ----------
st.set_page_config(page_title="🌙 SleepSense — Final", layout="wide", initial_sidebar_state="expanded")

# ---------- Colors & CSS ----------
PRIMARY_BTN_BG = "#00D4FF"   # cyan
PRIMARY_BTN_TEXT = "#000000"
SECONDARY_BTN_BG = "#FFB86B" # warm orange
SECONDARY_BTN_TEXT = "#000000"

st.markdown(
    f"""
    <style>
      /* App background and text */
      .stApp {{ background-color: #000000; color: #FFFFFF; }}
      .big-title {{ color: #FFFFFF; font-weight: 1000; font-size: 36px; letter-spacing: 0.6px; }}
      .subtitle {{ color:#BFCBDC; font-weight:900; font-size:15px; margin-bottom:8px; }}
      h1,h2,h3,h4,h5,p,label,span {{ color: #FFFFFF !important; font-weight: 900 !important; }}
      /* Sidebar */
      [data-testid="stSidebar"] {{ background-color:#000000; color:#FFF; border-right:1px solid #111; padding:18px; }}
      /* Primary button */
      div.stButton > button:first-child {{
          background: linear-gradient(90deg, {PRIMARY_BTN_BG}, #00b0e6);
          color: {PRIMARY_BTN_TEXT};
          font-weight: 1000;
          padding: 12px 18px;
          border-radius: 12px;
          border: none;
          box-shadow: 0 6px 18px rgba(0,0,0,0.6);
          font-size: 16px;
        }}
      /* Secondary button */
      .stButton>button[title="secondary"] {{
          background: linear-gradient(90deg, {SECONDARY_BTN_BG}, #ff9a3d);
          color: {SECONDARY_BTN_TEXT};
          font-weight: 1000;
          padding: 10px 14px;
          border-radius: 12px;
          border: none;
          font-size: 14px;
      }}
      /* Input labels bold */
      label, .stSlider label, .stNumberInput label, .stSelectbox label {{
          font-weight: 1000 !important;
          color: #fff !important;
        }}

      /* Result box & tips (big, bold, highlighted) */
      .result-box {{
        background:#0d1117;
        border:2px solid #222;
        padding:20px;
        border-radius:14px;
        color:#E6EEF3;
        font-weight:900;
      }}
      .suggestion-title {{
        font-size:22px;
        font-weight:1100;
        color:#FFD54F;
        margin-bottom:6px;
      }}
      .suggestion-explain {{
        font-size:18px;
        font-weight:1000;
        color:#FFFFFF;
        margin-bottom:12px;
      }}
      .tips {{
        display:block;
        margin-top:8px;
        margin-bottom:8px;
        padding:12px;
        border-radius:10px;
        background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      }}
      .tips li {{
        font-size:17px;
        font-weight: 1000;
        margin-bottom:8px;
        color:#FFFFFF;
      }}
      .tip-pill {{
        display:inline-block;
        padding:6px 10px;
        margin-right:8px;
        margin-bottom:6px;
        border-radius:18px;
        background: rgba(0,212,255,0.10);
        color: #00D4FF;
        font-weight:1000;
      }}
      .yoga-row img {{
        border-radius:10px;
        width:140px;
        height:auto;
        margin-right:10px;
        box-shadow:0 8px 22px rgba(0,0,0,0.6);
      }}
      a {{ color: #00D4FF !important; font-weight:900; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown("<div class='big-title'>🌙 SleepSense — Final</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter lifestyle & environment values → get multilingual advice (big bold text & images).</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    LANGUAGES = [
        "English","Hindi","Tamil","Telugu","Kannada","Malayalam","Marathi","Bengali",
        "Gujarati","Punjabi","Odia","Urdu","Assamese","Nepali","Bhojpuri"
    ]
    language = st.selectbox("Choose language", LANGUAGES, index=0)
    st.markdown("---")
    st.write("Model status")
    model = None
    try:
        model = load_model()
        if model is not None:
            st.success("Model loaded ✓")
        else:
            st.warning("Model not available")
    except Exception as e:
        st.warning("Model load error")
        st.write(str(e))

    if st.button("Create Dummy Model (dev)", key="create_dummy"):
        try:
            model = create_dummy_model(force=True)
            st.success("Dummy model created and saved.")
        except Exception as e:
            st.error("Failed to create dummy model: " + str(e))

    st.markdown("---")
    st.write("Display options")
    show_suggest_cards = st.checkbox("Show suggestion cards & images", value=True)
    show_dev = st.checkbox("Dev: show model summary", value=False)
    st.markdown("")

    # concise dev internals display (no long docs)
    if show_dev and model is not None:
        st.write("Model path:", str(MODEL_PATH))
        try:
            if hasattr(model, "steps"):
                inner = list(model.steps)[-1][1]
                st.write("Inner estimator:", type(inner).__name__)
                st.write("Intercept:", getattr(inner, "intercept_", None))
                coeff = getattr(inner, "coef_", None)
                st.write("Coef length:", None if coeff is None else len(coeff))
            else:
                st.write("Estimator:", type(model).__name__)
        except Exception as e:
            st.write("Error reading model internals:", e)

# ---------- Robust image helper ----------
def show_image_safe(url, caption=None, width=None, timeout=6):
    """
    Try st.image(url) first; if fails, download bytes via requests and display from temp file;
    finally fallback to a clickable link. Ensures something is always shown.
    """
    # try direct st.image
    try:
        if width:
            st.image(url, caption=caption, width=width)
        else:
            st.image(url, caption=caption, use_column_width=True)
        return
    except Exception:
        pass

    # if requests not available, fallback to link
    if requests is None:
        if caption:
            st.markdown(f"**{caption}** — [Open image]({url})")
        else:
            st.markdown(f"[Open image]({url})")
        return

    # try to GET and show from temp file
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img_bytes = resp.content
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img_bytes)
        tmp.flush()
        tmp.close()
        try:
            if width:
                st.image(tmp.name, caption=caption, width=width)
            else:
                st.image(tmp.name, caption=caption, use_column_width=True)
        except Exception:
            if caption:
                st.markdown(f"**{caption}** — [Open image]({url})")
            else:
                st.markdown(f"[Open image]({url})")
        return
    except RequestException:
        if caption:
            st.markdown(f"**{caption}** — [Open image]({url})")
        else:
            st.markdown(f"[Open image]({url})")
        return
    except Exception:
        if caption:
            st.markdown(f"**{caption}** — [Open image]({url})")
        else:
            st.markdown(f"[Open image]({url})")
        return

# ---------- Input form ----------
with st.form("input_form"):
    st.subheader("Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=8, max_value=100, value=28)
        family_size = st.number_input("Family size", min_value=1, max_value=20, value=4)
        work_hours = st.number_input("Work hours/day", min_value=0, value=9)
    with c2:
        avg_sleep_hours = st.slider("Average sleep hours", 0.0, 12.0, 7.0, 0.25)
        screen_time_hours = st.slider("Daily screen time (hrs)", 0.0, 16.0, 5.0, 0.25)
        tea_cups = st.number_input("Tea cups/day", min_value=0, value=2)
    with c3:
        coffee_cups = st.number_input("Coffee cups/day", min_value=0, value=1)
        late_snack = st.selectbox("Late snack?", ["No", "Yes"]) == "Yes"
        spice_intake = st.slider("Spice intensity (1-5)", 1, 5, 2)

    c4, c5 = st.columns(2)
    with c4:
        physical_activity_min = st.number_input("Physical activity (min/day)", min_value=0, value=40)
        bedtime_variability = st.slider("Bedtime variability (hrs)", 0.0, 5.0, 1.0, 0.25)
        stress_level = st.slider("Stress level (0-10)", 0, 10, 5)
    with c5:
        city_noise_dB = st.slider("City noise (dB)", 30, 120, 60)
        light_pollution_index = st.slider("Light pollution (0-100)", 0, 100, 70)
        air_quality_index = st.slider("Air Quality Index (AQI)", 0, 500, 90)

    is_metro = st.selectbox("Metro city?", ["No", "Yes"]) == "Yes"
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    city = st.text_input("City name", value="Mumbai")

    submitted = st.form_submit_button("Save inputs")

# ---------- Build input dict ----------
def build_input_dict():
    sleep_deficit = max(0, 7 - avg_sleep_hours)
    digital_fatigue = max(0.0, (screen_time_hours - 2) / 6.0)
    env_stress = (city_noise_dB / 100.0) + (light_pollution_index / 200.0) + (air_quality_index / 500.0)
    lifestyle_balance = max(-1.0, min(1.0, (physical_activity_min - 30.0) / 60.0))
    fatigue_env_interaction = digital_fatigue * env_stress
    late_snack_effect = 1.0 if late_snack else 0.0
    is_metro_num = 1 if is_metro else 0

    data = {
        "sleep_deficit": sleep_deficit,
        "digital_fatigue": digital_fatigue,
        "env_stress": env_stress,
        "lifestyle_balance": lifestyle_balance,
        "late_snack_effect": late_snack_effect,
        "fatigue_env_interaction": fatigue_env_interaction,
        "is_metro": is_metro_num,
        "avg_sleep_hours": avg_sleep_hours,
        "screen_time_hours": screen_time_hours,
        "stress_level": stress_level,
        "physical_activity_min": physical_activity_min,
        "age": age,
        "family_size": family_size
    }
    return data

# ---------- Prediction & visuals ----------
st.markdown("---")
col_left, col_right = st.columns([1, 1])
with col_left:
    do_predict = st.button("🔮 Predict Sleep Quality", key="predict_btn")
with col_right:
    viz_choice = st.selectbox("Visualization", ["Gauge", "Radar", "Simulate sleep hours"])

if do_predict:
    # ensure model exists
    if model is None:
        try:
            model = create_dummy_model()
            st.success("Dummy model created.")
        except Exception as e:
            st.error("Model not available and auto-create failed: " + str(e))
            model = None

    if model is None:
        st.error("Prediction not possible: model missing.")
    else:
        input_data = build_input_dict()
        try:
            score = predict_sleep_quality(input_data)
        except Exception as e:
            st.exception(e)
            st.error("Prediction failed.")
        else:
            # big bold summary header
            st.markdown(
                f"<div style='font-weight:1000;font-size:20px;color:#FFFFFF'>Predicted Sleep Quality — <span style='color:#00D4FF; font-weight:1200'>{score} / 100</span></div>",
                unsafe_allow_html=True
            )
            st.markdown(f"**Language:** **{language}**  •  **City:** **{city}**  •  **When:** **{datetime.now().strftime('%Y-%m-%d %H:%M')}**")
            st.markdown("---")

            # message_html — helper returns the localized card (HTML) with big tips and yoga images
            message_html = display_prediction(score, language=language)
            st.markdown(message_html, unsafe_allow_html=True)

            st.markdown("---")
            # Visuals
            if viz_choice == "Gauge":
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={'text': "Sleep Quality"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#00D4FF"},
                           'steps': [
                               {'range': [0, 40], 'color': "#ff6b6b"},
                               {'range': [40, 60], 'color': "#ffdf7e"},
                               {'range': [60, 80], 'color': "#c7f0a6"},
                               {'range': [80, 100], 'color': "#6fe6a7"}]}))
                fig.update_layout(height=360, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            elif viz_choice == "Radar":
                radar = {
                    "SleepHours": input_data["avg_sleep_hours"] / 10.0,
                    "DigitalFatigue": input_data["digital_fatigue"],
                    "EnvStress": min(1.0, input_data["env_stress"] / 2.0),
                    "Activity": min(1.0, input_data["physical_activity_min"] / 60.0),
                    "Stress": input_data["stress_level"] / 10.0
                }
                labels = list(radar.keys())
                vals = list(radar.values())
                fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=labels + [labels[0]], fill='toself'))
                fig.update_traces(fill='toself')
                fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=360)
                st.plotly_chart(fig, use_container_width=True)

            else:
                hours = np.linspace(4, 10, 30)
                preds = []
                for h in hours:
                    tmp = input_data.copy()
                    tmp['avg_sleep_hours'] = h
                    tmp['sleep_deficit'] = max(0, 7 - h)
                    tmp['digital_fatigue'] = max(0.0, (tmp['screen_time_hours'] - 2) / 6.0)
                    tmp['fatigue_env_interaction'] = tmp['digital_fatigue'] * tmp['env_stress']
                    preds.append(predict_sleep_quality(tmp))
                fig = px.line(x=hours, y=preds, labels={"x": "Sleep hours", "y": "Predicted score"})
                st.plotly_chart(fig, use_container_width=True)

            # summary suggestions
            if score >= 80:
                st.success("Excellent — maintain routine. Short: 10-min mindfulness, consistent bedtime.")
            elif score >= 60:
                st.info("Good — reduce screen 1 hr before bed; try light stretching.")
            elif score >= 40:
                st.warning("Moderate — start 7-day digital detox evenings; breathing exercises.")
            else:
                st.error("Poor — prioritize sleep hygiene; consult specialist if persistent.")

            # Downloads
            result = {"timestamp": datetime.now().isoformat(), "score": score, "language": language, "message_html": message_html, "inputs": input_data}
            st.download_button("Download JSON report", json.dumps(result, ensure_ascii=False, indent=2), file_name="sleepsense_report.json")
            st.download_button("Download summary (.txt)", f"SleepSense Summary\nDate: {datetime.now()}\nScore: {score}\n", file_name="sleepsense_summary.txt")

# ---------- Yoga & breathing media (bottom) ----------
st.markdown("---")
st.markdown("<div style='font-weight:1000;font-size:20px;color:#FFFFFF'>🧘 Yoga & Breathing — Quick practiced media</div>", unsafe_allow_html=True)
st.markdown("<div style='color:#BFCBDC;font-weight:900'>One image (live) and one short instructional video below — try these to improve sleep.</div>", unsafe_allow_html=True)
st.markdown("")

# Show a reliable public image (Unsplash) with robust fallback
yoga_image_url = "https://images.unsplash.com/photo-1554306274-5f61b1b5c3b1?auto=format&fit=crop&w=800&q=60"
show_image_safe(yoga_image_url, caption="Short Sun-salutation / gentle stretching (image)", width=560)

# Embed a YouTube video (public) — beginner-friendly guided bedtime yoga (works in Streamlit)
video_url = "https://www.youtube.com/watch?v=v7AYKMP6rOE"
try:
    st.video(video_url)
except Exception:
    st.markdown(f"Video: [Open on YouTube]({video_url})")

st.markdown("---")
st.caption("SleepSense — Final app. For production use validated labels, privacy & security, and host securely.")
