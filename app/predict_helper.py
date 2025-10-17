# file: app/predict_helper.py
# encoding: utf-8
"""
predict_helper.py — helper for SleepSense.
Exports:
 - FEATURE_COLS, MODEL_PATH
 - load_model(), create_dummy_model(force=False)
 - predict_sleep_quality(input_dict)
 - display_prediction(score, language)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# sklearn imports required for dummy model creation
try:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:
    Ridge = None
    make_pipeline = None
    StandardScaler = None

# ---------------- PATHS & FEATURES ----------------
try:
    CURRENT_DIR = Path(__file__).resolve().parent
except NameError:
    CURRENT_DIR = Path.cwd()

MODEL_PATH = CURRENT_DIR.parent / "models" / "ridge_model.pkl"

FEATURE_COLS = [
    'sleep_deficit','digital_fatigue','env_stress','lifestyle_balance',
    'late_snack_effect','fatigue_env_interaction','is_metro',
    'avg_sleep_hours','screen_time_hours','stress_level',
    'physical_activity_min','age','family_size'
]

# ---------------- Dummy model creation ----------------
def create_dummy_model(save_path: Path = MODEL_PATH, force: bool = False):
    """
    Create and save a realistic dummy Ridge pipeline to `save_path`.
    If sklearn is not installed, raises RuntimeError.
    If force=True, overwrite existing model.
    """
    if Ridge is None or make_pipeline is None or StandardScaler is None:
        raise RuntimeError("scikit-learn is required to create dummy model (pip install scikit-learn).")

    if save_path.exists() and not force:
        try:
            return joblib.load(save_path)
        except Exception:
            # if load fails, continue to recreate
            pass

    np.random.seed(42)
    n = 1200
    X = pd.DataFrame({
        'sleep_deficit': np.random.uniform(0, 6, n),
        'digital_fatigue': np.random.uniform(0, 1.2, n),
        'env_stress': np.random.uniform(0, 3, n),
        'lifestyle_balance': np.random.uniform(-1, 1, n),
        'late_snack_effect': np.random.randint(0, 2, n),
        'fatigue_env_interaction': np.random.uniform(0, 2.5, n),
        'is_metro': np.random.randint(0, 2, n),
        'avg_sleep_hours': np.random.uniform(3.5, 9.0, n),
        'screen_time_hours': np.random.uniform(0, 14, n),
        'stress_level': np.random.uniform(0, 10, n),
        'physical_activity_min': np.random.uniform(0, 200, n),
        'age': np.random.uniform(16, 70, n),
        'family_size': np.random.randint(1, 8, n)
    })

    # Hand-crafted realistic target (higher is better)
    y = (
        82
        - 6.5 * X["sleep_deficit"]
        - 11 * X["digital_fatigue"]
        - 5.5 * X["env_stress"]
        + 7.5 * X["lifestyle_balance"]
        - 4.5 * X["late_snack_effect"]
        - 4.5 * X["fatigue_env_interaction"]
        - 3 * X["is_metro"]
        + 5.0 * (X["avg_sleep_hours"] - 6)
        - 1.6 * (X["screen_time_hours"] - 5)
        - 2.2 * (X["stress_level"] - 5)
        + 0.04 * (X["physical_activity_min"] - 30)
        - 0.01 * (X["age"] - 30)
        + 0.3 * (X["family_size"] - 3)
        + np.random.normal(0, 3.0, n)
    )

    y = np.clip(y, 0, 100)

    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipeline.fit(X, y)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"[predict_helper] Dummy model created at: {save_path}")
    return pipeline

# ---------------- Load model ----------------
def load_model(auto_create: bool = True, force_create: bool = False):
    """
    Load model from MODEL_PATH. If missing and auto_create==True, create dummy model.
    Returns model or raises exception if creation fails.
    """
    if MODEL_PATH.exists() and not force_create:
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            # fallthrough to create new model
            print("[predict_helper] Warning: failed to load model, recreating:", e)

    if auto_create:
        return create_dummy_model(save_path=MODEL_PATH, force=force_create)
    raise FileNotFoundError(f"{MODEL_PATH} not found")

# ---------------- Prediction wrapper ----------------
def predict_sleep_quality(input_data: dict) -> float:
    """
    Predict sleep quality (0-100) for given input_data dict (partial allowed).
    Returns rounded float.
    """
    model = load_model(auto_create=True)
    df = pd.DataFrame([input_data])
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    df = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
    pred_raw = model.predict(df)[0]
    pred = float(pred_raw)
    pred = max(0.0, min(100.0, pred))
    return round(pred, 2)

# ---------------- Translations (includes South Indian languages) ----------------
_translations = {
    "English": {
        "excellent": "Excellent Sleep Quality", "poor": "Poor Sleep Quality", "critical": "Critical Sleep Condition",
        "rest": "Take proper rest and reduce stress.",
        "tips": ["Keep a fixed bedtime", "Reduce screen time 1 hour before bed", "Try 10–15 min light yoga/stretch"]
    },
    "Hindi": {
        "excellent": "उत्कृष्ट नींद गुणवत्ता", "poor": "कमज़ोर नींद गुणवत्ता", "critical": "गंभीर नींद स्थिति",
        "rest": "अच्छी नींद लें और तनाव कम करें।",
        "tips": ["नियत सोने का समय रखें", "सोने से 1 घंटा पहले स्क्रीन कम करें", "10–15 मिनट हल्की योग/स्ट्रेचिंग करें"]
    },
    "Tamil": {
        "excellent": "சிறந்த உறக்க தரம்", "poor": "மோசமான உறக்க தரம்", "critical": "கடுமையான உறக்க நிலை",
        "rest": "சரி விட்டு ஓய்வு எடுங்கள்.",
        "tips": ["நித்திய உறக்க நேரம் பின்பற்றவும்", "1 மணி முன் ஸ்க்ரீன் குறைக்கவும்", "10–15 நிமிட யோகா/ஸ்ட்ரெட்ச்"]
    },
    "Telugu": {
        "excellent": "అద్భుత నిద్ర నాణ్యత", "poor": "తగ్గిన నిద్ర నాణ్యత", "critical": "తీవ్ర నిద్ర సమస్య",
        "rest": "సరైన విశ్రాంతి తీసుకోండి మరియు ఒత్తిడిని తగ్గించండి.",
        "tips": ["నియమిత నిద్ర సమయం ఉంచండి", "నిద్రకి 1 గంట ముందు స్క్రీన్ తగ్గించండి", "10–15 నిమిషాల యోగా/స్ట్రెచ్"]
    },
    "Kannada": {
        "excellent": "ಅತ್ಯುತ್ತಮ ನಿದ್ರೆ ಗುಣಮಟ್ಟ", "poor": "ಕೆಟ್ಟ ನಿದ್ರೆ ಗುಣಮಟ್ಟ", "critical": "ಗಂಭীর ನಿದ್ರೆ ಸ್ಥಿತಿ",
        "rest": "ವಿಶ್ರಾಂತಿಯನ್ನೂ ಒಳಗೊಳ್ಳಿ ಮತ್ತು ಒತ್ತಡ ಕಡಿಮೆ ಮಾಡಿ.",
        "tips": ["ನಿಯಮಿತ ನಿದ್ರೆ ಸಮಯದನ್ನ ಮಾಡಿ", "ಒಂದು ಗಂಟೆ ಮೊದಲು ಪರ್ದೆ ಕಡಿಮೆ ಮಾಡಿ", "10–15 ನಿಮಿಷ ಯೋಗ/ಸ್ಟ್ರೆಚ್"]
    },
    "Malayalam": {
        "excellent": "മികച്ച ഉറക്കം", "poor": "തെളിവില്ലാത്ത ഉറക്കം", "critical": "ഗൗരവമായ ഉറക്ക പ്രശ്നം",
        "rest": "ശ്രദ്ധയോടെ വിശ്രമിക്കുക, സമ്മർദ്ദം കുറയ്ക്കുക.",
        "tips": ["സമയബന്ധിതമായി ഉറങ്ങുക", "1 മണിക്കൂര്‍ മുമ്പ് സ്ക്രീന്‍ ഒഴിവാക്കുക", "10–15 മിനിറ്റ് ലഘു യോഗ/സ്റ്റ്രെച്ച്"]
    },
    "Marathi": {
        "excellent": "उत्कृष्ट झोप गुणवत्ता", "poor": "कमी झोप गुणवत्ता", "critical": "गंभीर झोप स्थिती",
        "rest": "चांगली झोप घ्या आणि तणाव कमी करा.",
        "tips": ["नियत झोपेचा वेळ ठेवा", "झोपण्याच्या 1 तास आधी स्क्रीन कमी करा", "10–15 मिनिटे हलकी योग/स्ट्रेच"]
    },
    "Bengali": {
        "excellent": "চমৎকার ঘুমের মান", "poor": "দুর্বল ঘুম", "critical": "গুরুতর ঘুম সমস্যা",
        "rest": "বিশ্রাম নিন এবং চাপ কমান।",
        "tips": ["নিয়মিত ঘুমের সময় রাখুন", "ঘুমের আগে স্ক্রীন কমান", "১০–১৫ মিনিট হালকা যোগ/স্ট্রেচ"]
    },
    "Gujarati": {
        "excellent": "ઉત્કૃષ્ટ ઊંઘ ગુણવત્તા", "poor": "ખરાબ ઊંઘ", "critical": "ગંભીર ઊંઘ સ્થિતિ",
        "rest": "સારી ઊંઘ લો અને તણાવ ઘટાડો.",
        "tips": ["નિયમિત સુવાની સમયસૂચિ રાખો", "સ્લીપ પહેલાં સ્ક્રીન ટાળો", "૧૦–૧૫ મિનિટ યોગ/સ્ટ્રેચ"]
    },
    "Punjabi": {
        "excellent": "ਉਤਕ੍ਰਿਸ਼ਟ ਨੀਂਦ", "poor": "ਕਮਜ਼ੋਰ ਨੀਂਦ", "critical": "ਗੰਭੀਰ ਨੀਂਦ ਦੀ ਹਾਲਤ",
        "rest": "ਚੰਗੀ ਨੀਂਦ ਲਓ ਅਤੇ ਤਣਾਅ ਘਟਾਓ।",
        "tips": ["ਨਿਯਤ ਸਮੇਂ ਸੌਂਵੋ", "ਸਕਰੀਨ ਘਟਾਓ", "10-15 ਮਿੰਟ ਯੋਗ/ਸਟ੍ਰੈਚ"]
    },
    "Odia": {
        "excellent": "ଉତ୍କୃଷ୍ଟ ନିଦ୍ରା", "poor": "କମ୍ଜୋର ନିଦ୍ରା", "critical": "ଗୁରୁତର ନିଦ୍ରା",
        "rest": "ଭଲ ସୁଇନ୍ତୁ ଏବଂ ଚିନ୍ତା କମାନ୍ତୁ।",
        "tips": ["ନିୟମିତ ସମୟରେ ସୁଇନ୍ତୁ", "ସ୍କ୍ରିନ୍ କମ କରନ୍ତୁ", "10-15 ମିନିଟ୍ ଯୋଗ/ସ୍ଟ୍ରେଚ୍"]
    },
    "Urdu": {
        "excellent": "بہترین نیند", "poor": "خراب نیند", "critical": "شدید نیند کا مسئلہ",
        "rest": "اچھی نیند لیں اور دباؤ کم کریں۔",
        "tips": ["مقررہ وقت پر سوئیں", "سکرین کم کریں", "10-15 منٹ ہلکی ورزش کریں"]
    },
    "Assamese": {
        "excellent": "ভাল টোপ", "poor": "দুর্বল টোপ", "critical": "গম্ভীৰ টোপ সমস্যা",
        "rest": "বিশ্ৰাম লওক আৰু চাপ কমাওক।",
        "tips": ["নিয়মিত টোপ সময় বজাই ৰখা", "স্ক্ৰীণ সময় কমোৱা", "১০-১৫ মিনিট লঘু ব্যায়াম"]
    },
    "Nepali": {
        "excellent": "उत्तम निद्रा", "poor": "कमज़ोर निद्रा", "critical": "गम्भीर निद्रा समस्या",
        "rest": "राम्ररी निद्रा लिनुहोस् र तनाव घटाउनुहोस्।",
        "tips": ["नियमित निद्रा समय राख्नुहोस्", "स्क्रीन कम गर्नुहोस्", "१०-१५ मिनेट हल्का व्यायाम"]
    },
    "Bhojpuri": {
        "excellent": "बढ़िया नींद", "poor": "खराब नींद", "critical": "गंभीर नींद",
        "rest": "अच्छे से आराम करीं, तनाव घटाईं।",
        "tips": ["नियत समय पर सोईं", "स्क्रीन समय घटाईं", "10-15 मिनट हल्की व्यायाम"]
    }
}

# yoga images (public)
_YOGA_IMAGES = [
    "https://images.unsplash.com/photo-1554306274-5f61b1b5c3b1?auto=format&fit=crop&w=400&q=60",
    "https://upload.wikimedia.org/wikipedia/commons/7/79/Viparita_Karani.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/3/3a/Alternate_nostril_breathing.png"
]

# ---------------- Levels (30-ish) ----------------
_levels = [
    (99, "Elite sleeper, perfectly balanced."),
    (95, "Nearly perfect sleep hygiene; maintain consistency."),
    (92, "Excellent — small tweaks may improve energy."),
    (88, "Very good; deep sleep mostly intact."),
    (85, "Healthy routine with low stress."),
    (80, "Good sleep pattern; sustain schedule."),
    (75, "Decent — minor irregularities."),
    (70, "Moderate — lifestyle tweaks needed."),
    (66, "Mild fatigue — recoverable."),
    (63, "Average — occasional disturbances."),
    (60, "Fluctuating sleep hours; fix timing."),
    (56, "Noticeable tiredness; take care."),
    (52, "Stress/environment affecting rest."),
    (48, "Repeated disturbances; unwind more."),
    (45, "Inconsistent bedtime; create ritual."),
    (42, "Urban/digital impact lowering quality."),
    (38, "Low deep sleep; breathing helps."),
    (35, "Workload affecting rest; take breaks."),
    (30, "Stress-dominant routine; consider detox."),
    (28, "Overthinking before bed; try journaling."),
    (25, "Screen fatigue reducing deep sleep."),
    (22, "Work pressure visible in sleep pattern."),
    (18, "Fatigue building up; rest during day."),
    (15, "High exhaustion — prioritize rest."),
    (12, "Severe deprivation — seek advice."),
    (8,  "Chronic insomnia risk — avoid stimulants."),
    (5,  "Almost no restorative sleep."),
    (2,  "Near collapse of energy; urgent rest."),
    (0,  "Critical: extremely low rest.")
]

# ---------------- Build localized HTML ----------------
def _build_html(score: float, lang_key: str = "English") -> str:
    lang = _translations.get(lang_key, _translations["English"])
    tips = lang.get("tips", _translations["English"]["tips"])
    rest_text = lang.get("rest", _translations["English"]["rest"])

    for threshold, desc in _levels:
        if score >= threshold:
            # emoji by band
            if threshold >= 95:
                emoji = "💎"
            elif threshold >= 85:
                emoji = "🌙"
            elif threshold >= 70:
                emoji = "🙂"
            elif threshold >= 50:
                emoji = "🟡"
            elif threshold >= 30:
                emoji = "🔴"
            else:
                emoji = "⚰️"

            label = (lang.get("excellent") if threshold >= 80 else (lang.get("poor") if threshold >= 30 else lang.get("critical")))
            title = f"{emoji} {label} ({score})"
            explanation = f"{desc} {rest_text}"
            tips_html = "".join(f"<li>{t}</li>" for t in tips[:3])
            images_html = "".join(f"<img src='{u}' width='120' style='border-radius:6px;margin-right:8px'/>" for u in _YOGA_IMAGES)

            html = f"""
            <div style="padding:12px;border-radius:10px;background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));">
              <div style="font-size:16px;font-weight:700;margin-bottom:6px">{title}</div>
              <div style="margin-bottom:8px">{explanation}</div>
              <strong>Top tips:</strong>
              <ul style="margin:6px 0 10px 18px">{tips_html}</ul>
              <div style="display:flex;gap:8px">{images_html}</div>
            </div>
            """
            return html

    return f"<div>❌ Invalid score ({score}) — please check inputs.</div>"

# ---------------- Public display function ----------------
def display_prediction(pred_value: float, language: str = "English", verbose: bool = False) -> str:
    """
    Return HTML card (localized) for pred_value using language key.
    """
    try:
        score = float(pred_value)
    except Exception:
        score = -9999.0
    if verbose:
        print("[display_prediction] score:", score, "lang:", language)
    return _build_html(score, lang_key=language)
