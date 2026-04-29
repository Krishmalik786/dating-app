import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from catboost import CatBoostClassifier
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()
client = Groq(api_key=api_key)

# ─── Load Models & Artifacts ───────────────────────────────
@st.cache_resource
def load_models():
    try:
        with open('models/stacking_model.pkl', 'rb') as f:
            stacking_model = pickle.load(f)
        with open('models/mlp_model_improved.pkl', 'rb') as f:
            mlp_model = pickle.load(f)
        with open('models/xgb_model_improved.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('models/lgb_model_improved.pkl', 'rb') as f:
            lgb_model = pickle.load(f)

        cb_model = CatBoostClassifier()
        cb_model.load_model('models/cb_model_improved.cbm')

        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_mapping.pkl', 'rb') as f:
            label_mapping = pickle.load(f)
        with open('models/feature_columns_enhanced.pkl', 'rb') as f:
            feature_columns_enhanced = pickle.load(f)
        with open('models/model_results_summary.pkl', 'rb') as f:
            results_summary = pickle.load(f)

        # Build tag list from raw data (mirrors Cell 16 of notebook)
        raw = pd.read_csv('data/dating_dataset.csv')
        all_tags = sorted(set(t.strip() for row in raw['interest_tags'] for t in row.split(',')))

        models = {
            'Stacking Ensemble': stacking_model,
            'MLP (Neural Net)':  mlp_model,
            'XGBoost':           xgb_model,
            'LightGBM':          lgb_model,
            'CatBoost':          cb_model,
        }

        return models, scaler, label_mapping, feature_columns_enhanced, all_tags, results_summary

    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.stop()

models, scaler, label_mapping, feature_columns_enhanced, ALL_TAGS, results_summary = load_models()
reverse_mapping = {v: k for k, v in label_mapping.items()}
best_model_name = results_summary.get('best_model', 'Stacking Ensemble')

# ─── Preprocessing ─────────────────────────────────────────
# Mirrors Cell 34 of the notebook exactly
def preprocess_profile(profile):
    df = pd.DataFrame([profile])

    # ── Ordinal encoding derived from numeric values (Cell 16 bins) ──
    usage_bins   = [0, 5, 15, 30, 60, 120, 180, 301]
    usage_labels = [0, 1, 2, 3, 4, 5, 6]
    df['usage_ordinal'] = pd.cut(
        df['app_usage_time_min'], bins=usage_bins,
        labels=usage_labels, right=False, include_lowest=True
    ).astype(float)

    swipe_bins   = [0, 0.2, 0.4, 0.7, 1.01]
    swipe_labels = [0, 1, 2, 3]
    df['swipe_ordinal'] = pd.cut(
        df['swipe_right_ratio'], bins=swipe_bins,
        labels=swipe_labels, right=False, include_lowest=True
    ).astype(float)

    # ── Interest tag OHE — 49 individual flags (Cell 16) ─────
    interests_str = str(profile.get('interest_tags', ''))
    for tag in ALL_TAGS:
        df[f'tag_{tag}'] = 1 if tag in interests_str else 0

    # ── Engineered features ───────────────────────────────────
    df['interest_count']       = len(interests_str.split(','))
    df['profile_completeness'] = (df['bio_length'] / 500) * 0.5 + (df['profile_pics_count'] / 6) * 0.5
    df['engagement_rate']      = df['mutual_matches'] / (df['likes_received'] + 1)
    df['is_night_user']        = int(profile['last_active_hour'] >= 22 or profile['last_active_hour'] <= 4)
    df['activity_score']       = df['app_usage_time_min'] * df['swipe_right_ratio']
    df['social_score']         = df['message_sent_count'] * df['emoji_usage_rate']
    df['msg_per_match']        = df['message_sent_count'] / (df['mutual_matches'] + 1)
    df['like_efficiency']      = df['mutual_matches'] / (df['likes_received'] + 1)
    df['profile_appeal']       = df['likes_received'] / (df['profile_pics_count'] + 1)
    df['bio_per_pic']          = df['bio_length'] / (df['profile_pics_count'] + 1)
    df['match_ratio']          = df['mutual_matches'] / (df['app_usage_time_min'] + 1)
    df['comm_effort']          = df['message_sent_count'] * df['app_usage_time_min']
    df['hour_sin']             = np.sin(2 * np.pi * df['last_active_hour'] / 24)
    df['hour_cos']             = np.cos(2 * np.pi * df['last_active_hour'] / 24)
    df['log_likes']            = np.log1p(df['likes_received'])
    df['log_msgs']             = np.log1p(df['message_sent_count'])
    df['log_matches']          = np.log1p(df['mutual_matches'])
    df['sq_swipe']             = df['swipe_right_ratio'] ** 2
    df['sq_engagement']        = df['engagement_rate'] ** 2
    df['sq_social']            = df['social_score'] ** 2

    # ── Drop raw columns removed during training ──────────────
    df.drop(columns=['interest_tags'], errors='ignore', inplace=True)

    # ── One-hot encode categoricals ───────────────────────────
    nominal_cols = ['gender', 'sexual_orientation', 'location_type',
                    'income_bracket', 'education_level', 'swipe_time_of_day']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    # ── Scale numeric features (Cell 21 scaler — 28 features) ─
    num_features = [
        'app_usage_time_min', 'swipe_right_ratio', 'likes_received',
        'mutual_matches', 'profile_pics_count', 'bio_length',
        'message_sent_count', 'emoji_usage_rate', 'last_active_hour',
        'usage_ordinal', 'swipe_ordinal',
        'interest_count', 'profile_completeness', 'engagement_rate',
        'activity_score', 'social_score', 'msg_per_match', 'like_efficiency',
        'profile_appeal', 'bio_per_pic', 'match_ratio', 'comm_effort',
        'log_likes', 'log_msgs', 'log_matches', 'sq_swipe', 'sq_engagement', 'sq_social'
    ]
    cols_to_scale = [c for c in num_features if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # ── Task-2 interaction features on scaled values ──────────
    for col_a, col_b, name in [
        ('likes_received',     'swipe_right_ratio',    'likes_x_swipe'),
        ('profile_pics_count', 'bio_length',           'profile_x_bio'),
        ('app_usage_time_min', 'message_sent_count',   'app_x_msg'),
        ('emoji_usage_rate',   'message_sent_count',   'emoji_x_msg'),
        ('engagement_rate',    'social_score',         'engage_x_social'),
        ('like_efficiency',    'profile_completeness', 'like_x_profile'),
    ]:
        a = df[col_a].values if col_a in df.columns else 0
        b = df[col_b].values if col_b in df.columns else 0
        df[name] = a * b

    # ── Align to training feature set ─────────────────────────
    df = df.reindex(columns=feature_columns_enhanced, fill_value=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

# ─── Predict ───────────────────────────────────────────────
def predict(profile, model_name):
    model = models[model_name]
    processed = preprocess_profile(profile)

    raw_pred = model.predict(processed)
    pred_encoded = int(np.array(raw_pred).flatten()[0])
    pred_label   = reverse_mapping[pred_encoded]

    probabilities = model.predict_proba(processed)[0]
    confidence    = float(probabilities[pred_encoded]) * 100

    top5_idx = probabilities.argsort()[-5:][::-1]
    top5 = [(reverse_mapping[i], round(float(probabilities[i]) * 100, 2)) for i in top5_idx]

    return pred_label, confidence, top5, probabilities

# ─── Groq AI Explanation ───────────────────────────────────
def get_ai_explanation(profile, predicted_outcome, confidence):
    prompt = f"""
You are an AI assistant for a dating app.
A machine learning model analyzed a user profile and predicted their likely match outcome.

User Profile:
- Gender: {profile['gender']}
- Sexual Orientation: {profile['sexual_orientation']}
- Location: {profile['location_type']}
- Education: {profile['education_level']}
- Income: {profile['income_bracket']}
- Interests: {profile['interest_tags']}
- App Usage: {profile['app_usage_time_min']} minutes/day
- Swipe Right Ratio: {profile['swipe_right_ratio']}
- Likes Received: {profile['likes_received']}
- Mutual Matches: {profile['mutual_matches']}
- Profile Pics: {profile['profile_pics_count']}
- Bio Length: {profile['bio_length']} characters
- Messages Sent: {profile['message_sent_count']}
- Emoji Usage Rate: {profile['emoji_usage_rate']}
- Active Hour: {profile['last_active_hour']}:00

ML Model Prediction: {predicted_outcome}
Confidence: {confidence:.1f}%

Please provide:
1. A friendly 2-3 sentence explanation of this prediction for the user
2. 2 specific strengths of this profile that improve match chances
3. 2 actionable suggestions to improve their match outcome

Keep the tone warm, encouraging, and constructive.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400
    )
    return response.choices[0].message.content

# ─── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="💘 Dating Match Predictor", layout="centered")

st.title("💘 Dating App Match Predictor")
st.markdown("Fill in your profile details to predict your match outcome and get AI-powered insights.")
st.divider()

# ─── Model selector in sidebar ─────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Settings")
    selected_model = st.selectbox(
        "Choose model",
        list(models.keys()),
        index=list(models.keys()).index('Stacking Ensemble')
    )
    st.caption(f"📌 Best from training: **{best_model_name}**")

    st.divider()
    st.markdown("**All trained models:**")
    for row in results_summary.get('all_results', []):
        st.markdown(f"- {row['Model']}: `{row['Accuracy']:.1f}%`")

# ─── Input Form ────────────────────────────────────────────
st.subheader("👤 Your Profile")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender",
        ["Male", "Female", "Non-binary", "Genderfluid", "Transgender", "Prefer Not to Say"])

    sexual_orientation = st.selectbox("Sexual Orientation",
        ["Straight", "Gay", "Bisexual", "Lesbian", "Pansexual",
         "Asexual", "Queer", "Demisexual"])

    location_type = st.selectbox("Location Type",
        ["Urban", "Suburban", "Metro", "Rural", "Small Town", "Remote Area"])

    income_bracket = st.selectbox("Income Bracket",
        ["Very Low", "Low", "Middle", "Upper-Middle", "High", "Very High"])

    education_level = st.selectbox("Education Level",
        ["High School", "Bachelor's", "Master's", "MBA", "PhD", "Postdoc",
         "No Formal Education"])

    swipe_time_of_day = st.selectbox("When do you usually swipe?",
        ["Morning", "Afternoon", "Evening", "Late Night",
         "After Midnight", "Early Morning"])

with col2:
    interest_tags = st.text_input("Interests (comma separated)",
        value="Fitness, Movies, Traveling")

    app_usage_time_min = st.slider("Daily App Usage (minutes)", 0, 300, 90)

    swipe_right_ratio = st.slider("Swipe Right Ratio", 0.0, 1.0, 0.5, 0.01)

    likes_received = st.number_input("Likes Received", 0, 200, 80)

    mutual_matches = st.number_input("Mutual Matches", 0, 30, 10)

    profile_pics_count = st.slider("Profile Pics Count", 0, 6, 3)

    bio_length = st.slider("Bio Length (characters)", 0, 500, 150)

    message_sent_count = st.number_input("Messages Sent", 0, 100, 40)

    emoji_usage_rate = st.slider("Emoji Usage Rate", 0.0, 1.0, 0.3, 0.01)

    last_active_hour = st.slider("Last Active Hour (0–23)", 0, 23, 20)

st.divider()

# ─── Predict Button ────────────────────────────────────────
if st.button("🔮 Predict My Match Outcome", use_container_width=True):

    profile = {
        "gender":             gender,
        "sexual_orientation": sexual_orientation,
        "location_type":      location_type,
        "income_bracket":     income_bracket,
        "education_level":    education_level,
        "interest_tags":      interest_tags,
        "app_usage_time_min": app_usage_time_min,
        "swipe_right_ratio":  swipe_right_ratio,
        "likes_received":     int(likes_received),
        "mutual_matches":     int(mutual_matches),
        "profile_pics_count": profile_pics_count,
        "bio_length":         bio_length,
        "message_sent_count": int(message_sent_count),
        "emoji_usage_rate":   emoji_usage_rate,
        "last_active_hour":   last_active_hour,
        "swipe_time_of_day":  swipe_time_of_day,
    }

    with st.spinner("Analyzing your profile..."):
        pred_label, confidence, top5, probabilities = predict(profile, selected_model)

    # ─── Results ───────────────────────────────────────────
    st.subheader("🎯 Prediction Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Outcome", pred_label)
    with col2:
        st.metric("Confidence Score", f"{confidence:.1f}%")
    with col3:
        st.metric("Model Used", selected_model.split()[0])

    st.markdown("**Top 5 Possible Outcomes:**")
    for outcome, prob in top5:
        st.progress(min(int(prob), 100), text=f"{outcome} — {prob:.1f}%")

    st.divider()

    # ─── AI Explanation ────────────────────────────────────
    st.subheader("🤖 AI-Powered Insights")
    with st.spinner("Getting AI insights..."):
        explanation = get_ai_explanation(profile, pred_label, confidence)

    st.markdown(explanation)

    st.divider()
    st.caption("⚠️ Note: Predictions are based on a synthetic dataset for demonstration purposes.")
