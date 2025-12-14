# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Student Stress & Focus Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# Styling
# =====================================================
PRIMARY = "#4CAF50"
ACCENT = "#1E88E5"
DANGER = "#E53935"
WARNING = "#FB8C00"
SUCCESS = "#43A047"
BG = "#f9fafb"
TEXT = "#333"

st.markdown(f"""
<style>
:root {{
  --primary: {PRIMARY};
  --accent: {ACCENT};
  --danger: {DANGER};
  --warning: {WARNING};
  --success: {SUCCESS};
  --bg: {BG};
  --text: {TEXT};
}}
body {{ background-color: var(--bg); color: var(--text); }}
.stButton>button {{
  background-color: var(--primary);
  color: white; border-radius: 8px; border: none; padding: 0.5rem 1rem;
}}
.stTabs [role="tab"] {{
  background-color: #eef6ff; color: #000; padding: 8px; border-radius: 5px; margin-right: 6px;
}}
.block-container {{ padding-top: 1rem; }}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Paths
# =====================================================
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
TRACK_FILE = MODELS_DIR / "weekly_tracker.csv"  # نحفظ التتبع هنا

# =====================================================
# Load Latest Model
# =====================================================
models = sorted(MODELS_DIR.glob("stress_model_*.joblib"))
if not models:
    st.error("Model not found. Run train.py first.")
    st.stop()

MODEL_PATH = models[-1]
pipe = joblib.load(MODEL_PATH)

# Load metadata if exists
meta_file = MODELS_DIR / "metadata.json"
metadata = None
if meta_file.exists():
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

# =====================================================
# Session State
# =====================================================
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =====================================================
# Helpers
# =====================================================
def stress_label_map(v):
    return {0: "Low", 1: "Medium", 2: "High"}.get(int(v), "Medium")

def stress_style(level):
    if level == "Low":
        return "🟢", SUCCESS
    if level == "Medium":
        return "🟡", WARNING
    return "🔴", DANGER

def focus_label(sleep, support):
    # بسيط وواضح
    if sleep >= 7 and support >= 3:
        return "Good"
    elif sleep >= 5:
        return "Average"
    return "Poor"

def focus_percent(sleep, support):
    # تحويل لدرجة نسبية لعرض progress
    base = 50
    sleep_bonus = min(max((sleep - 5) * 10, -20), 30)   # من -20 إلى +30
    support_bonus = support * 8                          # 0 إلى 40
    score = max(0, min(100, base + sleep_bonus + support_bonus))
    return int(score)

def save_weekly_row(row: dict):
    MODELS_DIR.mkdir(exist_ok=True)
    df_row = pd.DataFrame([row])
    if TRACK_FILE.exists():
        df = pd.read_csv(TRACK_FILE)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    df.to_csv(TRACK_FILE, index=False)

def load_weekly():
    if TRACK_FILE.exists():
        return pd.read_csv(TRACK_FILE)
    return pd.DataFrame(columns=["date", "stress", "focus", "anxiety_level", "depression",
                                 "sleep_quality", "academic_performance", "social_support"])

# =====================================================
# Tabs
# =====================================================
tab_results, tab_weekly, tab_form, tab_analysis = st.tabs([
    "📊 النتائج",
    "📅 التتبع الأسبوعي",
    "📝 Questionnaire | الاستبيان",
    "📈 تحليل النموذج"
])

# =====================================================
# Questionnaire Tab
# =====================================================
with tab_form:
    st.subheader("📝 Questionnaire | الاستبيان")

    with st.form("questionnaire_form"):
        q_anxiety = st.select_slider(
            "هل شعرت بالقلق خلال الأسبوع الماضي؟",
            options=["أبدًا", "قليلًا", "أحيانًا", "كثيرًا", "دائمًا"]
        )

        q_depression = st.select_slider(
            "هل شعرت بانخفاض في المزاج أو فقدان الاهتمام؟",
            options=["أبدًا", "قليلًا", "أحيانًا", "كثيرًا", "دائمًا"]
        )

        sleep_quality = st.slider("عدد ساعات النوم يوميًا", 0, 10, 7)
        academic_performance = st.slider("المعدل التراكمي GPA", 0.0, 4.0, 2.5)
        q_support = st.select_slider(
            "مستوى الدعم الاجتماعي",
            options=["ضعيف جدًا", "ضعيف", "متوسط", "جيد", "قوي"]
        )

        submit_q = st.form_submit_button("🔍 تنبأ")

    if submit_q:
        anxiety_map = {"أبدًا": 0, "قليلًا": 5, "أحيانًا": 10, "كثيرًا": 20, "دائمًا": 30}
        depression_map = anxiety_map
        support_map = {"ضعيف جدًا": 0, "ضعيف": 1, "متوسط": 3, "جيد": 4, "قوي": 5}

        anxiety = anxiety_map[q_anxiety]
        depression = depression_map[q_depression]
        social_support = support_map[q_support]

        # Build Input
        X_new = pd.DataFrame([{
            "anxiety_level": anxiety,
            "depression": depression,
            "sleep_quality": sleep_quality,
            "academic_performance": academic_performance,
            "social_support": social_support
        }])

        # Feature Engineering (نفس منطق التدريب)
        X_new["mental_load_index"] = (X_new["anxiety_level"] + X_new["depression"]) / 2
        X_new["sleep_support_interaction"] = X_new["sleep_quality"] * X_new["social_support"]

        pred = pipe.predict(X_new)[0]

        stress = stress_label_map(pred)
        focus = focus_label(sleep_quality, social_support)
        focus_pct = focus_percent(sleep_quality, social_support)
        icon, color = stress_style(stress)

        st.session_state.last_result = {
            "date": datetime.date.today().isoformat(),
            "stress": stress,
            "focus": focus,
            "anxiety_level": anxiety,
            "depression": depression,
            "sleep_quality": sleep_quality,
            "academic_performance": academic_performance,
            "social_support": social_support
        }

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric(label="مستوى التوتر", value=f"{icon} {stress}")
        with c2:
            st.metric(label="مستوى التركيز", value=focus)
        with c3:
            st.progress(focus_pct, text=f"Focus score: {focus_pct}%")

        st.markdown(f"""
        <div style="margin-top:0.5rem;padding:0.75rem;border-left:6px solid {color};background:#fff;border-radius:8px">
        <b>ملاحظة:</b> النتائج مبنية على المدخلات الحالية، جرّب تعديل النوم أو الدعم الاجتماعي وشاهد التغييرات فورًا.
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# Results Tab
# =====================================================
with tab_results:
    st.subheader("📊 النتائج والتوصيات")
    if st.session_state.last_result:
        r = st.session_state.last_result
        icon, color = stress_style(r["stress"])

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric(label="مستوى التوتر", value=f"{icon} {r['stress']}")
        with c2:
            st.metric(label="مستوى التركيز", value=r["focus"])
        with c3:
            pct = focus_percent(r["sleep_quality"], r["social_support"])
            st.progress(pct, text=f"Focus score: {pct}%")

        st.markdown("### ✅ التوصيات")
        recs = []
        # توصيات ديناميكية
        if r["sleep_quality"] < 6:
            recs.append("🛌 حاول تحسين روتين النوم (ثبّت ميعاد النوم + قلل الشاشات قبل النوم).")
        if r["social_support"] < 2:
            recs.append("🤝 زوّد الدعم الاجتماعي (تواصل مع أصدقاء/عائلة، مجموعات دراسة).")
        if r["anxiety_level"] > 20:
            recs.append("🧘 مارس تمارين التنفس العميق 5–10 دقائق يوميًا.")
        if r["depression"] > 15:
            recs.append("📋 قسّم المهام الكبيرة إلى خطوات صغيرة مع راحات قصيرة.")

        for rec in recs:
            st.write(rec)
    else:
        st.info("قم بالتنبؤ من تبويب الاستبيان أولًا لعرض النتائج هنا.")

# =====================================================
# Weekly Tracking
# =====================================================
with tab_weekly:
    st.subheader("📅 التتبع الأسبوعي")
    c1, c2 = st.columns([1, 1])

    if st.session_state.last_result:
        if c1.button("💾 حفظ الحالة الحالية في CSV"):
            save_weekly_row(st.session_state.last_result)
            st.success("تم حفظ الحالة بنجاح في models/weekly_tracker.csv")
    else:
        st.info("لا توجد نتيجة حالية للحفظ. قم بالتنبؤ أولًا من تبويب الاستبيان.")

    # عرض الجدول والرسوم
    dfw = load_weekly()
    if not dfw.empty:
        st.markdown("### البيانات المحفوظة")
        st.dataframe(dfw, use_container_width=True)

        # تحويل labels لقيم رقمية للعرض البياني
        stress_map = {"Low": 1, "Medium": 2, "High": 3}
        focus_map = {"Poor": 1, "Average": 2, "Good": 3}

        df_plot = dfw.copy()
        df_plot["date"] = pd.to_datetime(df_plot["date"])
        df_plot["stress_score"] = df_plot["stress"].map(stress_map)
        df_plot["focus_score"] = df_plot["focus"].map(focus_map)

        st.markdown("### تطور المؤشرات بمرور الوقت")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(df_plot.set_index("date")[["stress_score"]], height=240)
        with c2:
            st.line_chart(df_plot.set_index("date")[["focus_score"]], height=240)
    else:
        st.info("لا توجد بيانات محفوظة بعد. احفظ أول حالة لبدء التتبع.")

# =====================================================
# Model Analysis Tab
# =====================================================
with tab_analysis:
    st.subheader("📈 تحليل النموذج")

    if metadata:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ℹ️ Metadata")
            st.json(metadata)
        with c2:
            st.markdown("### الأداء")
            st.metric("Test Accuracy", f"{metadata.get('test_accuracy', 0):.3f}")
            st.metric("Best CV Accuracy", f"{metadata.get('best_cv_accuracy', 0):.3f}")
    else:
        st.info("لم يتم العثور على metadata.json. شغّل train.py بعد التعديلات.")

    # عرض Confusion Matrix
    cm_path = MODELS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.markdown("### Confusion Matrix")
        st.image(str(cm_path))
    else:
        st.info("لم يتم العثور على صورة Confusion Matrix. تأكد من تشغيل train.py وحفظ الصورة.")

    # عرض Feature Importance
    fi_path = MODELS_DIR / "feature_importance.csv"
    if fi_path.exists():
        st.markdown("### Feature Importance")
        fi = pd.read_csv(fi_path)
        fig, ax = plt.subplots(figsize=(6, 3 + 0.3 * len(fi)))
        sns.barplot(x="importance", y="feature", data=fi, ax=ax, palette="Blues_r")
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("لم يتم العثور على feature_importance.csv. تأكد من تشغيل train.py.")
