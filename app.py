"""KrishiMitra AI - crop leaf disease detection with Grad-CAM and bilingual (English/Hindi) advice.

Run:  streamlit run app.py
"""
import glob
import json
import os

import streamlit as st
from PIL import Image

from src.predictor import Predictor

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = sorted(glob.glob(os.path.join(ROOT, "assets", "samples", "*.jpg")))
METRICS_PATH = os.path.join(ROOT, "reports", "metrics.json")

st.set_page_config(page_title="KrishiMitra AI", page_icon="🌿", layout="wide")

T = {
    "en": dict(
        title="KrishiMitra AI", tagline="Crop leaf disease detection with explainable AI",
        upload="Upload a leaf photo (JPG/PNG)", or_sample="...or try a sample leaf", none="-- choose --",
        your_leaf="Your leaf", heatmap="Where the model looked (Grad-CAM)",
        heat_help="Red/yellow areas influenced the prediction the most.",
        result="Diagnosis", plant="Crop", disease="Condition", confidence="Confidence", top3="Top-3 predictions",
        cause="Cause", symptoms="Symptoms", treatment="Treatment", prevention="Prevention",
        low_conf="Low confidence. The photo may be unclear or the crop may not be in the 38 supported classes. Take a clear photo of a single leaf on a plain background.",
        healthy_msg="Good news: this leaf looks healthy.",
        start="Upload a leaf photo or choose a sample from the left to begin.",
        disclaimer="This is an AI-based aid, not a final diagnosis. Please consult your local Krishi Vigyan Kendra (KVK) or agriculture officer before using any chemical.",
        about="About the model", supported="Supported crops",
        type="Type",
        types={"fungal": "Fungal", "fungal (oomycete)": "Fungus-like (oomycete)", "bacterial": "Bacterial",
               "viral": "Viral", "pest": "Pest (mites)", "healthy": "Healthy"},
        lang="Language",
    ),
    "hi": dict(
        title="कृषिमित्र AI", tagline="फ़सल की पत्ती से रोग पहचान - समझाने वाली AI के साथ",
        upload="पत्ती की फ़ोटो अपलोड करें (JPG/PNG)", or_sample="...या कोई सैंपल पत्ती चुनें", none="-- चुनें --",
        your_leaf="आपकी पत्ती", heatmap="मॉडल ने कहाँ देखा (Grad-CAM)",
        heat_help="लाल/पीले हिस्सों ने नतीजे पर सबसे ज़्यादा असर डाला।",
        result="निदान", plant="फ़सल", disease="स्थिति", confidence="विश्वास स्तर", top3="टॉप-3 अनुमान",
        cause="कारण", symptoms="लक्षण", treatment="उपचार", prevention="बचाव",
        low_conf="विश्वास कम है। फ़ोटो साफ़ नहीं है या यह फ़सल 38 समर्थित वर्गों में नहीं है। सादे बैकग्राउंड पर एक पत्ती की साफ़ फ़ोटो लें।",
        healthy_msg="अच्छी ख़बर: यह पत्ती स्वस्थ दिख रही है।",
        start="शुरू करने के लिए पत्ती की फ़ोटो अपलोड करें या बाईं ओर से सैंपल चुनें।",
        disclaimer="यह AI आधारित सहायता है, अंतिम निदान नहीं। कोई भी दवा डालने से पहले नज़दीकी कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से सलाह लें।",
        about="मॉडल के बारे में", supported="समर्थित फ़सलें",
        type="प्रकार",
        types={"fungal": "फफूंद जनित", "fungal (oomycete)": "जल-फफूंद (ऊमाइसीट)", "bacterial": "जीवाणु जनित",
               "viral": "विषाणु जनित", "pest": "कीट (माइट)", "healthy": "स्वस्थ"},
        lang="भाषा",
    ),
}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.km-hero {background: linear-gradient(120deg,#14532d,#15803d 60%,#65a30d); color:#fff; padding:1.2rem 1.5rem;
          border-radius:14px; margin-bottom:1rem;}
.km-hero h1 {color:#fff; margin:0; font-size:2rem;}
.km-hero p {margin:.2rem 0 0; opacity:.9;}
.km-card {border:1px solid #e5e7eb; border-radius:12px; padding:1rem 1.2rem; background:#fff;}
.km-badge {display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.8rem; font-weight:600;
           background:#ecfccb; color:#3f6212; margin-left:.4rem;}
.km-badge.bad {background:#fee2e2; color:#991b1b;}
.km-big {font-size:1.45rem; font-weight:700; margin:.2rem 0 .4rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_predictor():
    return Predictor()


with st.sidebar:
    lang = "hi" if st.radio("Language / भाषा", ["English", "हिन्दी"], horizontal=True) == "हिन्दी" else "en"
    t = T[lang]
    up = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png"])
    sample_names = [t["none"]] + [os.path.splitext(os.path.basename(p))[0].split("_", 1)[-1].replace("_", " ") for p in SAMPLES]
    choice = st.selectbox(t["or_sample"], sample_names)
    st.divider()
    with st.expander(t["about"]):
        if os.path.exists(METRICS_PATH):
            m = json.load(open(METRICS_PATH))
            st.write(f"MobileNetV2 (transfer learning), {m['n_classes']} classes")
            st.write(f"Test accuracy: **{m['test_accuracy']*100:.1f}%** on {m['n_test']} held-out images")
            st.write(f"Top-3 accuracy: **{m['test_top3_accuracy']*100:.1f}%**")
        st.caption("Dataset: PlantVillage (Hughes & Salathe, 2015)")
    with st.expander(t["supported"]):
        st.write("Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato")

st.markdown(f"<div class='km-hero'><h1>🌿 {t['title']}</h1><p>{t['tagline']}</p></div>", unsafe_allow_html=True)

img = None
if up is not None:
    img = Image.open(up).convert("RGB")
elif choice != t["none"]:
    img = Image.open(SAMPLES[sample_names.index(choice) - 1]).convert("RGB")

if img is None:
    st.info(t["start"])
    st.caption(t["disclaimer"])
    st.stop()

pred = get_predictor()
p = pred.predict(img)
overlay, _, _ = pred.gradcam(img)
info = pred.info(p.label)
suffix = "_hi" if lang == "hi" else "_en"

c1, c2, c3 = st.columns([1, 1, 1.25])
with c1:
    st.markdown(f"**{t['your_leaf']}**")
    st.image(img.resize((320, 320)), width="stretch")
with c2:
    st.markdown(f"**{t['heatmap']}**")
    st.image(overlay, width="stretch")
    st.caption(t["heat_help"])
with c3:
    is_healthy = info["type"] == "healthy"
    badge_cls = "km-badge" if is_healthy else "km-badge bad"
    st.markdown(
        f"<div class='km-card'><div style='color:#6b7280'>{t['result']}</div>"
        f"<div class='km-big'>{info['plant' + suffix]} · {info['disease' + suffix]}</div>"
        f"<div>{t['type']}: <span class='{badge_cls}'>{t['types'].get(info['type'], info['type'])}</span></div></div>",
        unsafe_allow_html=True)
    st.write("")
    st.markdown(f"**{t['confidence']}: {p.confidence*100:.1f}%**")
    st.progress(min(max(p.confidence, 0.0), 1.0))
    if p.confidence < 0.5:
        st.warning(t["low_conf"])
    elif is_healthy:
        st.success(t["healthy_msg"])
    st.markdown(f"**{t['top3']}**")
    for lab, prob in p.top3:
        k = pred.info(lab)
        st.write(f"{k['plant' + suffix]} · {k['disease' + suffix]} — {prob*100:.1f}%")
        st.progress(min(max(prob, 0.0), 1.0))

st.divider()
tabs = st.tabs([f"🦠 {t['cause']}", f"🔍 {t['symptoms']}", f"💊 {t['treatment']}", f"🛡️ {t['prevention']}"])
with tabs[0]:
    st.write(info["cause" + suffix])
for tab, key in zip(tabs[1:], ["symptoms", "treatment", "prevention"]):
    with tab:
        st.markdown("\n".join(f"- {x}" for x in info[key + suffix]))
st.caption("⚠️ " + t["disclaimer"])
