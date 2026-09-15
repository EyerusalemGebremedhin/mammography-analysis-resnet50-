import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Mammography Classification Console",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    
    .stApp {
        background: #F8FAFC;
    }
    
    /* Optimized intermediate alignment layout bounding box */
    .dashboard-container {
        max-width: 840px;
        margin: 0 auto;
        padding: 1.5rem 1rem;
    }
    
    .app-title-card {
        padding: 2.2rem;
        border-radius: 16px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .app-title-card h1 {
        font-size: 2.2rem;
        color: #0F172A;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    .app-title-card p {
        color: #475569;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }
    
    .notice-banner {
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        background: rgba(220, 38, 38, 0.04);
        border: 1px solid rgba(220, 38, 38, 0.15);
        color: #991B1B;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    .workspace-card {
        border-radius: 16px;
        padding: 2.5rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.5rem;
    }
    
    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0284C7;
        letter-spacing: 0.05em;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 0.5rem;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px !important;
        border: 2px dashed #0284C7 !important;
        background: #F0F9FF !important;
        padding: 2.5rem 1rem !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #0369A1 !important;
    }
    
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #FFFFFF !important;
        color: #0284C7 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #F8FAFC !important;
        box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.15) !important;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.6rem 1.5rem;
        border-radius: 6px;
        font-weight: 800;
        font-size: 1rem;
        letter-spacing: 0.03em;
    }
    .badge-normal {background: #DCFCE7; color: #14532D; border: 1px solid #BBF7D0;}
    .badge-alert {background: #FEE2E2; color: #7F1D1D; border: 1px solid #FCA5A5;}
    .badge-review {background: #FEF3C7; color: #78350F; border: 1px solid #FDE68A;}
    
    .gauge-track {
        width: 100%;
        height: 10px;
        border-radius: 5px;
        background: #E2E8F0;
        overflow: hidden;
        margin: 0.8rem 0;
    }
    .gauge-fill {height: 100%; border-radius: 5px;}
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 0.8rem 0;
        font-weight: 700;
        font-size: 1rem;
        background: #0284C7;
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #0369A1;
        color: white;
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.3);
    }

    div[data-testid="stImage"] img {
        border-radius: 12px;
        max-height: 400px;
        width: auto !important;
        margin: 1.5rem auto;
        display: block;
        border: 1px solid #CBD5E1;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08);
    }
    
    .specs-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.25rem;
        margin-top: 2rem;
    }
    .spec-item {
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        font-size: 0.8rem;
        color: #475569;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    }
    
    .stMarkdown, p, span, label { color: #0F172A !important; }
    div[data-testid="stWidgetLabel"] p { color: #334155 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)

st.markdown("""
<div class="app-title-card">
    <h1>Mammography Classification Console</h1>
    <p><b>Lead Developer:</b> Eyerusalem Gebremdhin | Specialized Computer Vision Framework</p>
</div>
<div class="notice-banner">
    COURSEWORK DISCLAIMER: This computer vision system represents an academic prototype built for an AkiraChix student assignment submission. It is not an FDA-approved clinical diagnostic instrument and must not be used for actual medical decisions or patient triage.
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_saved_network():
    return tf.keras.models.load_model("breast_mammography_model.h5")

try:
    model = load_saved_network()
except Exception:
    st.error("Model array missing. Ensure 'breast_mammography_model.h5' is placed in your active workspace directory.")
    st.stop()
st.markdown('<div class="workspace-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">1. Mammogram Scan Input Canvas</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop screening image asset here", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Active Image Frame Buffer")
    st.write("")
    run_analysis = st.button("Initialize Automated Diagnostic Evaluation", type="primary")
else:
    run_analysis = False
    st.info("Awaiting mammography image file upload to activate inference loops.")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded is not None and run_analysis:
    st.markdown('<div class="workspace-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">2. Model Diagnostics Output</div>', unsafe_allow_html=True)
    
    with st.spinner("Processing tissue matrix mapping..."):
        arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
        tensor_batch = np.expand_dims(arr, axis=0)
        prob_malignant = float(model.predict(tensor_batch, verbose=0).flatten())
        
    pred_class = 1 if prob_malignant >= 0.5 else 0
    is_borderline = 0.42 <= prob_malignant <= 0.58
    
    if is_borderline:
        badge_class, summary_label, accent_color = "badge-review", "BORDERLINE INTERPRETATION", "#D97706"
        plain_summary = "The calculated features fall closely alongside the central decision boundary. Visual patterns represent ambiguous transitions or non-specific tissue density changes."
        confidence = max(prob_malignant, 1 - prob_malignant) * 100
    elif pred_class == 1:
        badge_class, summary_label, accent_color = "badge-alert", "MALIGNANT SIGNALS ISOLATED", "#DC2626"
        plain_summary = "The model isolates asymmetric structural configurations, irregular spicular borders, or dense tissue clusters that align with malignant variations recorded in the reference training set."
        confidence = prob_malignant * 100
    else:
        badge_class, summary_label, accent_color = "badge-normal", "BENIGN STRUCTURES LOGGED", "#16A34A"
        plain_summary = "The network traces smooth, uniform circumscribed edges or standard uniform densities suggesting stable tissue characteristics, clear margins, or benign anomalies."
        confidence = (1 - prob_malignant) * 100

    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <span class="status-badge {badge_class}">{summary_label}</span>
    </div>
    <div style="font-size: 0.9rem; color: #0F172A; display: flex; justify-content: space-between;">
        <span>Structural Correspondence Level</span>
        <span style="font-weight: bold; color: {accent_color};">{confidence:.2f}%</span>
    </div>
    <div class="gauge-track">
        <div class="gauge-fill" style="width: {confidence}%; background: {accent_color};"></div>
    </div>
    <div style="margin-top: 1.5rem; padding: 1.2rem; background: #F8FAFC; border-left: 4px solid {accent_color}; border-radius: 4px;">
        <div style="font-size: 0.75rem; color: #475569; font-weight: bold; margin-bottom: 0.4rem;">PLAIN-LANGUAGE SUMMARY</div>
        <p style="color: #334155; font-size: 0.92rem; line-height: 1.6; margin: 0;">{plain_summary}</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Diagnostic Tensor Layer Data Logs"):
        st.json({
            "Computed Malignancy Density": round(prob_malignant, 5),
            "Computed Benign Metric Scope": round(1 - prob_malignant, 5),
            "Assigned Hard Class Index": pred_class,
            "Is Borderline Signal Active": is_borderline
        })
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="specs-grid">
    <div class="spec-item"><b>Architecture</b><br>ResNet50 CNN</div>
    <div class="spec-item"><b>Dimensions</b><br>224x224 RGB</div>
    <div class="spec-item"><b>Activation Node</b><br>Sigmoid Unit</div>
    <div class="spec-item"><b>Training Base</b><br>CBIS-DDSM Subset</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
