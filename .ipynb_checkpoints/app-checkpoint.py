import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Mandatory Student Information & Coursework Disclaimer
st.title("Mammography Analysis Portal")
st.markdown("##### Developed by: Eyerusalem Gebremdhin")
st.write("**Scan Type:** Screening Mammogram (X-ray) | **Target:** Benign vs. Malignant")
st.warning(DISCimport streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Mandatory Student Information & Coursework Disclaimer
st.title("Mammography Analysis Portal")
st.markdown("##### Developed by: Eyerusalem Gebremdhin")
st.write("**Scan Type:** Screening Mammogram (X-ray) | **Target:** Benign vs. Malignant")
st.warning("DISCLAIMER: This application is a student coursework prototype built exclusively for an AkiraChix academic assignment. It is not an FDA-approved medical diagnostic tool and must not be used for actual clinical decisions.")

# 2. File Upload Widget
uploaded_file = st.file_uploader("Upload a screening mammogram image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preview Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Mammogram Preview", use_container_width=True)
    
    # 3. Cached Model Loader
    @st.cache_resource
    def load_diagnostic_model():
        return tf.keras.models.load_model("breast_mammography_model.h5")
    
    try:
        model = load_diagnostic_model()
        
        # 4. Data Preprocessing (Must match notebook Step 5 exactly)
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # 5. Model Prediction
        with st.spinner("Running deep learning diagnostic inference..."):
            prediction = model.predict(img_batch)[0][0]
            
        # 6. Display Clear Visual Outputs & Translation
        st.subheader("Diagnostic Assessment Results")
        if prediction >= 0.5:
            confidence = prediction * 100
            st.error(f"Prediction: Malignant Signs Detected ({confidence:.2f}% Confidence)")
            st.write("👉 **Plain-Language Translation:** The deep learning model identifies high-density structural changes consistent with malignant tumors. Immediate clinical follow-up and diagnostic verification are recommended.")
        else:
            confidence = (1 - prediction) * 100
            st.success(f"Prediction: Benign / Negative for Malignancy ({confidence:.2f}% Confidence)")
            st.write("👉 **Plain-Language Translation:** The model detects tissue characteristics suggesting stable, non-cancerous anomalies or clean margins. Regular preventative screenings should continue.")
            
    except Exception as e:
        st.error("Could not load the trained model file. Please ensure 'breast_mammography_model.h5' is in the same folder.")
"LAIMER: This application is a student coursework prototype built exclusively for an AkiraChix academic assignment. It is not an FDA-approved medical diagnostic tool and must not be used for actual clinical decisions.")

# 2. File Upload Widget
uploaded_file = st.file_uploader("Upload a screening mammogram image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preview Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Mammogram Preview", use_container_width=True)
    
    # 3. Cached Model Loader
    @st.cache_resource
    def load_diagnostic_model():
        return tf.keras.models.load_model("breast_mammography_model.h5")
    
    try:
        model = load_diagnostic_model()
        
        # 4. Data Preprocessing (Must match notebook Step 5 exactly)
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # 5. Model Prediction
        with st.spinner("Running deep learning diagnostic inference..."):
            prediction = model.predict(img_batch)[0][0]
            
        # 6. Display Clear Visual Outputs & Translation
        st.subheader("Diagnostic Assessment Results")
        if prediction >= 0.5:
            confidence = prediction * 100
            st.error(f"Prediction: Malignant Signs Detected ({confidence:.2f}% Confidence)")
            st.write("**Plain-Language Translation:** The deep learning model identifies high-density structural changes consistent with malignant tumors. Immediate clinical follow-up and diagnostic verification are recommended.")
        else:
            confidence = (1 - prediction) * 100
            st.success(f"Prediction: Benign / Negative for Malignancy ({confidence:.2f}% Confidence)")
            st.write("*Plain-Language Translation:** The model detects tissue characteristics suggesting stable, non-cancerous anomalies or clean margins. Regular preventative screenings should continue.")
            
    except Exception as e:
        st.error("Could not load the trained model file. Please ensure 'breast_mammography_model.h5' is in the same folder.")
