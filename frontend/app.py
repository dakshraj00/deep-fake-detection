import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🧠 Deepfake Detector")
st.write("Upload an image to check if it's **Real** or **Fake**")

uploaded_file = st.file_uploader("Choose an image",
                                  type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image",
             use_container_width=True)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing..."):
            try:
                # ✅ Correct tuple format for FastAPI UploadFile
                files    = {"file": (uploaded_file.name,
                                     uploaded_file.getvalue(),
                                     uploaded_file.type)}
                response = requests.post("http://127.0.0.1:8000/predict",
                                         files=files, timeout=30)
                response.raise_for_status()
                result = response.json()

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure FastAPI is running.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()

        pred  = result["prediction"]
        conf  = result["confidence"]
        pfake = result["probability_fake"]
        preal = result["probability_real"]

        # Result display
        if pred == "Fake":
            st.error(f"🚨 Prediction : **{pred}**")
        else:
            st.success(f"✅ Prediction : **{pred}**")

        st.metric("Confidence",      f"{conf}%")

        # Probability bar
        col1, col2 = st.columns(2)
        col1.metric("🟢 P(Real)", f"{preal}%")
        col2.metric("🔴 P(Fake)", f"{pfake}%")

        st.progress(int(pfake))   # visual bar showing fake probability