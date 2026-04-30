import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Rice Disease Detection", page_icon="🌾", layout="wide")

st.title("🌾 Rice Leaf Disease Detection System")
st.markdown("Upload a rice leaf image or use your camera to detect diseases")

BACKEND_URL = "http://localhost:8000"

with st.sidebar:
    st.header("About")
    st.info("This system detects 7 rice leaf diseases:\n\n"
            "- Blast\n"
            "- Healthy\n"
            "- Insect\n"
            "- Leaf Folder\n"
            "- Scald\n"
            "- Stripes\n"
            "- Tungro")
    st.header("How to use")
    st.markdown("1. Upload a rice leaf image\n"
                "2. Or use webcam capture\n"
                "3. Get instant disease detection")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    st.subheader("📸 Or Capture from Camera")
    camera_image = st.camera_input("Take a picture")

with col2:
    st.subheader("🔍 Detection Result")

    image_to_predict = None

    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption="Uploaded Image", use_container_width=True)

    elif camera_image is not None:
        image_to_predict = Image.open(camera_image)
        st.image(image_to_predict, caption="Captured Image", use_container_width=True)

    if image_to_predict is not None:
        img_bytes = io.BytesIO()
        image_to_predict.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()

        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    files={"file": img_bytes}
                )

                if response.status_code == 200:
                    result = response.json()

                    disease = result['disease']
                    confidence = result['confidence']

                    if disease == 'healthy':
                        st.success(f"✅ **Result: {disease.upper()}**")
                    else:
                        st.error(f"⚠️ **Result: {disease.upper()}**")

                    st.metric("Confidence", f"{confidence:.2%}")

                    st.subheader("All Class Scores")
                    for cls, score in result['all_scores'].items():
                        st.progress(score, text=f"{cls}: {score:.2%}")

                else:
                    st.error("Failed to get prediction from backend")
            except Exception as e:
                st.error(f"Error: {e}. Make sure backend is running at {BACKEND_URL}")
