import streamlit as st
from PIL import Image
from model_helper import _load_model, predict

st.set_page_config(page_title="Fruit Freshness Classification", page_icon="🍓")
st.title("Fruit Freshness Classification")
st.caption("Upload a fruit image to classify it as Fresh or Spoiled.")


@st.cache_resource
def get_model():
    return _load_model()


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        model = get_model()
        predicted_class, confidence = predict(image, model)

        st.success(f"Prediction: {predicted_class}")
        st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"Unable to process image or run prediction: {e}")
else:
    st.info("Please upload a JPG or PNG image to start.")
