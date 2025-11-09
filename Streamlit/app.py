import streamlit as st
from model_helper import predict

st.title("Fruit Freshness Classification")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png", "jpeg"]) 

if uploaded_file is not None:
        image_path = "temp_file.jpg" 

    try:
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        
        st.image(uploaded_file, caption="Uploaded File")
        
   
        prediction = predict(image_path)
        
        st.info(f"Predicted Class: {prediction}")

    except Exception as e:
        
        st.error(f"An error occurred during file processing or prediction: {e}")