import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st
model=tf.keras.models.load_model(r"N:\RvsAI\model_2.keras")
st.sidebar.header("Real vs AI")
st.sidebar.subheader("English")
st.sidebar.text("Read vs AI is a deep learning web application that analyzes images to determine whether they are real or AI-generated.")
st.sidebar.text("Upload an image to get an instant prediction with a clear confidence score.")
st.sidebar.subheader("العربي")
st.sidebar.text("Read vs AI هو تطبيق ويب للتعلم العميق يقوم بتحليل الصور لتحديد ما إذا كانت حقيقية أم تم إنشاؤها بواسطة الذكاء الاصطناعي.")
st.sidebar.text("قم بتحميل صورة للحصول على تنبؤ فوري مع درجة ثقة واضحة.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")
        

IMG_SIZE = (256, 256)

if uploaded_file is not None:
    processed_image =image.resize(IMG_SIZE)
    st.subheader("Processed Image (256x256)")
    st.image(processed_image, caption="Image sent to model")

        
    x = np.array(processed_image) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0][0]

    if pred >= 0.5:
        st.success("Real Image ")
        confidence = pred*100
        st.progress(int(confidence),text=f"{confidence:.0f}%")
        
        
    else:
        st.error("AI-Generated Image")
        con=(1-pred)*100
        st.progress(int(con),text=f"{con:.0f}%")