import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# CSS styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #ffffff, #e6faff);
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
        color: #333333;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .styled-textarea > label {
        font-size: 26px !important;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    .styled-textarea textarea {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 10px !important;
        padding: 10px !important;
        font-size: 22px !important;
        font-family: 'Segoe UI', sans-serif;
        color: #333333 !important;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
    }
    div.stButton > button {
        display: block;
        margin: 20px auto;
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        color: #333;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 12px;
        border: none;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        transition: background 0.3s ease, transform 0.2s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #fddb92, #d1fdff);
        transform: scale(1.05);
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='main-title'>📰 NewsVerifier</h1>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; padding: 10px 0 20px 0; font-size: 24px;font-family: 'Segoe UI', sans-serif; color: #444;'>
        Enter a news article below to check whether it's REAL or FAKE.
    </div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="styled-textarea">✍️ <b>News Article</b></div>', unsafe_allow_html=True)
    inputn = st.text_area("", "", height=200)


if st.button("🔍 Check News"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.markdown(
                "<div class='result-box'><h3 style='color:green;'>✅ The News is Real!</h3></div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='result-box'><h3 style='color:red;'>❌ The News is Fake!</h3></div>",
                unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze.")
