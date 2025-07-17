import streamlit as st
import joblib
import pandas as pd

spam_model=joblib.load("spam_classifier.pkl")
language_model=joblib.load("lang_det.pkl")
news_model=joblib.load("news_cat.pkl")
review_model=joblib.load("review.pkl")

st.set_page_config(layout="wide", page_title="LENS eXpert", page_icon="🔍")

# Custom background and style
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f3f4f7 0%, #dfe9f3 100%);
        }
        .stTextInput > div > div > input {
            font-size: 20px !important;
            color: #FFFF00;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("""
    <h1 style='background-color: #daf759; 
                font-size: 36px; 
                color: #000; 
                padding: 10px; 
                border-radius: 12px; 
                text-align: center;
                margin-top: -30px;'>
         LENS eXpert(NLP Suite)
    </h1>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📨 Spam Classifier", "🌐 Language Detection", "🍽️ Food Review Sentiment", "🗞️ News Classification"])

with tab1:
    st.subheader("📨 Spam Classifier")
    msg= st.text_input("Enter text")
    if st.button("Prediction" ,key="b1"):
        pred=spam_model.predict([msg])
        if pred[0]==0:
            st.image("spam.jpg")
        else:
            st.image("not_spam.png")

    uploaded_file = st.file_uploader("📁 Upload a file (CSV/TXT):", type=["csv"], key="upload_msg")
   
  
    if uploaded_file:
            
        df_spam=pd.read_csv(uploaded_file,header=None,names=['Msg'])
       
        pred=spam_model.predict(df_spam.Msg)
        df_spam.index=range(1,df_spam.shape[0]+1)
        df_spam["Prediction"]=pred
        df_spam["Prediction"]=df_spam["Prediction"].map({0:'Spam',1:'Not Spam'})
        st.dataframe(df_spam)

with tab2:
    st.subheader("🌍 Language Detection")
    msg= st.text_input("Enter text to detect language")

    if st.button("Detect Language", key="b2"):
        pred = language_model.predict([msg])
        st.success(f"Predicted Language: {pred[0]}")

    uploaded_file_lang = st.file_uploader("📁 Upload a file (CSV/TXT):", type=["csv"], key="upload_lang")
    if uploaded_file_lang:
        df_lang = pd.read_csv(uploaded_file_lang, header=None, names=["Text"])
        predictions = language_model.predict(df_lang["Text"])
        df_lang["Predicted Language"] = predictions
        df_lang.index = range(1, len(df_lang) + 1)
        st.dataframe(df_lang)

with tab3:
    st.subheader("🍔 Food Review Sentiment Analysis")
    
    review = st.text_area("Enter food review text")

    if st.button("Analyze Sentiment", key="b3"):
        prediction = review_model.predict([review])[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Sentiment: {sentiment}")

    uploaded_file_review = st.file_uploader("📁 Upload a file (CSV/TXT):", type=["csv"], key="upload_review")


    if uploaded_file_review:
        df_reviews = pd.read_csv(uploaded_file_review, header=None, names=["Review"])
        predictions = review_model.predict(df_reviews["Review"])
        df_reviews["Sentiment"] = predictions
        df_reviews["Sentiment"] = df_reviews["Sentiment"].map({1: "Positive", 0: "Negative"})
        df_reviews.index = range(1, len(df_reviews)+1)
        st.dataframe(df_reviews)

# --- Tab 4: News Classification ---
with tab4:
    st.subheader("🗞️ News Classification")
    
    msg4 = st.text_input("📰 Enter a news headline or article to classify the topic:", key="msg_input4")
    
    if st.button("🔍 Predict", key="news_detection"):
        pred = news_model.predict([msg4])
        st.success(f"📰 {pred[0]}")

    uploaded_file = st.file_uploader("📁 Upload a file (CSV/TXT):", type=["csv", "txt"], key="news_file")
    if uploaded_file:
        df_news = pd.read_csv(uploaded_file, header=None, names=["Msg"])
        pred = news_model.predict(df_news.Msg)
        df_news.index = range(1, df_news.shape[0] + 1)
        df_news["Prediction"] = pred
        st.dataframe(df_news)      


st.sidebar.image("C:\\Users\\Deepak\\Desktop\\img.png")
with st.sidebar.expander("🧑‍🤝‍🧑 About us"):
    st.write("This NLP project is created as part of my exploration into machine learning and its real-world applications.")
with st.sidebar.expander("📞 Contact us"):
    st.write("9319561817")
    st.write(" 📩 deepakmaury1062004@gmail.com")
with st.sidebar.expander("🤝 Help & Instructions"):
    st.markdown("""
    <ul style='font-size: 14px;'>
        <li>Type or upload text to test the model.</li>
        <li>Use supported file formats: <b>.csv</b> or <b>.txt</b>.</li>
        <li>After prediction, download the result using the button.</li>
    </ul>
    """, unsafe_allow_html=True)
    



    


