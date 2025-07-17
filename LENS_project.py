import streamlit as st
import joblib
import pandas as pd

spam_model=joblib.load("spam_classifier.pkl")
language_model=joblib.load("lang_det.pkl")
news_model=joblib.load("news_cat.pkl")
review_model=joblib.load("review.pkl")



st.title("LENSE eXpert(NLP Suits)")
tab1,tab2,tab3,tab4=st.tabs(["🤖 Spam Classifier","Language Detection","Food Review Sentiment","News Classification"])
with tab1:
    msg=st.text_input("Enter Msg")
    if st.button("Prediction" ,key="b1"):
        pred=spam_model.predict([msg])
        if pred[0]==0:
            st.image("spam.jpg")
        else:
            st.image("not_spam.png")

    uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"])
   
  
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

    uploaded_file_lang = st.file_uploader("Upload text file for language detection", type=["txt", "csv"], key="upload_lang")

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

    uploaded_file_review = st.file_uploader("Upload CSV file of reviews", type=["csv"], key="upload_review")

    if uploaded_file_review:
        df_reviews = pd.read_csv(uploaded_file_review, header=None, names=["Review"])
        predictions = review_model.predict(df_reviews["Review"])
        df_reviews["Sentiment"] = predictions
        df_reviews["Sentiment"] = df_reviews["Sentiment"].map({1: "Positive", 0: "Negative"})
        df_reviews.index = range(1, len(df_reviews)+1)
        st.dataframe(df_reviews)


st.sidebar.image("C:\\Users\\Deepak\\Desktop\\img.png")
with st.sidebar.expander("🧑‍🤝‍🧑 About us"):
    st.write("This NLP project is created as part of my exploration into machine learning and its real-world applications.")
with st.sidebar.expander("📞 Contact us"):
    st.write("9319561817")
    st.write(" 📩 deepakmaury1062004@gmail.com")
    


