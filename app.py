import re
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl','rb'))
tfidf = pk.load(open('scaler.pkl','rb'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return ' '.join(text.split())

st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter Movie Review")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])
        result = model.predict(vector)[0]

        if result == 1:
            st.success("✅ Positive Review\n1")
        else:
            st.error("❌ Negative Review\n0")
