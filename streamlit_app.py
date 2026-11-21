import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("📩 SMS Spam Detector")
st.write("Check whether a message is Spam or Normal.")

message = st.text_area("Enter your message")

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_tfidf = tfidf.transform([message])

        prediction = model.predict(message_tfidf)[0]
        probabilities = model.predict_proba(message_tfidf)[0]

        spam_probability = probabilities[list(model.classes_).index("spam")]
        ham_probability = probabilities[list(model.classes_).index("ham")]

        if prediction == "spam":
            st.error("🚨 Spam Message Detected!")
        else:
            st.success("✅ Normal Message (Ham)")

        st.write(f"Spam Probability: {spam_probability:.2%}")
        st.write(f"Ham Probability: {ham_probability:.2%}")