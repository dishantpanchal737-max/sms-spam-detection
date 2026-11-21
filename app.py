import joblib

model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

print("SMS Spam Detector Ready!")

while True:
    message = input("\nEnter SMS (or exit): ")

    if message.lower() == "exit":
        break

    message_tfidf = tfidf.transform([message])
    prediction = model.predict(message_tfidf)

    if prediction[0] == "spam":
        print("SPAM MESSAGE!")
    else:
        print("NORMAL MESSAGE (HAM)")