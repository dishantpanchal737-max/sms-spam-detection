import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

x = df["message"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer()

x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

model = MultinomialNB()

model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("--------- SMS SPAM DETECTOR---------")

while True:
    message = input("enter your sms: ")

    if message.lower() == "exit":
        break
    message_tfidf = tfidf.transform([message])

    prediction = model.predict(message_tfidf)

    if prediction[0] == "spam":
        print("🚨 spam message")
    else:
        print("✔️normal message")

joblib.dump(model, "spam_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("model and TF-IDF saved successfully!")
