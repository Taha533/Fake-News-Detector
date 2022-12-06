from flask import Flask, render_template,request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
tfidf = TfidfVectorizer(stop_words="english", max_df= 0.7)
model = pickle.load(open("classifier.pkl", "rb")) # #092756;

df = pd.read_csv('WELFake_Dataset.csv')
df = df.dropna()
x = df['text']
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_detector(news):
    tfid_xtrain = tfidf.fit_transform(x_train)
    tfid_xtest = tfidf.transform(x_test)
    news = [news]
    vect = tfidf.transform(news)
    pred = model.predict(vect)

    if int(pred) == 0:
        pred = ["FAKE"]
    elif int(pred) == 1:
        pred = ["REAL"]
    return pred

@app.route('/')

def home():
    return render_template("index.html")
@app.route("/predict", methods = ["POST"])
def predict():
    if request.method == "POST":
        message = request.form['message']
        pred = fake_news_detector(message)
        print(pred)
        return render_template("index.html", prediction = pred)
    else:
        return render_template("index.html", prediction = "Server Error!")

if __name__ == "__main__":
    app.run(debug=True)