import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

df=pd.read_csv('Corona_NLP_test.csv')

xtrain, xtest, ytrain, ytest=train_test_split(
    df['OriginalTweet'], df['Sentiment'], test_size=0.2, random_state=42)

model=Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('nb', MultinomialNB())
]).fit(xtrain, ytrain)

ypred=model.predict(xtest)

print("=== Text Classification using TF-IDF and Navie Bayes Classifier ===")
print("classification Report: ")
print(classification_report(ypred, ytest))
print("Confusion Report: ")
print(confusion_matrix(ypred, ytest))

print("Prediction: ")
predict=model.predict(["I dislike this product!"])
print(predict)


