import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

df= pd.read_csv('StressLevelDataset.csv')

df.fillna(df.mode().iloc[0], inplace=True)
for col in df.select_dtypes('object'): df[col]=LabelEncoder().fit_transform(df[col])

print("========== EDA ============")
print(df.describe())
print("Mode: ")
print(df.mode())
print("Median: ")
print(df.median())
print("Skewness: ")
print(df.skew())
print("Kurtosis: ")
print(df.kurt())

sns.histplot(df['stress_level']); plt.show()
sns.countplot(x=df.columns[-1],data=df); plt.show()
sns.heatmap(df.corr(), annot=True); plt.show()

xtrain, xtest, ytrain, ytest=train_test_split(
    df.drop('stress_level', axis=1),df['stress_level'], test_size=0.2, random_state=42
)

model=Pipeline([
    ('scalar', StandardScaler()),
    ('rf', RandomForestClassifier())
]).fit(xtrain, ytrain)

ypred=model.predict(xtest)

print("Classification Report: ")
print(classification_report(ypred, ytest))
print("Confusion Matrix: ")
print(confusion_matrix(ypred, ytest))