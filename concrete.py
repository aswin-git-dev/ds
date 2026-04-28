import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df=pd.read_csv('Concrete_Data.csv')

df.fillna(df.mode().iloc[0], inplace=True)
for col in df.select_dtypes('object'): df[col]=LabelEncoder().fit_transform(df[col])

xtrain, xtest, ytrain, ytest = train_test_split(
    df.drop('Concrete_compressive_strength', axis=1), df['Concrete_compressive_strength'], test_size=0.2
)

model=Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor())
]).fit(xtrain,ytrain)

ypred=model.predict(xtest)
print("MAE",mean_absolute_error(ytest,ypred))
print("MSE",mean_squared_error(ytest, ypred))
print("R2",r2_score(ytest, ypred))