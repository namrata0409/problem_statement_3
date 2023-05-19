# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
df = pd.read_csv(r"C:\Users\BAPS\Desktop\Problem3\statement3.csv")
print(df)
df.head()
df.info()
df["Churn"].value_counts()

#exploratory data analysis
import matplotlib.pyplot as plt
import seaborn as sns


#analysing demographic data points
cols = ['gender','SeniorCitizen',"Partner","Dependents"]
numerical = cols

plt.figure(figsize=(20,4))

for i, col in enumerate(numerical):
    ax = plt.subplot(1, len(numerical), i+1)
    sns.countplot(x=str(col), data=df)
    ax.set_title(f"{col}")

#relationship between cost and customer churn    
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)

#relationship between customer churn and technology related variables
cols = ['InternetService',"TechSupport","OnlineBackup","Contract"]

plt.figure(figsize=(12,4))

for i, col in enumerate(cols):
    ax = plt.subplot(1, len(cols), i+1)
    sns.countplot(x ="Churn", hue = str(col), data = df)
    ax.set_title(f"{col}")
    
    
#data preprocessing -cleanup
#total changes - changing datatype
df['TotalCharges'] = df['TotalCharges'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
df.info()
#encoding catagorical variables into numerical 
cat_features = df.drop(['customerID','TotalCharges','MonthlyCharges','SeniorCitizen','tenure'],axis=1)

cat_features.head()
cat_features.info()

#encoding using Scikit-Learn’s label encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_cat = cat_features.apply(le.fit_transform)
df_cat.head()
df_cat.info()

#merge new dataframe into the older one
num_features = df[['customerID','TotalCharges','MonthlyCharges','SeniorCitizen','tenure']]
finaldf = pd.merge(num_features, df_cat, left_index=True, right_index=True)
finaldf.head()
finaldf.info()


#splitting training and test sets
import pickle
from sklearn.model_selection import train_test_split

finaldf = finaldf.dropna()
finaldf = finaldf.drop(['customerID'],axis=1)
finaldf.head()
finaldf.info()
X = finaldf.drop(['Churn'],axis=1)
y = finaldf['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.36, random_state=42)

#oversampling training set
from imblearn.over_sampling import SMOTE

oversample = SMOTE(k_neighbors=5)
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_smote, y_smote

#check the number of samples in each class (to check if they are equal)
y_train.value_counts()

#customer churn prediction using random forrest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=46)
rf.fit(X_train,y_train)

#evaluating prediction model 
from sklearn.metrics import accuracy_score

preds = rf.predict(X_test)
print(accuracy_score(preds,y_test))

#export model into pickle file
pickle.dump(rf, open('sol3.pkl','wb'))


