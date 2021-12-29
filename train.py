import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso,LassoCV, RidgeCV,LarsCV,ElasticNet,ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle


df = pd.read_csv('Admission_Prediction.csv')
# print("DF loaded")
df['GRE Score'] = df['GRE Score'].fillna(df['GRE Score'].mean())
df['TOEFL Score'] = df['TOEFL Score'].fillna(df['TOEFL Score'].mean())
df['University Rating'] = df['University Rating'].fillna(df['University Rating'].mean())
# print("NA values filled with their respective mean")
df.drop('Serial No.',axis=1,inplace=True)
# print("Column : serial no dropped")
y = df['Chance of Admit']
df.drop(columns=['Chance of Admit'], inplace = True)
x = df
sc = StandardScaler()
x_sc = sc.fit_transform(x)
# print("Data is normalized")
# print("VIF for the dataset")
# for i in range(x_sc.shape[1]): #no multicollinearity as VIF < 10
    # print(variance_inflation_factor(x_sc, i), df.columns[i])
x_train,x_test,y_train,y_test = train_test_split(x_sc,y, test_size=.10, random_state=100)

# print("Running ELasticNET CV")
elasticcv = ElasticNetCV(alphas=None, cv=10)
elasticcv.fit(x_train,y_train)

# print("Running Elastic Net")
elasticLR = ElasticNet(alpha=elasticcv.alpha_,l1_ratio=elasticcv.l1_ratio_)
elasticLR.fit(x_train,y_train)

# print("Accuracy on test Dataset - ")
# print(elasticLR.score(x_test,y_test))

pickle.dump(elasticLR,open('admission_pred_Elastic_LR_model.pickle','wb'))

print(elasticLR.predict(sc.transform([[337.000000,118.0,4.0,4.5,4.5,9.65,1]])))
