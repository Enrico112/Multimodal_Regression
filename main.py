import pandas as pd
import numpy as np
from sklearn import linear_model
import math

# get df
df = pd.read_csv('hiring.csv')

# change col names
df = df.rename(columns={"test_score(out of 10)":"test",
                        "interview_score(out of 10)":"score",
                        "salary($)":"salary"})

# add zero to na experience
df.experience = df.experience.fillna('zero')
# parse experience from word to number
for i in range(len(df)):
    if df.experience[i] == 'zero':
        df.experience[i] = 0
    elif df.experience[i] == 'one':
        df.experience[i] = 1
    elif df.experience[i] == 'two':
        df.experience[i] = 2
    elif df.experience[i] == 'three':
        df.experience[i] = 3
    elif df.experience[i] == 'four':
        df.experience[i] = 4
    elif df.experience[i] == 'five':
        df.experience[i] = 5
    elif df.experience[i] == 'six':
        df.experience[i] = 6
    elif df.experience[i] == 'seven':
        df.experience[i] = 7
    elif df.experience[i] == 'eigth':
        df.experience[i] = 8
    elif df.experience[i] == 'nine':
        df.experience[i] = 9
    elif df.experience[i] == 'ten':
        df.experience[i] = 10
    elif df.experience[i] == 'eleven':
        df.experience[i] = 11
    else:
        print('error')

# add mean to na test
df.test = df.test.fillna( math.floor(df.test.mean()))

# fit lin reg
reg = linear_model.LinearRegression()
reg.fit(df[['experience','test','score']],df.salary)

# predict values
# 2yr exp, 9test, 6 score
reg.predict([[2,9,6]])
# 2yr, 10test, 10score
reg.predict([[2,10,10]])

# this is equal to:
# intercept + coef1*2 + coef2*10 + coef3*10
reg.intercept_+np.sum(reg.coef_*[2,10,10])