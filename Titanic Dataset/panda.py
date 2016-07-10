import pandas as pd
import numpy as np
import pylab as P
from bokeh.charts.conftest import test_data

df = pd.read_csv('train.csv', header=0)
print(df['Age'][0:10])
for i in range(1, 4):
    print i, len(df[(df['Sex'] == 'male') & (df['Pclass'] == i)])
df['Age'].hist()
P.show()
df['Age'].dropna().hist(bins=16, range=(0, 80), alpha=.5)
P.show()
df['Gender'] = 4
df['Gender'] = df['Sex'].map(lambda x: x[0].upper())
df['Gender'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)
median_ages = np.zeros((2,3))
median_ages
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df[(df['Gender'] == i) & \
                               (df['Pclass'] == j + 1)]['Age'].dropna().median()

print(median_ages)
df['AgeFill'] = df['Age']
df.head()
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]
print(df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10))
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass
print(df.dtypes[df.dtypes.map(lambda x: x=='object')])
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df = df.drop(['Age'], axis=1)
df = df.dropna()
train_data = df.values
print(train_data)
# Import the random forest package
from sklearn.ensemble import RandomForestClassifier

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)
print(output)