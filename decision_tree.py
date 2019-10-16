import numpy as np
import pandas as pd
import sklearn.tree as skt
import sklearn.preprocessing as sk

#load data
train_raw = open(u"data/train.csv")
test_raw = open(u"data/test.csv")

train_data = pd.read_csv(train_raw)
test_data = pd.read_csv(test_raw)

#clean data --> Removing columns with high amount of NaN values, the survived column because those are our "ground truths" and any other banal columns.
train_data.drop(['Cabin', 'Name', 'PassengerId'], axis = 1, inplace = True)
test_data.drop(['Cabin', 'Name'], axis = 1, inplace = True)

#through exploring the data, 'age' is the only column with NaN values. This function will drop the rows with any NaN values.
train_data.dropna(inplace = True)

#seperating the 'ground truth' lables from the dataframe. The purpose is to match the form of sklearn's fit function.
survived_lables_train = train_data['Survived']
train_data.drop(['Survived'], axis = 1, inplace = True)

#setup the resultant dataframe to get the unencoded passenger ID's. They should not be modified so I take them from the original Dataframe.
result_final = test_data['PassengerId'].to_frame()
test_data.drop(['PassengerId'], axis = 1, inplace = True)

#label encoder to make the data ML friendly. Has the nice added feature of encoding NaN values on the testing data.
le = sk.LabelEncoder()
train_2 = train_data.apply(le.fit_transform)
test_2 = test_data.apply(le.fit_transform)

#Train
clf = skt.DecisionTreeClassifier().fit(train_2,survived_lables_train)
result = clf.predict(test_2).tolist()
result_final['Survived'] = result

#save results in the proper structure to submit to kaggle! :)
result_final.to_csv(r"C:\Users\Sheil\codeDownloads\Titanic\data\results.csv", header = True, index = None)