import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer
import math

_class = "Survived"
_features = ["Pclass", "Sex", "Age", "SibSp", "Parch",  "Fare", "Embarked"]


def enum_sex(df):
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})


def enum_embarked(df):
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})


# get the dataframes
dev_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# enumerate sex & embarked for both df.s
enum_sex(dev_df)
enum_embarked(dev_df)
enum_sex(test_df)
enum_embarked(test_df)

# get train & val df
split = math.floor(0.85 * len(dev_df))
train_df = dev_df[:split]
val_df = dev_df[split:]

train_y = train_df[_class]
train_x = train_df[_features]
val_y = val_df[_class]
val_x = val_df[_features]
test_x = test_df[_features]

# create Imputer to change NaN val.s with their mean
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

imp = imp.fit(train_x)
train_x_imp = imp.fit(train_x).transform(train_x)
val_x_imp = imp.fit(val_x).transform(val_x)
test_x_imp = imp.fit(test_x).transform(test_x)

# train decision tree w/ gini index as classifier
dtree = DecisionTreeClassifier(criterion="gini")
dtree.fit(train_x_imp, train_y)
pred_val_y = dtree.predict(val_x_imp)

# calculate accuracy for validation set
report = classification_report(y_true=val_y, y_pred=pred_val_y)
print("report", report)
