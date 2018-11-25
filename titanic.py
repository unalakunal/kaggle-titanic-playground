#!/usr/bin/python3

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
dev_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")


def generate_report(clf, x, y, printout=False, to_file=False):
    """
    Predicts for validation set and calculates 
    precision,recall,f1-score and support
    for each label
    """
    clf_name = clf.__class__.__name__
    print("creating classification report for ", clf_name)
    if clf_name == "GridSearchCV":
        clf = clf.best_estimator_

    pred_val_y = clf.predict(x)
    # calculate metrics for validation set
    report = classification_report(y_true=y, y_pred=pred_val_y)
    if printout:
        print(report)
    if to_file:
        with open("report/report_" + clf_name, "w") as f:
            f.write(report)


def preprocess():
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

    # use Imputer to change NaN val.s with column mean val.s
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(train_x)
    train_x_imp = imp.fit(train_x).transform(train_x)
    val_x_imp = imp.fit(val_x).transform(val_x)
    test_x_imp = imp.fit(test_x).transform(test_x)

    return train_x_imp, train_y, val_x_imp, val_y, test_x_imp


def predict_test(clf, test_x, to_file=False):
    clf_name = clf.__class__.__name__
    print("creating predictions for", clf_name)
    if clf_name == "GridSearchCV":
        clf = clf.best_estimator_

    pred_y = clf.predict(test_x)
    if to_file:
        write_file(pred_y, "pred_" + clf_name)

    return pred_y


def write_file(pred_y, fname="res"):
    # create result csv file w/ columns=["PassengerId","Survived"]
    res = pd.DataFrame(test_df["PassengerId"])
    res = res.assign(Survived=pd.Series(pred_y)).set_index("PassengerId")
    res.to_csv("pred/" + fname + ".csv")


def main():
    train_x, train_y, val_x, val_y, test_x = preprocess()

    # decision tree
    dtree = DecisionTreeClassifier(criterion="gini")
    dtree.fit(train_x, train_y)
    generate_report(dtree, val_x, val_y, printout=True, to_file=True)

    # grid search on top of decision tree
    grid_dtree = GridSearchCV(
        DecisionTreeClassifier(criterion="gini"),
        param_grid={
            "min_samples_split": range(2, 10),
            "max_depth": range(2, 25),
            "min_samples_leaf": range(1, 10)
        },
        n_jobs=8
    )
    grid_dtree.fit(train_x, train_y)
    print("grid search best param.s", grid_dtree.best_params_)
    print("grid search best score", grid_dtree.best_score_)
    generate_report(grid_dtree, val_x, val_y, printout=True, to_file=True)

    # random forests
    rforests = RandomForestClassifier(n_estimators=100)
    rforests.fit(train_x, train_y)
    generate_report(rforests, val_x, val_y, printout=True, to_file=True)

    # predict test data
    pred_y_dtree = predict_test(dtree, test_x, to_file=True)
    pred_y_dtree = predict_test(grid_dtree, test_x, to_file=True)
    pred_y_dtree = predict_test(rforests, test_x, to_file=True)


if __name__ == "__main__":
    main()
