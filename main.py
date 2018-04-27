from decision_tree import *
import pandas as pd
from sklearn.model_selection import train_test_split


def accuracy(labels, hypotheses):
    count = 0.0
    correct = 0.0

    for l, h in zip(labels, hypotheses):
        count += 1.0
        if l == h:
            correct += 1.0
    return correct / count


df = pd.read_csv('data/zoo.csv', sep=",")
# print(df.head())
X = df.iloc[:, 1:16]
Y = df.iloc[:, 17]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
X_train = X_train.values.tolist()
y_train = y_train.values.tolist()
X_test = X_test.values.tolist()
y_test = y_test.values.tolist()

# print("X_test=", X_test)
# print("y_test=", y_test)

dt_clf = decision_tree()
dt_clf.fit(X_train, y_train)
hyp = dt_clf.predict(X_test)
score = accuracy(y_test, hyp)
print("Entropy Accuracy score=", score)

dt_clf_g = decision_tree(criterion="gini")
dt_clf_g.fit(X_train, y_train)
hyp_g = dt_clf_g.predict(X_test)
score = accuracy(y_test, hyp_g)
print("Gini Accuracy score=", score)
