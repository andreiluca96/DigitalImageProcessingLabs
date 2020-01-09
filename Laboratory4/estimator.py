import json

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut


if __name__ == '__main__':
    f = open("features.txt", "r")

    x = np.zeros((384, 76))
    y = np.zeros((384, ))

    idx = 0
    for line in f.readlines():
        feature = np.array(json.loads(line))
        x[idx] = feature
        y[idx] = idx // 6
        idx += 1

    knn = KNeighborsClassifier(n_neighbors=1, algorithm='auto', weights='distance', metric='canberra')

    loo = LeaveOneOut()
    loo.get_n_splits(x)

    idx = 0
    correct = 0
    for train_index, test_index in loo.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        knn.fit(X_train, y_train)
        y_predicted = knn.predict(X_test)

        print(f'{y_test[0]} {y_predicted[0]}')

        idx += 1

        if y_predicted[0] == y_test[0]:
            correct += 1

    print(f'Accuracy: {round(correct/idx * 100, 2)}%')

