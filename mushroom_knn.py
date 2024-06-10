from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import numpy as np
from ucimlrepo import fetch_ucirepo

import time

import warnings
warnings.filterwarnings('ignore')

def plot_validation_curves(train_results, test_results, title, x_label, y_label, save_name, parameter_range, is_log):

    #Find mean
    train_results_mean = np.mean(train_results, axis=1)
    test_results_mean = np.mean(test_results, axis=1)

    plt.figure()
    #Plot Mean
    if is_log:
        plt.semilogx(parameter_range, train_results_mean)
        plt.semilogx(parameter_range, test_results_mean)
    else:
        plt.plot(parameter_range, train_results_mean)
        plt.plot(parameter_range, test_results_mean)

    #Plot STD
    train_results_std = np.std(train_results, axis=1)
    test_results_std = np.std(test_results, axis=1)
    plt.fill_between(parameter_range, train_results_mean-train_results_std, train_results_mean + train_results_std, alpha=0.4)
    plt.fill_between(parameter_range, test_results_mean - test_results_std, test_results_mean + test_results_std, alpha=0.4)

    plt.legend(['Train Results', 'Test Results'])
    plt.title(title, fontsize=10)
    # plt.ylim(0.4,1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(save_name)
    plt.show()
    return

def plot_learning_curves(x_range, train_results, test_results, title, x_label,y_label, save_name, parameter_range):
    #Find mean
    train_results_mean = np.mean(train_results, axis=1)
    test_results_mean = np.mean(test_results, axis=1)

    plt.figure()
    #Plot MEAN
    plt.plot(x_range, train_results_mean)
    plt.plot(parameter_range, test_results_mean)

    #Plot STD
    test_results_std = np.std(test_results, axis=1)
    train_results_std = np.std(train_results, axis=1)
    plt.fill_between(parameter_range, train_results_mean - train_results_std, train_results_mean + train_results_std, alpha=0.4)
    plt.fill_between(parameter_range, test_results_mean - test_results_std, test_results_mean + test_results_std, alpha=0.4)

    plt.legend(['Train Results', 'Test Results'])
    plt.title(title, fontsize=10)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.ylim(0.7,1)
    plt.grid()
    plt.savefig(save_name)
    plt.show()
    return


def main_mushroom():
    mushroom = fetch_ucirepo(id=73)
    mushroom_data = mushroom.data

    X = mushroom_data.features
    y = mushroom_data.targets.poisonous

    mappings = list()
    encoder = LabelEncoder()

    for column in range(len(X.columns)):
        X[X.columns[column]] = encoder.fit_transform(X[X.columns[column]])
        mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
        mappings.append(mappings_dict)

    y[y == 'p'] = 1
    y[y == 'e'] = 0
    y = y.astype(int)

    # Pre process data onto the same scale
    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=620)

    ttr = open("./time_train_results.txt","a")
    ca = open("./classification_accuracy.txt","a")
    bp = open("./best_params.txt","a")

    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

    parameter_range = np.arange(1,30,1)

    #Alternating K
    train_results, test_results = validation_curve(knn, X_train, y_train, param_name='n_neighbors', param_range=parameter_range, cv=5)
    plot_validation_curves(train_results=train_results, test_results=test_results, title="Mushroom - KNN - Validation: Accuracy vs K (# Neighbors)", x_label="K", y_label="Accuracy", save_name="mushroom_KNN_validation_1_K", parameter_range=parameter_range, is_log=False)

    #Alternating p - analyze results
    total_train_results = []
    total_test_results = []

    p_range = ["Manhattan", "Euclidean", "Minkowski"]

    knn_1 = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=1)
    knn_1.fit(X_train, y_train)
    y_predict_train = knn_1.predict(X_train)
    y_predict_test = knn_1.predict(X_test)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_test, y_predict_test))

    knn_2 = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)
    knn_2.fit(X_train, y_train)
    y_predict_train = knn_2.predict(X_train)
    y_predict_test = knn_2.predict(X_test)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_test, y_predict_test))

    knn_3 = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=3)
    knn_3.fit(X_train, y_train)
    y_predict_train = knn_3.predict(X_train)
    y_predict_test = knn_3.predict(X_test)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_test, y_predict_test))

    train_results_mean = np.array(total_train_results)
    test_results_mean = np.array(total_test_results)

    plt.figure()
    X_axis = np.arange(len(p_range))

    plt.bar(X_axis - 0.2, train_results_mean, 0.4)
    plt.bar(X_axis + 0.2, test_results_mean, 0.4)
    plt.xticks(X_axis, p_range)

    for i in range(len(p_range)):
        plt.text(i - 0.2, train_results_mean[i], str(round(train_results_mean[i],3)), ha='center', va='bottom')
        plt.text(i + 0.2, test_results_mean[i], str(round(test_results_mean[i],3)), ha='center', va='bottom')

    plt.legend(['Train Results', 'Test Results'], loc='best')
    plt.title('Mushroom - KNN - Validation: Accuracy vs Distance Metrics (p)', fontsize=10)
    plt.xlabel('p')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('mushroom_KNN_validation_2_p.png')
    plt.show()

    # Optimize hyperperameters using GridSearchCV
    parameter_range = np.arange(1,100,1)
    param_grid = {'n_neighbors': parameter_range, 'p': [1, 2, 3], 'weights': ['uniform','weighted']}

    knn_optimized = GridSearchCV(knn, param_grid=param_grid, cv=5)
    start = time.time()
    knn_optimized.fit(X_train, y_train)
    end = time.time()

    y_predictions = knn_optimized.predict(X_test)
    scoring_accuracy = accuracy_score(y_test, y_predictions)

    classifier_accuracy = scoring_accuracy
    print(f"Mushroom - KNN - Optimized Accuracy: {classifier_accuracy}")
    print(f"Mushroom - KNNOptimized Parameters: {str(knn_optimized.best_params_)} \n")
    print(f"Mushroom - KNN - Time taken {str(end - start)}")

    ca.write(f"Mushroom - KNN - Optimized Accuracy: {classifier_accuracy}\n")
    ttr.write(f"Mushroom - KNN - Time taken {str(end - start)}\n")
    bp.write(f"Mushroom - KNNOptimized Parameters: {str(knn_optimized.best_params_)} \n")

    knn_optimized_model = KNeighborsClassifier(n_neighbors=knn_optimized.best_params_['n_neighbors'], p=knn_optimized.best_params_['p'], weights=knn_optimized.best_params_['weights'])
    train_sizes_abs, train_results, test_results= learning_curve(knn_optimized_model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

    param_range = np.linspace(0.1,1.0,10)*100
    plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results, title="Mushroom - KNN - Accuracy vs Learning Rate (Optimized Params)", x_label="Training Examples (%)", y_label="Accuracy", save_name="mushroom_KNN_learning_1.png",parameter_range=param_range)


if __name__ == '__main__':
    main_mushroom()