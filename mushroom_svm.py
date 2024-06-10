from sklearn import preprocessing
from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

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
    plt.ylim(0.4,1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(save_name)
    plt.show()
    return

def plot_learning_curves(x_range, train_results, test_results, title, x_label,y_label, save_name, parameter_range):
    #Find mean
    test_results_mean = np.mean(test_results, axis=1)
    train_results_mean = np.mean(train_results, axis=1)

    plt.figure()
    #Plot MEAN
    plt.plot(x_range, train_results_mean)
    plt.plot(parameter_range, test_results_mean)

    #Plot STD
    train_results_std = np.std(train_results, axis=1)
    test_results_std = np.std(test_results, axis=1)
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

    model_svm = svm.SVC(kernel='linear', random_state=620)
    parameter_range = np.logspace(-5,5,11)

    # Validation Curve 1 - Accuracy vs C
    train_results, test_results = validation_curve(model_svm, X_train, y_train, param_name='C', param_range= parameter_range, cv=5)
    plot_validation_curves(train_results=train_results, test_results=test_results, title="Mushroom - SVM - Validation: Accuracy vs C (Linear Kernel)", x_label="C", y_label="Accuracy", save_name="mushroom_SVM_validation_1_C.png", parameter_range=parameter_range, is_log=True)


    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    total_train_results = []
    total_test_results = []

    # # Validation Curve 2 - Accuracy vs Kernel Types
    svm_linear = svm.SVC(kernel='linear', random_state=620)

    svm_linear.fit(X_train, y_train)
    y_predict_test = svm_linear.predict(X_test)
    y_predict_train = svm_linear.predict(X_train)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_predict_test, y_test))

    svm_rbf = svm.SVC(kernel='rbf', random_state=620)
    svm_rbf.fit(X_train, y_train)
    y_predict_test = svm_rbf.predict(X_test)
    y_predict_train = svm_rbf.predict(X_train)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_predict_test, y_test))

    svm_poly = svm.SVC(kernel='poly', random_state=620)
    svm_poly.fit(X_train, y_train)
    y_predict_test = svm_poly.predict(X_test)
    y_predict_train = svm_poly.predict(X_train)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_predict_test, y_test))

    svm_sigmoid = svm.SVC(kernel='sigmoid', random_state=620)
    svm_sigmoid.fit(X_train, y_train)
    y_predict_test = svm_sigmoid.predict(X_test)
    y_predict_train = svm_sigmoid.predict(X_train)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_predict_test, y_test))

    train_results_mean = np.array(total_train_results)
    test_results_mean = np.array(total_test_results)

    plt.figure()
    X_axis = np.arange(len(kernels))

    plt.bar(X_axis - 0.2, train_results_mean, 0.4)
    plt.bar(X_axis + 0.2, test_results_mean, 0.4)
    plt.xticks(X_axis, kernels)

    plt.legend(['Train Results', 'Test Results'])
    plt.title('Mushroom - SVM - Validation: Accuracy vs Kernel', fontsize=10)
    plt.xlabel('Kernel')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('mushroom_SVM_validation_2.png')
    plt.show()

    print(train_results_mean)
    print(test_results_mean)

    #Optimize hyperperameters using GridSearchCV
    param_grid = {'C': parameter_range, 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

    model_svm_optimized = GridSearchCV(model_svm, param_grid=param_grid, cv=5)
    start = time.time()
    model_svm_optimized.fit(X_train, y_train)
    end = time.time()
    print(f"Neural Network Optimized Parameters: {str(model_svm_optimized.best_params_)} \n")
    print(f"Time taken f{str(end-start)}")

    y_predict = model_svm_optimized.predict(X_test)
    classifier_accuracy = accuracy_score(y_test, y_predict)


    ca.write(f"Mushroom - SVM Optimized Accuracy: {classifier_accuracy}\n")
    bp.write(f"Mushroom - SVM Optimized Parameters: {str(model_svm_optimized.best_params_)} \n")
    ttr.write(f"Mushroom - SVM Train Time: {str(end-start)} \n")

    print(f"Mushroom - SVM Train Time: {str(end-start)} \n")
    print(f"Mushroom - SVM Optimized Parameters: {str(model_svm_optimized.best_params_)} \n")
    print(f"Mushroom - SVM Optimized Accuracy: {str(classifier_accuracy)}")

    svm_optimized = svm.SVC(random_state=42, C=model_svm_optimized.best_params_['C'], kernel=model_svm_optimized.best_params_['kernel'])
    train_sizes_abs, train_results, test_results= learning_curve(svm_optimized, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

    param_range = np.linspace(0.1,1.0,10)*100
    plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results, title="Mushroom - SVM - Accuracy vs Learning Rate (Optimized Params)", x_label="Training Examples (%)", y_label="Accuracy", save_name="mushroom_SVM_learning_1.png",parameter_range=param_range)



if __name__ == '__main__':
    main_mushroom()