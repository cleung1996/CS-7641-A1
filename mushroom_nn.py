from sklearn import preprocessing
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
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
    plt.ylim(0.4,1.1)
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
    test_results_std = np.std(test_results, axis=1)
    train_results_std = np.std(train_results, axis=1)
    plt.fill_between(parameter_range, test_results_mean - test_results_std, test_results_mean + test_results_std, alpha=0.4)
    plt.fill_between(parameter_range, train_results_mean - train_results_std, train_results_mean + train_results_std, alpha=0.4)

    plt.legend(['Train Results', 'Test Results'])
    plt.title(title, fontsize=10)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0.7,1.1)
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

    y[y=='p'] = 1
    y[y=='e'] = 0
    y = y.astype(int)

    # Pre process data onto the same scale
    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=620)

    ttr = open("./time_train_results.txt","a")
    ca = open("./classification_accuracy.txt","a")
    bp = open("./best_params.txt","a")

    cnn = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=620, max_iter=3000)
    parameter_range = np.logspace(-6,3,10)
    train_results, test_results = validation_curve(cnn, X_train, y_train, param_name="alpha", param_range=parameter_range, cv=5)

    # # Plot validation curve 1 - Accuracy Vs Alpha
    plot_validation_curves(train_results, test_results, title="Mushroom - Neural Network - Validation Curve (Accuracy vs Alpha)", x_label= "Alpha", y_label="Accuracy", save_name='mushroom_NN_validation_1_alpha.png', parameter_range=parameter_range, is_log=True)

    train_results, test_results = validation_curve(cnn, X_train, y_train, param_name="learning_rate_init", param_range=parameter_range, cv=5)

    # Plot validation curve 2 - Accuracy Vs Learning Rate
    plot_validation_curves(train_results, test_results, title="Mushroom - Neural Network - Validation Curve (Accuracy vs Learning Rate)", x_label= "Learning Rate", y_label="Accuracy", save_name='mushroom_NN_validation_2_learning_rate.png', parameter_range=parameter_range, is_log=True)


    # Plot validation curve 3 - Accuracy vs Num Nodes

    train_results, test_results = validation_curve(cnn, X_train, y_train, param_name="hidden_layer_sizes",
                                                   param_range=np.arange(2, 20, 2), cv=5)

    plot_validation_curves(train_results=train_results, test_results=test_results,
                           title="Mushroom - Neural Network - Validation Curve (Accuracy vs. # Nodes (Layers kept at 1))",
                           x_label="Num Nodes (in Single Layer)", y_label="Accuracy",
                           save_name="mushroom_NN_validation_3_nodes.png", parameter_range=np.arange(2, 20, 2),
                           is_log=False)

    # Plot validation curve 4 - Accuracy vs Activation
    total_train_results = []
    total_test_results = []

    p_range = ["relu", "logistic", "tanh"]

    nn_relu = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=620, max_iter=3000, activation="relu")
    nn_relu.fit(X_train, y_train)
    y_predict_train = nn_relu.predict(X_train)
    y_predict_test = nn_relu.predict(X_test)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_test, y_predict_test))

    nn_logistic = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=620, max_iter=3000, activation="logistic")
    nn_logistic.fit(X_train, y_train)
    y_predict_train = nn_logistic.predict(X_train)
    y_predict_test = nn_logistic.predict(X_test)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_test, y_predict_test))

    nn_tanh = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=620, max_iter=3000, activation="tanh")
    nn_tanh.fit(X_train, y_train)
    y_predict_train = nn_tanh.predict(X_train)
    y_predict_test = nn_tanh.predict(X_test)
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
        plt.text(i - 0.2, train_results_mean[i], str(round(train_results_mean[i], 3)), ha='center', va='bottom')
        plt.text(i + 0.2, test_results_mean[i], str(round(test_results_mean[i], 3)), ha='center', va='bottom')

    plt.legend(['Train Results', 'Test Results'], loc='best')
    plt.title(''
              'Mushroom - Neural Network - Validation: Accuracy vs Activation Func', fontsize=10)
    plt.xlabel('Activation Function')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('mushroom_NN_validation_4_activation.png')
    plt.show()

    #Optimize hyperperameters using GridSearchCV
    param_grid = {'alpha': np.logspace(-4,3,8), 'learning_rate_init': np.logspace(-5,1,7), 'activation': ['relu', 'tanh'], 'hidden_layer_sizes':np.arange(2, 20, 2)}
    cnn_optimized = GridSearchCV(cnn, param_grid=param_grid, cv=5)


    #Analyze time to train on Clock time
    start = time.time()
    cnn_optimized.fit(X_train, y_train)
    end = time.time()
    ttr.write(f"Mushroom - Neural Network Optimization Training Time: {str(end-start)} \n")
    bp.write(f"Mushroom - Neural Network Optimized Parameters: {str(cnn_optimized.best_params_)} \n")

    y_predict = cnn_optimized.predict(X_test)
    classifier_accuracy = accuracy_score(y_test, y_predict)

    ca.write(f"Neural Network Performance: {classifier_accuracy}\n")

    num_nodes_optimized = cnn_optimized.best_params_['hidden_layer_sizes']


    print("Accuracy for best neural network:", classifier_accuracy)
    print(f"Best Parameters are {str(cnn_optimized.best_params_)}")

    cnn_learn = MLPClassifier(hidden_layer_sizes=(num_nodes_optimized, ), random_state=620, max_iter=3000, learning_rate_init=cnn_optimized.best_params_['learning_rate_init'], alpha=cnn_optimized.best_params_['alpha'], activation=cnn_optimized.best_params_['activation'])
    train_sizes_abs, train_results, test_results = learning_curve(cnn_learn, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=5)

    param_range = np.linspace(0.1,1.0,10)*100
    plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results, title="Mushroom - Neural Network - Learning Curve (Accuracy vs % Trained)", save_name="mushroom_NN_learning_curve_score.png", x_label="Percent Trained", y_label="Accuracy", parameter_range=param_range)

    cnn_optimized = MLPClassifier(hidden_layer_sizes=(num_nodes_optimized, ), random_state=620, max_iter=1, alpha=cnn_optimized.best_params_['alpha'],learning_rate_init=cnn_optimized.best_params_['learning_rate_init'], activation=cnn_optimized.best_params_['activation'], warm_start=True)

    epochs = 300
    train_score = np.zeros(epochs)
    validation_score = np.zeros(epochs)
    train_loss = np.zeros(epochs)

    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_train, y_train, test_size=0.3, random_state=620)

    for epoch in range(0,epochs):
        cnn_optimized.fit(X_train_new, y_train_new)

        train_score[epoch] = accuracy_score(y_train_new, cnn_optimized.predict(X_train_new))
        validation_score[epoch] = accuracy_score(y_test_new, cnn_optimized.predict(X_test_new))

        train_loss[epoch] = cnn_optimized.loss_

    y_predictions = cnn_optimized.predict(X_test_new)
    scoring_accuracy = accuracy_score(y_test_new, y_predictions)
    print(scoring_accuracy)

    plt.figure()
    plt.plot(np.arange(epochs)+1, train_score)
    plt.plot(np.arange(epochs)+1, validation_score)
    plt.legend(['Train Score', 'Validation Score'])
    plt.title("Mushroom - Neural Network - Scoring Curve (Accuracy vs Epochs)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    # plt.ylim(0.87,1)
    plt.grid()
    plt.savefig('mushroom_NN_Scoring.png')
    plt.show()

    plt.figure()
    plt.plot(np.arange(epochs)+1, train_loss)
    # plt.ylim(0,1)
    plt.legend(['Train Loss'])
    plt.title("Mushroom - Neural Network - Training Loss (Accuracy vs Epochs)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig('mushroom_NN_Loss.png')
    plt.show()



if __name__ == '__main__':
    main_mushroom()