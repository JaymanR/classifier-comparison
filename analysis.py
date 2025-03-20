import matplotlib.pyplot as plt
import numpy as np
from helper import evaluate_performance, prepare_data, extract_features
from naive_bayes import NB_Classifier
from perceptron import Perceptron_Classifier

def make_plt(nb, perceptron, type):
    size = [r[0] * 100 for r in nb]

    nb_acc = [r[1] for r in nb]
    perceptron_acc = [r[1] for r in perceptron]

    nb_std = [r[2] for r in nb]
    perceptron_std = [r[2] for r in perceptron]

    nb_time = [r[3] for r in nb]
    perceptron_time = [r[3] for r in perceptron]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].plot(size, nb_acc, label='Naive Bayes', marker='o')
    axes[0].plot(size, perceptron_acc, label='Perceptron', marker='s')
    axes[0].set_xlabel('(%) of Data Points Used')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{type}: Data Set Size vs Accuracy')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(size, nb_std, label='Naive Bayes', marker='o')
    axes[1].plot(size, perceptron_std, label='Perceptron', marker='s')
    axes[1].set_xlabel('(%) of Data Points Used')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].set_title(f'{type}: Standard Deviation')
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(size, nb_time, label='Naive Bayes', marker='o')
    axes[2].plot(size, perceptron_time, label='Perceptron', marker='s')
    axes[2].set_xlabel('(%) of Data Points Used')
    axes[2].set_ylabel('Time')
    axes[2].set_title(f'{type}: Data Set Size vs Time')
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()

def analyze():
    path_train = 'data/facedata/facedatatrain'
    path_train_label = 'data/facedata/facedatatrainlabels'
    path_test = 'data/facedata/facedatatest'
    path_test_label = 'data/facedata/facedatatestlabels'

    train_data_face, train_labels_face = prepare_data(path_train, path_train_label, 451, 60, 70)
    test_data_face, test_labels_face = prepare_data(path_test, path_test_label, 150, 60, 70)

    train_features_face = extract_features(train_data_face)
    test_features_face = extract_features(test_data_face)

    num_classes_face = len(np.unique(train_labels_face))
    num_features_face = train_features_face.shape[1]

    nb_model = NB_Classifier()
    perceptron_model = Perceptron_Classifier(num_classes_face, num_features_face)

    print('training NB face model')
    nb_results_face = evaluate_performance(nb_model, train_features_face, train_labels_face, test_features_face, test_labels_face)
    print('training perceprton face model')
    perceptron_results_face = evaluate_performance(perceptron_model, train_features_face, train_labels_face, test_features_face, test_labels_face)

    make_plt(nb_results_face, perceptron_results_face, 'Face')

    path_train = 'data/digitdata/trainingimages'
    path_train_label = 'data/digitdata/traininglabels'
    path_test = 'data/digitdata/testimages'
    path_test_label = 'data/digitdata/testlabels'

    train_data_digit, train_labels_digit = prepare_data(path_train, path_train_label, 5000, 28, 28)
    test_data_digit, test_labels_digit = prepare_data(path_test, path_test_label, 1000, 28, 28)

    train_features_digit = extract_features(train_data_digit)
    test_features_digit = extract_features(test_data_digit)

    num_classes_digit = len(np.unique(train_labels_digit))
    num_features_digit = train_features_digit.shape[1]

    nb_model = NB_Classifier()
    perceptron_model = Perceptron_Classifier(num_classes_digit, num_features_digit)

    print('\ntraining NB digits model')
    nb_results_digit = evaluate_performance(nb_model, train_features_digit, train_labels_digit, test_features_digit, test_labels_digit)
    print('training perceptron digits model')
    perceptron_results_digit = evaluate_performance(perceptron_model, train_features_digit, train_labels_digit, test_features_digit, test_labels_digit)

    make_plt(nb_results_digit, perceptron_results_digit, 'Digits')

    plt.show()

if __name__ == '__main__':
    analyze()


