import numpy as np
from samples import loadDataFile, loadLabelsFile
from helper import extract_features, evaluate_performance, prepare_data, print_results

class Perceptron_Classifier:
    def __init__(self, unique_labels, num_features):
        self.unique_labels = unique_labels
        self.num_features = num_features
        self.weights = np.zeros((unique_labels, num_features))

    def train(self, feature_matrix, labels, complete_iterations=10):
        for i in range(complete_iterations):
            for x, y in zip(feature_matrix, labels):
                scores = np.dot(self.weights, x)
                y_h = np.argmax(scores)

                if y_h != y:
                    self.weights[y] += x
                    self.weights[y_h] -= x

    def classify(self, feature_matrix):
        predictions = []
        for x in feature_matrix:
            scores = np.dot(self.weights, x)
            y_h = np.argmax(scores)
            predictions.append(y_h)
        return np.array(predictions)
    
def run_Perceptron():
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

    face_model = Perceptron_Classifier(num_classes_face, num_features_face)

    results_face = evaluate_performance(face_model, train_features_face, train_labels_face, test_features_face, test_labels_face)
    
    print_results(results_face, 'face', 'P')

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

    digit_model = Perceptron_Classifier(num_classes_digit, num_features_digit)

    results_digit = evaluate_performance(digit_model, train_features_digit, train_labels_digit, test_features_digit, test_labels_digit)
    
    print_results(results_digit, 'digit', 'P')

if __name__ == '__main__':
    run_Perceptron()
