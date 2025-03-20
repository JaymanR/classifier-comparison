import numpy as np
from samples import loadDataFile, loadLabelsFile
from helper import extract_features, evaluate_performance, prepare_data, print_results

class NB_Classifier:
    def __init__(self):
        self.prior_prob = {}
        self.feature_prob = {}

    def train(self, feature_matrix, labels):
        num_samples = len(labels)
        num_features = feature_matrix.shape[1]
        class_labels = np.unique(labels)

        self.prior_prob = {}
        self.feature_prob = {}
        for i in class_labels:
            count = np.sum(labels == i)
            self.prior_prob[i] = count / num_samples

            self.feature_prob[i] = {}
            class_samples = feature_matrix[labels == i]

            for j in range(num_features):
                self.feature_prob[i][j] = {0: 0, 1: 0}

                feature_col = class_samples[:, j]
                count_1 = np.sum(feature_col == 1)
                self.feature_prob[i][j][1] = (count_1 + 1) / (len(class_samples) + 2)
                self.feature_prob[i][j][0] = 1 - self.feature_prob[i][j][1]

    def classify(self, feature_matrix):
        predictions = []
        for i in feature_matrix:
            class_scores = {}
            for j in self.prior_prob:
                log_prob = np.log(self.prior_prob[j])
                for ind, val in enumerate(i):
                    log_prob += np.log(self.feature_prob[j][ind][val])
                class_scores[j] = log_prob
            predictions.append(max(class_scores, key=class_scores.get))

        return np.array(predictions)

def run_NB():
    model = NB_Classifier()

    path_train = 'data/facedata/facedatatrain'
    path_train_label = 'data/facedata/facedatatrainlabels'
    path_test = 'data/facedata/facedatatest'
    path_test_label = 'data/facedata/facedatatestlabels'

    train_data_face, train_labels_face = prepare_data(path_train, path_train_label, 451, 60, 70)
    test_data_face, test_labels_face = prepare_data(path_test, path_test_label, 150, 60, 70)

    train_features_face = extract_features(train_data_face)
    test_features_face = extract_features(test_data_face)

    results_face = evaluate_performance(model, train_features_face, train_labels_face, test_features_face, test_labels_face)

    print_results(results_face, 'face', 'NB')

    path_train = 'data/digitdata/trainingimages'
    path_train_label = 'data/digitdata/traininglabels'
    path_test = 'data/digitdata/testimages'
    path_test_label = 'data/digitdata/testlabels'

    train_data_digit, train_labels_digit = prepare_data(path_train, path_train_label, 5000, 28, 28)

    test_data_raw_digit, test_labels_digit = prepare_data(path_test, path_test_label, 1000, 28,28)

    train_features_digit = extract_features(train_data_digit)
    test_features_digit = extract_features(test_data_raw_digit)

    results_digit = evaluate_performance(model, train_features_digit, train_labels_digit, test_features_digit, test_labels_digit)

    print_results(results_digit, 'digit', 'NB')

if __name__ == '__main__':
    run_NB()