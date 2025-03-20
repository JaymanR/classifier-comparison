import time
import numpy as np
from samples import loadDataFile, loadLabelsFile

def extract_features(data):
    features = []
    for d in data:
        pixels = d.getPixels()
        flatten = [1 if px > 0 else 0 for row in pixels for px in row]
        features.append(flatten)

    return np.array(features)

def evaluate_performance(model, train_features, train_labels, test_features, test_labels, iterations=5):
    #print('\nTraining Started\n')
    train_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    for p in train_percentages:
        #print(f'Training with {int(p*100)}% of data...')
        acc = []
        train_times = []
        
        for i in range(iterations):
            num_train_samples = int(len(train_features) * p)
            random_indices = np.random.choice(len(train_features), num_train_samples, replace = False)

            selected_features = train_features[random_indices]
            selected_labels = train_labels[random_indices]

            start_time = time.time()
            model.train(selected_features, selected_labels)
            end_time = time.time()

            train_time = end_time - start_time
            train_times.append(train_time)

            predictions = model.classify(test_features)
            accuracy = np.mean(predictions == test_labels)
            acc.append(accuracy)

        mean_acc = np.mean(acc)
        std_acc = np.std(acc)
        mean_train_time = np.mean(train_times)

        results.append([p, mean_acc, std_acc, mean_train_time])

    return results

def prepare_data(path_data, path_label, n, w, h):
    data = loadDataFile(path_data, n, w, h)
    labels = np.array(loadLabelsFile(path_label, n))

    return data, labels

def print_results(results, type, alg):
    if type == 'face' and alg == 'NB':
        print('\nFace Naive Bayes Classification Results:\n')
    elif type == 'digit' and alg == 'NB':
        print('\nDigit Naive Bayes Classification Results:\n')
    elif type =='face' and alg == 'P':
        print('\nFace Perceptron Classification Results:\n')
    elif type =='digit' and alg == 'P':
        print('\nDigit Perceptron Classification Results:\n')


    for r in results:
        print(f'{r[0]*100}% training data\nMean Acc: {r[1]:.2f}, STD of Acc: {r[2]:.2f}, Mean Time: {r[3]:.4f}\n')