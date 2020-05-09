import numpy as np
import csv
import math
import random

def get_classification(test_sample, distances):
    label = test_sample[-1]
    correct = 0
    wrong = 0
    for i in distances:
        if label==i[0][-1]:
            correct+=1
        else:
            wrong+=1
    if correct>wrong:
        return 1
    else:
        return 0


def train_model(training_data, testing_data, k):
    correct = 0
    wrong = 0
    for test_sample in testing_data:
        distance = []
        for training_sample in training_data:
            distance.append([training_sample, calculate_euclidean_distance(training_sample[:-1], test_sample[:-1])])
        distance.sort(key=lambda x: x[1])
        outcome = get_classification(test_sample, distance[:k])
        if outcome == 1:
            correct+=1
        else:
            wrong+=1
    print(correct, wrong, correct*100/(correct+wrong))


def calculate_euclidean_distance(point1, point2):
    sum = 0.0
    for i in range(len(point1)):
        sum+=(point1[i] - point2[i])**2
    return math.sqrt(sum)

def create_data():
    with open('Breast_cancer_data.csv', newline='') as f:
        reader = csv.reader(f)
        final_data = list(reader)
        final_data.pop(0)
        random.shuffle(final_data)

        for i in range(len(final_data)):
            for j in range(len(final_data[i])):
                final_data[i][j] = float(final_data[i][j])

        training_data = final_data[:int(0.80 * len(final_data))]
        testing_data = final_data[int(0.80 * len(final_data)):]
    return training_data, testing_data

def main():
    training_data, testing_data = create_data()
    train_model(training_data, testing_data, 5)


if __name__ == "__main__":
    main()