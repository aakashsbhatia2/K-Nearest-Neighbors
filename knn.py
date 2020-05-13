import csv
import math
import random
import sys

def get_classification(test_sample, distances):
    """

    Here, i take the test sample and k nearest points as input parameters.
    I compare the label of the test point with the labels of the neighbors.
    If they are the same, i increment "correct". Else i increment "wrong".
    Eventually, if the correct classification > wrong classification i return 1 to the train_model function. Else, i return 0.

    """

    label = test_sample[-1]
    correct = 0
    wrong = 0
    for i in distances:
        if label == i[0][-1]:
            correct += 1
        else:
            wrong += 1
    if correct > wrong:
        return 1
    else:
        return 0


def train_model(training_data, testing_data, k):

    """

    For each point in the test data, I find the euclidean distance between that point and each point in the training data
    I find the k closest values
    I classify these points based on the k nearest neighbors using the get_classification function
    If the point is correctly classified, the function get_classification returns 1. Else, it returns 0
    Based on this result, i compute the correctly classified points, wrongly classified points and Accuracy of the model

    """
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
    print("Correctly classified: ", correct, "\nWrongly classified: ", wrong, "\nAccuracy: ", round(correct*100/(correct+wrong),2), "%")


def calculate_euclidean_distance(point1, point2):

    """

    Here, I take two points and compute the euclidean distance between the points

    """

    sum = 0.0
    for i in range(len(point1)):
        sum+=(point1[i] - point2[i])**2
    return math.sqrt(sum)

def create_data(path):

    """
    Here, I am reading the csv in the path entered by the user.
    I am creating a list of lists (containing all the information in the csv).
    I remove the first row (header)
    I shuffle the data using random.shuffle
    Since each value in the list generated is a string, I convert it to float
    I split the data into training and testing data (80-20 (train-test) split)
    return the training and testing data

    """
    with open(path, newline='') as f:
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

    """

    To run the code, use the following command in the terminal:
    python3 knn.py --path *path to data* --k *k_val*

    Example:
    python3 knn.py --path Breast_cancer_data.csv --k 3

    """
    k_value = 0
    path = ""

    for i in range(len(sys.argv)):
        if sys.argv[i] == "--path":
            path = sys.argv[i+1]
        if sys.argv[i] == "--k":
            k_value = int(sys.argv[i+1])

    training_data, testing_data = create_data(path)

    train_model(training_data, testing_data, k_value)

if __name__ == "__main__":
    main()