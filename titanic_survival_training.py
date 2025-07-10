"""
File: titanic_survived.py
Name:
----------------------------------
This file contains 3 of the most important steps
in machine learning:
1) Data pre-processing
2) Training
3) Predicting
"""
import math

TRAIN_DATA_PATH = 'titanic_data/train.csv'         # Training data path
NUM_EPOCHS = 1000                                  # Number of iterations over all data
ALPHA = 0.01                                       # Known as step size or learning rate


def sigmoid(k):
    """
    :param k: float, linear function value
    :return: float, probability of the linear function value
    """
    return 1/(1+math.exp(-k))


def dot(lst1, lst2):
    """
    : param lst1: list, the feature vector
    : param lst2: list, the weights
    : return: float, the dot product of 2 list
    """
    return sum(lst1[i]*lst2[i] for i in range(len(lst1)))


def main():
    # Milestone 1
    training_data = data_pre_processing()

    # (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],y)
    weights = [0]*len(training_data[0][0]) #第二個[0]是指第一位船員的整顆tuple(如上列)，第二個[0]是指這顆tuple的0號位置(即['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    # Milestone 3
    training(training_data, weights)
    print(weights)

    # Milestone 4
    predict(training_data, weights)


# Milestone 1
def data_pre_processing():
    """
    Read the training data from TRAIN_DATA_PATH and get ready for training!
    :return: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    """
    train_data = []
    with open(TRAIN_DATA_PATH,'r') as f:
        is_first = True
        for line in f:
            if is_first:
                is_first = False
            else:
                feature_vector, y = feature_extractor(line)
                train_data.append((feature_vector, y))
    return train_data


# Milestone 2 記得做normalization
def feature_extractor(line):
    """
    : param line: str, the line of data extracted from the training set
    : return: Tuple(list, label), feature_vector and label of a passenger
    """
    line = line.strip()        # Remove the '\n' at the end of each line
    data_list = line.split(',')
    feature_vector = []
    label = int(data_list[1])
    for i in range(len(data_list)):
        if i == 2:
            # Pclass
            feature_vector.append((int(data_list[i])-1)/(3-1))
        elif i == 5:
            # Gender
            if data_list[i] == 'male':
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        elif i == 6:   #處理連續型缺漏資料-->填平均數或中位數
            # Age
            if not data_list[i]: # 即 if False(缺漏資料)
                feature_vector.append((29.699-0.42)/(80-0.42))
            else:
                feature_vector.append((float(data_list[i])-0.42)/(80-0.42))    #年齡有小數點
        elif i == 7:
            # SibSp
            feature_vector.append(int(data_list[i])/8)
        elif i == 8:
            # Parch
            feature_vector.append(int(data_list[i])/6)
        elif i == 10:
            # Fare
            feature_vector.append(float(data_list[i])/512.3292)    #注意票價有小數點
        elif i == 12:
            # Embarked, S=0, C=1, Q=2
            if data_list[i] == 'S':
                feature_vector.append(0/2)
            elif data_list == 'C':
                feature_vector.append(1/2)
            elif data_list == 'Q':
                feature_vector.append(2/2)
            else: #處理連續型缺漏資料-->填眾數
                feature_vector.append(0/2)

    return feature_vector, label


# Milestone 3
def training(training_data, weights):
    """
    : param training_data: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    : param weights: list[float], the weight vector (the parameters on each feature)
    """
    print_every = 50      # controls the number of prints on console
    for epoch in range(NUM_EPOCHS):
        cost = 0
        for x, y in training_data:  # x為7個feature_vector； y為答案
            #################################
            k = dot(weights, x)
            h = sigmoid(k)
            loss = -(y*math.log(h)+(1-y)*math.log(1-h))
            cost += loss
            # S.G.D
            for j in range(len(weights)):
                weights[j] = weights[j] - ALPHA * (h-y) * x[j]
            #################################
        cost /= (2*len(training_data))
        if epoch % print_every == 0:
            print('Cost over all data:', cost)


# Milestone 4
def predict(training_data, weights):
    """
    : param training_data: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    : param weights: list[float], the weight vector (the parameters on each feature)
    """
    acc = 0
    num_data = 0
    for x, y in training_data:
        prediction = get_prediction(x, weights)
        print('True Label: ' + str(y) + ' --------> Predict: ' + str(prediction))
        if y == prediction:
            acc += 1
        num_data += 1
    print('---------------------------')
    print('Acc: ' + str(acc / num_data))


def get_prediction(x, weights):
    """
    : param x: list[float], the value of each data on
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    : param weights: list[float], the weight vector (the parameters on each feature)
    : return: int, the classification prediction of x (if it is > 0 then the passenger may survive)
    """
    k = dot(x, weights)
    return 1 if k > 0 else 0


if __name__ == '__main__':
    main()
