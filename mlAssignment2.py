from statistics import mean, stdev
import numpy
import pandas as pd
from collections import Counter
from pyensae.graphhelper import draw_diagram
from sklearn import tree, metrics
from dtreeplt import dtreeplt
import os

# Who wrote which methods/classes
# Caireann wrote the methods clac_gini_uncertainity, info_gain, cart_tree, create_tree_image, classify and test_tree
# Robert wrote the Question, Node and leaf classes and also the best_split and the build_tree methods
# Both wrote the main method which imports the data, runs both classifiers ten times and prints the two trees at the end

# main method written by both Robert and Caireann 
def main():
    # import the hazelnut.csv file using panda
    df = pd.read_csv('mlAssignment2Dataset.csv', header=0, delimiter=",")

    # choose the number of training cases
    training_number = 10

    # find attribute names and class labels from the dataframe
    feature_names = df.columns.tolist()
    target_variable = df.columns[-1]
    target_names = df[target_variable].unique()

    # initialize variables
    average_accuracy = []
    clf_accuracy = []
    learning_curve_accuracy = []
    clf_learning_curve_accuracy = []

    # run the tests ten times and get the average accuracy
    for x in range(0, training_number):
        # randomly shuffle the data and split into training and test subsets
        df = df.sample(frac=1)
        train_split = int((len(df) * 2) / 3)
        training = df.values[0:train_split]
        test = df.values[train_split - 1:]
        # get the number of columns in the test and training sets
        training_length = training.shape[1]
        test_length = test.shape[1]

        # run the sklearn decision tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(training[:, :test_length-1], training[:, training_length-1])

        # Test the imported decision tree algorithm
        test_predictions = clf.predict(test[:, :test_length-1])
        clf_accuracy.append((metrics.accuracy_score(test[:, -1], test_predictions))*100)
        clf_result = "\nScikit CART: test %d accuracy: %f%%" % (x+1, clf_accuracy[x])
        print(clf_result)

        # test the accuracy of our tree
        accuracy = test_tree(build_tree(training), test)
        average_accuracy.append(accuracy)
        result = "Our tree: Test %d has accuracy %d%%" % (x+1, accuracy)
        print(result)

    print("\nAverage accuracy for scikit tree after 10 runs is {:.2f}%, +/- {:.2f}%".format(mean(clf_accuracy), stdev(clf_accuracy)))
    print("Average accuracy after 10 runs is {:.2f}%, +/- {:.2f}%".format(numpy.mean(average_accuracy), stdev(average_accuracy)))

    # print out the sklearn decision tree
    dtree = dtreeplt(model=clf, feature_names=feature_names, target_names=target_names)
    fig = dtree.view()
    fig.savefig('output.png')

    # create a diagram of our decision tree
    final_string = cart_tree(build_tree(training), None)
    create_tree_image(final_string)

    # get data for learning curves and save to file to be used in excel
    for instances in range(2, len(training), 2):
        accuracy = test_tree(build_tree(training[0:instances]), test)
        learning_curve_accuracy.append(accuracy)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(training[0:instances, :training_length-1], training[0:instances, training_length-1])
        test_predictions = clf.predict(test[:, :test_length-1])
        clf_learning_curve_accuracy.append((metrics.accuracy_score(test[:, -1], test_predictions))*100)

    print(clf_learning_curve_accuracy)
    print(learning_curve_accuracy)
    newFile = open("mlAssignment2LearningCurve.csv", 'a+')
    newFile.write(str(learning_curve_accuracy))
    newFile.write(str(clf_learning_curve_accuracy))
    newFile.close()


# class written by Robert
# Data at each node is split depending on a question (eg/ is attribute number 7 greater than 15)
class Question(object):
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def check_answer(self, data_point):
        val = data_point[self.column]
        return val >= self.value

    # return the question as a string to be printed in the node
    def print_question(self):
        return " Is_attribute_%s_greater_than_%s" % (self.column, str(self.value))
        # TODO Replace self.column with something to get attribute name


# class written by Robert
# Leaf node holds the probabilities of an input being a class
class Leaf(object):
    def __init__(self, data):
        self.probability = Counter([row[-1] for row in data])


# class written by Robert
# each node has a question upon which the data at that node will be split into two subsets, true and false
class Node(Question, object):
    def __init__(self, question, true_subtree, false_subtree):
        self.question = question
        self.true_subtree = true_subtree
        self.false_subtree = false_subtree


# function written by Caireann
# The root node should be the attribute with the least gini index
# When gini is 0, the node is a leaf node
# Gini = 1 - SUM(probability of object being classified correctly)**2
def calc_gini_uncertainty(data):
    # takes in the attribute data (not class data)
    # set gini uncertainty to 1
    uncertainty = 1
    # count how many data points (number of rows)
    data_points = len(data)
    # counts occurrences of each label (type of nut)
    count_labels = Counter([row[-1] for row in data])

    # calculate gini uncertainty for the current subset of data
    for label, count in count_labels.items():
        # probability of a data point having a certain label
        p_labels = count / data_points
        p_matching = p_labels ** 2
        # uncertainty is 1 - probability of matching
        uncertainty -= p_matching
    return uncertainty


# function written by Caireann
# info gain is calculated by subtracting the uncertainty of parent node with the weighted uncertainty of the child nodes
def info_gain(left, right, uncertainty):
    data_points = len(left) + len(right)
    weight_left = len(left) / data_points
    weight_right = len(right) / data_points

    left_uncertainty = calc_gini_uncertainty(left)
    right_uncertainty = calc_gini_uncertainty(right)

    weighted_uncertainty = weight_left * left_uncertainty + weight_right * right_uncertainty
    return uncertainty - weighted_uncertainty


# function written by Robert
# must find the best way to split the data at a node by finding the best gain and best question
def best_split(data):
    # initialise variables
    best_gain = 0
    best_question = None

    cur_uncertainty = calc_gini_uncertainty(data)
    features = len(data[0]) - 1  # number of columns - 1 as last column contains labels

    # for each feature, get all values and find best question for best split
    for column in range(features):
        values = set([row[column] for row in data])

        # for each value, create a question and split data into true and false based on that question
        for val in values:
            q = Question(column, val)
            true_data = [row for row in data if q.check_answer(row)]
            false_data = [row for row in data if not q.check_answer(row)]

            # if either split is 0, data was not split on the question so skip
            if len(true_data) == 0 or len(false_data) == 0:
                continue

            information_gain = info_gain(false_data, true_data, cur_uncertainty)

            if information_gain > best_gain:
                best_gain = information_gain
                best_question = q

    return best_gain, best_question


# function written by Robert
# build a tree by recursively calling the function for each nodes branches until a leaf is reached
def build_tree(data):
    # get best split for the nodes data
    information_gain, q = best_split(data)

    # info gain is 0 then we have reached a leaf and we can return the leaf node
    if information_gain == 0:
        return Leaf(data)

    # get the data subsets based on the question
    true_data = [row for row in data if q.check_answer(row)]
    false_data = [row for row in data if not q.check_answer(row)]

    # call build tree method again to construct the nodes branches
    true_subtree = build_tree(true_data)
    false_subtree = build_tree(false_data)

    return Node(q, true_subtree, false_subtree)


# declare global variables to track the strings used to print the tree
diag_string = ""
i = 0


# function written by Caireann
def cart_tree(node, prev_node):
    tree_strings = []
    global diag_string
    global i

    # if has no previous node then the previous question is "" else get the previous nodes question
    if not prev_node:
        prev_question = ""
    else:
        prev_question = prev_node.question.print_question()

    # check if node is a leaf node
    if isinstance(node, Leaf):
        # i used to define the different predictions in the diagram
        i += 1
        # add predictions as a string to diag_string
        tree_strings.append(
            prev_question + " ->" + " Prediction_" + str(i) + "_" + str(list(node.probability.keys())[0]) + "_"
            + str(list(node.probability.values()))[1])
        for tree_string in tree_strings:
            diag_string += (tree_string + ";\n")
        return

    if prev_question != "":
        tree_strings.append(prev_question + " ->" + node.question.print_question())

    # recursively call the method for the current nodes branches
    cart_tree(node.true_subtree, node)
    cart_tree(node.false_subtree, node)

    # add the question for the current node to diag_string
    for tree_string in tree_strings:
        diag_string += (tree_string + ";\n")

    return diag_string


# function written by Caireann
# draw the image of the tree and save it to a png file
def create_tree_image(string):
    img = draw_diagram("blockdiag {\n" + string + "\n}")
    img.save("mlAssignment2DecisionTree.png")


# function written by Caireann
# check what the prediction of the leaf is
def classify(node, test_row):
    # check if is a leaf
    if isinstance(node, Leaf):
        return node.probability

    # if not a leaf recursively call the method on the nodes branches until the leaf is reached
    if node.question.check_answer(test_row):
        return classify(node.true_subtree, test_row)
    else:
        return classify(node.false_subtree, test_row)


# function written by Caireann
# find the accuracy by testing on test data set
def test_tree(node, test_data):
    # initialize variables
    correct = 0
    incorrect = 0

    if os.path.exists("Results.txt"):
        os.remove("Results.txt")

    # create a file to write the results of the two algorithms to
    f = open("Results.txt", "w+")

    # for each row of the test data
    for row in test_data:
        # get the predicted class of the node
        predicted_class = classify(node, row)
        # if the predicted class equals the actual class
        if str(list(predicted_class.keys())[0]) == row[-1]:
            correct += 1
        else:
            incorrect += 1

        # Write predictions and actual Classes to file
        f.write("Predicted Class: %s, Actual Class: %s\n" % (list(predicted_class.keys())[0], row[-1]))

    f.close()

    # calculate accuracy of algorithm
    total = correct + incorrect
    accuracy = (correct / total) * 100

    print("%d out of %d were classified correctly for our decision tree" % (correct, total))
    return accuracy


main()
