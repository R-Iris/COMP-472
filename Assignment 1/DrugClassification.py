import pandas as pd
import matplotlib.pyplot as plt
import sklearn.naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import sys
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score as acc_s
from sklearn.metrics import f1_score as f1

# Reading the csv file and printing it to the console
document = pd.read_csv("drug200.csv")
# print(document)

# Plotting the distribution of the instances in each class and storing the graph in a file
plt.hist(document.Drug, bins=range(6), edgecolor="black", align="left")
plt.ylabel("Number of drugs")
plt.xlabel("Categories of drugs")
plt.title("Distribution of drugs")
plt.savefig("drug-distribution.pdf")

# Ordinal features will be categorized as followed: 0 = LOW, 1 = NORMAL, 2 = HIGH
# Nominal features will be categorized as followed: 0 = FEMALE, 1 = MALE
document.BP = pd.Categorical(document.BP, ["LOW", "NORMAL", "HIGH"], ordered=True)
document.BP = document.BP.cat.codes

document.Cholesterol = pd.Categorical(document.Cholesterol, ["LOW", "NORMAL", "HIGH"], ordered=True)
document.Cholesterol = document.Cholesterol.cat.codes

document.Sex = pd.Categorical(document.Sex, ["F", "M"])
document.Sex = document.Sex.cat.codes

# NP Array of features (excluding class "Drug")
features = np.array(document[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]])


# The default split size in the sklearn model is 75% training and 25% testing data split
# doc_train, doc_test, target_train, target_test = train_test_split(allDocuments, document.Drug)
doc_train, doc_test, target_train, target_test = train_test_split(features, document.Drug)

file = open("drugs-performance.txt", "w")
# Temporarily setting the system output to the file in order to copy the matrix format to the file
np.set_printoptions(threshold=sys.maxsize)
temp = sys.stdout
sys.stdout = file  # Makes the print() function write to the file with the corresponding format

# Gaussian Naive Bayes Classifier
# Accepts an array-like of prior probabilities of classes
GaussianNB = sklearn.naive_bayes.GaussianNB()
GaussianNB.fit(doc_train, target_train)
gaussian_predictions = GaussianNB.predict(doc_test)
print("(a) *** Gaussian Naive Bayes: Default Parameters ***\n\n")
print("Confusion Matrix:")
print(cm(target_test, gaussian_predictions))
print("\nPrecision, Recall, and F1-measure:")
print(cr(target_test, gaussian_predictions))
print("\nAccuracy:")
print(acc_s(target_test, gaussian_predictions))
print("\nMacro F1 Score:")
print(f1(target_test, gaussian_predictions, average='macro'))
print("\nWeighted-Average F1 Score:")
print(f1(target_test, gaussian_predictions, average='weighted'))

baseDT = DecisionTreeClassifier()
baseDT.fit(doc_train, target_train)
baseDT_predictions = baseDT.predict(doc_test)
print("\n\n(b) *** Base Decision Tree Classifier: Default Parameters ***\n\n")
print("Confusion Matrix:")
print(cm(target_test, baseDT_predictions))
print("\nPrecision, Recall, and F1-measure:")
print(cr(target_test, baseDT_predictions))
print("\nAccuracy:")
print(acc_s(target_test, baseDT_predictions))
print("\nMacro F1 Score:")
print(f1(target_test, baseDT_predictions, average='macro'))
print("\nWeighted-Average F1 Score:")
print(f1(target_test, baseDT_predictions, average='weighted'))

topDT = DecisionTreeClassifier()
parametersDT = [{'criterion': ['gini', 'entropy'],
                 'max_depth': [1, 2],
                 'min_samples_split': [2, 3, 4]}]
GSCV = sklearn.model_selection.GridSearchCV(topDT, parametersDT, scoring='f1_weighted')
GSCV.fit(doc_train, target_train)
topDT_predictions = GSCV.predict(doc_test)
print("\n\n(c) *** Top Decision Tree Classifier: using Grid Search CV ***\n")
print("\nParameters to choose from:")
print("\ncriterion: gini or entropy")
print("max depth : 1, 2")
print("min samples split: 2, 3, 4")
print("\nBest Parameters:")
print(GSCV.best_params_)
print("\nConfusion Matrix:")
print(cm(target_test, topDT_predictions))
print("\nPrecision, Recall, and F1-measure:")
print(cr(target_test, topDT_predictions))
print("\nAccuracy:")
print(acc_s(target_test, topDT_predictions))
print("\nMacro F1 Score:")
print(f1(target_test, topDT_predictions, average='macro'))
print("\nWeighted-Average F1 Score:")
print(f1(target_test, topDT_predictions, average='weighted'))


perceptron = sklearn.linear_model.Perceptron()
perceptron.fit(doc_train, target_train)
perceptron_predictions = perceptron.predict(doc_test)
print("\n\n(d) *** Perceptron: Default Parameters ***\n\n")
print("Confusion Matrix:")
print(cm(target_test, perceptron_predictions))
print("\nPrecision, Recall, and F1-measure:")
print(cr(target_test, perceptron_predictions))
print("\nAccuracy:")
print(acc_s(target_test, perceptron_predictions))
print("\nMacro F1 Score:")
print(f1(target_test, perceptron_predictions, average='macro'))
print("\nWeighted-Average F1 Score:")
print(f1(target_test, perceptron_predictions, average='weighted'))


baseMLP = MLPClassifier(hidden_layer_sizes=100, activation='logistic', solver='sgd')
baseMLP.fit(doc_train, target_train)
baseMLP_predictions = baseMLP.predict(doc_test)
print("\n\n(e) *** Base Multi-Layered Perceptron: 1 hidden layer of 100 neurons, ")
print("sigmoid/logistic as activation function, stochastic gradient descent ***\n\n")
print("Confusion Matrix:")
print(cm(target_test, baseMLP_predictions))
print("\nPrecision, Recall, and F1-measure:")
print(cr(target_test, baseMLP_predictions))
print("\nAccuracy:")
print(acc_s(target_test, baseMLP_predictions))
print("\nMacro F1 Score:")
print(f1(target_test, baseMLP_predictions, average='macro'))
print("\nWeighted-Average F1 Score:")
print(f1(target_test, baseMLP_predictions, average='weighted'))


topMLP = MLPClassifier()
parametersMLP = [{'activation': ['identity', 'logistic', 'tanh', 'relu'],
                  'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
                  'solver': ['adam', 'sgd']}]
GSCV_MLP = sklearn.model_selection.GridSearchCV(topMLP, parametersMLP)
GSCV_MLP.fit(doc_train, target_train)
GSCV_MLP_predictions = GSCV_MLP.predict(doc_test)
print("\n\n(f) *** Top Multi-Layered Perceptron: using Grid Search CV ***\n")
print("\nParameters to choose from:")
print("\nactivation: identity, logistic, tanh, relu")
print("hidden layer sizes : (30, 50) and (10, 10, 10)")
print("solver: adam and sgd")
print("\nBest Parameters:")
print(GSCV_MLP.best_params_)
print("\nConfusion Matrix:")
print(cm(target_test, GSCV_MLP_predictions))
print("\nPrecision, Recall, and F1-measure:")
print(cr(target_test, GSCV_MLP_predictions))
print("\nAccuracy:")
print(acc_s(target_test, GSCV_MLP_predictions))
print("\nMacro F1 Score:")
print(f1(target_test, GSCV_MLP_predictions, average='macro'))
print("\nWeighted-Average F1 Score:")
print(f1(target_test, GSCV_MLP_predictions, average='weighted'))

print("\n\n\n#############################################\n\n\n")

print("Step 8: running each model 10 times\n\n")

# 10 steps loop for each of the classifiers
# for each model and append the average accuracy, average macro-average F1, average  weighted-average  F1
# AND   standard  deviation  for  the  accuracy,  the  standard  deviation of  the  macro-average  F1,
# and  the  standard  deviation  of  the  weighted-average  F1

print("*** Gaussian NB ***")

avgAccGNB = np.empty(10)
avgMacroF1GNB = np.empty(10)
avgWF1GNB = np.empty(10)

for i in range(10):
    GaussianNB.fit(doc_train, target_train)
    gaussian_predictions = GaussianNB.predict(doc_test)
    avgAccGNB[i] = acc_s(target_test, gaussian_predictions)
    avgMacroF1GNB[i] = f1(target_test, gaussian_predictions, average='macro')
    avgWF1GNB[i] = f1(target_test, gaussian_predictions, average='weighted')

print("\nAverage Accuracy:")
print(np.average(avgAccGNB))
print("\nAverage Macro F1 Score:")
print(np.average(avgMacroF1GNB))
print("\nAverage Weighted-Average F1 Score:")
print(np.average(avgWF1GNB))

print("\nStandard Deviation Accuracy:")
print(np.std(avgAccGNB))
print("\nStandard Deviation Macro F1 Score:")
print(np.std(avgMacroF1GNB))
print("\nStandard Deviation Weighted-Average F1 Score:")
print(np.std(avgWF1GNB))


print("\n*** Base-DT ***")

avgAccDT = np.empty(10)
avgMacroF1DT = np.empty(10)
avgWF1DT = np.empty(10)

for i in range(10):
    baseDT.fit(doc_train, target_train)
    baseDT_predictions = baseDT.predict(doc_test)
    avgAccDT[i] = acc_s(target_test, baseDT_predictions)
    avgMacroF1DT[i] = f1(target_test, baseDT_predictions, average='macro')
    avgWF1DT[i] = f1(target_test, baseDT_predictions, average='weighted')

print("\nAverage Accuracy:")
print(np.average(avgAccDT))
print("\nAverage Macro F1 Score:")
print(np.average(avgMacroF1DT))
print("\nAverage Weighted-Average F1 Score:")
print(np.average(avgWF1DT))

print("\nStandard Deviation Accuracy:")
print(np.std(avgAccDT))
print("\nStandard Deviation Macro F1 Score:")
print(np.std(avgMacroF1DT))
print("\nStandard Deviation Weighted-Average F1 Score:")
print(np.std(avgWF1DT))


print("\n*** Top-DT ***")

avgAccTop_DT = np.empty(10)
avgMacroF1Top_DT = np.empty(10)
avgWF1Top_DT = np.empty(10)

for i in range(10):
    GSCV.fit(doc_train, target_train)
    topDT_predictions = GSCV.predict(doc_test)
    avgAccTop_DT[i] = acc_s(target_test, topDT_predictions)
    avgMacroF1Top_DT[i] = f1(target_test, topDT_predictions, average='macro')
    avgWF1Top_DT[i] = f1(target_test, topDT_predictions, average='weighted')

print("\nAverage Accuracy:")
print(np.average(avgAccTop_DT))
print("\nAverage Macro F1 Score:")
print(np.average(avgMacroF1Top_DT))
print("\nAverage Weighted-Average F1 Score:")
print(np.average(avgWF1Top_DT))

print("\nStandard Deviation Accuracy:")
print(np.std(avgAccTop_DT))
print("\nStandard Deviation Macro F1 Score:")
print(np.std(avgMacroF1Top_DT))
print("\nStandard Deviation Weighted-Average F1 Score:")
print(np.std(avgWF1Top_DT))


print("\n*** Perceptron ***")

avgAccPer = np.empty(10)
avgMacroF1Per = np.empty(10)
avgWF1Per = np.empty(10)

for i in range(10):
    perceptron.fit(doc_train, target_train)
    perceptron_predictions = perceptron.predict(doc_test)
    avgAccPer[i] = acc_s(target_test, perceptron_predictions)
    avgMacroF1Per[i] = f1(target_test, perceptron_predictions, average='macro')
    avgWF1Per[i] = f1(target_test, perceptron_predictions, average='weighted')

print("\nAverage Accuracy:")
print(np.average(avgAccPer))
print("\nAverage Macro F1 Score:")
print(np.average(avgMacroF1Per))
print("\nAverage Weighted-Average F1 Score:")
print(np.average(avgWF1Per))

print("\nStandard Deviation Accuracy:")
print(np.std(avgAccPer))
print("\nStandard Deviation Macro F1 Score:")
print(np.std(avgMacroF1Per))
print("\nStandard Deviation Weighted-Average F1 Score:")
print(np.std(avgWF1Per))


print("\n*** Base-MLP ***")

avgAccMLP = np.empty(10)
avgMacroF1MLP = np.empty(10)
avgWF1MLP = np.empty(10)

for i in range(10):
    baseMLP.fit(doc_train, target_train)
    baseMLP_predictions = baseMLP.predict(doc_test)
    avgAccMLP[i] = acc_s(target_test, baseMLP_predictions)
    avgMacroF1MLP[i] = f1(target_test, baseMLP_predictions, average='macro')
    avgWF1MLP[i] = f1(target_test, baseMLP_predictions, average='weighted')

print("\nAverage Accuracy:")
print(np.average(avgAccMLP))
print("\nAverage Macro F1 Score:")
print(np.average(avgMacroF1MLP))
print("\nAverage Weighted-Average F1 Score:")
print(np.average(avgWF1MLP))

print("\nStandard Deviation Accuracy:")
print(np.std(avgAccMLP))
print("\nStandard Deviation Macro F1 Score:")
print(np.std(avgMacroF1MLP))
print("\nStandard Deviation Weighted-Average F1 Score:")
print(np.std(avgWF1MLP))


print("\n*** Top-MLP ***")

avgAccTop_MLP = np.empty(10)
avgMacroF1Top_MLP = np.empty(10)
avgWF1Top_MLP = np.empty(10)

for i in range(10):
    GSCV_MLP.fit(doc_train, target_train)
    GSCV_MLP_predictions = GSCV_MLP.predict(doc_test)
    avgAccTop_MLP[i] = acc_s(target_test, GSCV_MLP_predictions)
    avgMacroF1Top_MLP[i] = f1(target_test, GSCV_MLP_predictions, average='macro')
    avgWF1Top_MLP[i] = f1(target_test, GSCV_MLP_predictions, average='weighted')

print("\nAverage Accuracy:")
print(np.average(avgAccTop_MLP))
print("\nAverage Macro F1 Score:")
print(np.average(avgMacroF1Top_MLP))
print("\nAverage Weighted-Average F1 Score:")
print(np.average(avgWF1Top_MLP))

print("\nStandard Deviation Accuracy:")
print(np.std(avgAccTop_MLP))
print("\nStandard Deviation Macro F1 Score:")
print(np.std(avgMacroF1Top_MLP))
print("\nStandard Deviation Weighted-Average F1 Score:")
print(np.std(avgWF1Top_MLP))

sys.stdout = temp  # Putting the system output back to normal

