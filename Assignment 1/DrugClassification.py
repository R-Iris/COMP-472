import pandas as pd
import matplotlib.pyplot as plt
import sklearn.naive_bayes
from sklearn.model_selection import train_test_split
import numpy as np

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
# print(document.BP)

document.Cholesterol = pd.Categorical(document.Cholesterol, ["LOW", "NORMAL", "HIGH"], ordered=True)
document.Cholesterol = document.Cholesterol.cat.codes
# print(document.Cholesterol)

document.Sex = pd.Categorical(document.Sex, ["F", "M"])
document.Sex = document.Sex.cat.codes
# print(document.Sex)
# print(document.Drug)
# print(document.Age.array)
allDocuments = np.array(document)
# print(allDocuments)

# The default split size in the sklearn model is 75% training and 25% testing data split
# doc_train, doc_test, target_train, target_test = train_test_split(allDocuments, document.Drug)
# print(doc_train)
doc_train, doc_test, target_train, target_test = train_test_split(allDocuments, document.Drug)
# print(doc_train)

# Gaussian Naive Bayes Classifier
# Accepts an array-like of prior probabilities of classes
GaussianNB = sklearn.naive_bayes.GaussianNB()
