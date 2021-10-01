import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Reading the csv file and printing it to the console
document = pd.read_csv("drug200.csv")
print(document)

# Plotting the distribution of the instances in each class and storing the graph in a file
plt.hist(document.Drug, bins=range(6), edgecolor="black", align="left")
plt.ylabel("Number of drugs")
plt.xlabel("Categories of drugs")
plt.title("Distribution of drugs")
plt.savefig("drug-distribution.pdf")
# plt.show()

# Ordinal features will be categorized as followed: 0 = LOW, 1 = NORMAL, 2 = HIGH
# Nominal features will be categorized as followed: 0 = FEMALE, 1 = MALE
document.BP = pd.Categorical(document.BP, ["LOW", "NORMAL", "HIGH"], ordered=True)
document.BP = document.BP.cat.codes
print(document.BP)

document.Cholesterol = pd.Categorical(document.Cholesterol, ["LOW", "NORMAL", "HIGH"], ordered=True)
document.Cholesterol = document.Cholesterol.cat.codes
print(document.Cholesterol)

document.Sex = pd.Categorical(document.Sex, ["F", "M"])
document.Sex = document.Sex.cat.codes
print(document.Sex)

# The default split size in the sklearn model is 75% training and 25% testing data split
train_test_split(document.Age, document.Sex, document.BP, document.Cholesterol, document.Na_to_K, document.Drug)
