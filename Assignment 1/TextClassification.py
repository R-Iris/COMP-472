import matplotlib.pyplot as plt
import os
import sklearn.datasets
import sklearn.feature_extraction as fe
import sklearn.model_selection as me
import sklearn.metrics as met
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import sys

# Array that contains the subfolder names
classifications = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']

# Array that contains the number of documents in each subfolder
instances = [len(os.listdir("BBC/business")),
             len(os.listdir("BBC/entertainment")),
             len(os.listdir("BBC/politics")),
             len(os.listdir("BBC/sport")),
             len(os.listdir("BBC/tech"))]

totalDoc = (len(os.listdir("BBC/business")) +
            len(os.listdir("BBC/entertainment")) +
            len(os.listdir("BBC/politics")) +
            len(os.listdir("BBC/sport")) +
            len(os.listdir("BBC/tech")))

# Creating a bar chart
plt.bar(classifications, instances)
plt.xlabel('Classification')
plt.ylabel('Instances')
plt.title('BBC Distribution')
# plt.savefig("BBC-distribution.pdf")

# Loading the corpus
data = sklearn.datasets.load_files("BBC", encoding='latin1', load_content=True)
# data is a bunch of 5 attributes, data, target, target_names, DESCR, filesnames
# The data attribute is a list of strings, with each string being the raw content of a file

#for i in range(5)
# sum(values if their target is i)

# Creates a dictionary of the vocabulary of the corpus
vectorizer = fe.text.CountVectorizer()
# Count vectorizer seperates the words in each file into their own strings
vector = vectorizer.fit_transform(data.data)
# vector = vectorizer.transform(data.data)
# Vector is the count of each word in the format of a 2d array
# ft = vectorizer.get_feature_names_out()  # Gets the feature names, output is an array[5]

# ! Proof that CountVectorizer() stores their words and instances properly !
# print(vectorizer.vocabulary_)  # Prints dictionary of the words and their count
# print(vector.toarray())  # Prints an array

doc_train, doc_test, target_train, target_test = me.train_test_split(vector.toarray(), data.target, test_size=0.2, train_size=0.8, random_state=None)
# train_test_split splits a list into 2, in this case, 2 arguments return 4 lists
# doc is the words and the count of their instances, target is their class (category, file directory)

# ! Proof that the split is done properly !
print("Total number of document: ",
      totalDoc)
print("The training set has: ", len(doc_train), "elements, it makes up", len(doc_train) / totalDoc * 100,
      "% of the total set"
      "\nThe test set has: ", len(doc_test), "elements, it makes up", len(doc_test) / totalDoc * 100, "% of the total set")

# Training a multinomial Naive Bayes Classifier on the training set
MultiNB = MultinomialNB()
MultiNB.fit(doc_train, target_train)

# Evaluating the MultiNB to the test set
predictions = MultiNB.predict(doc_test)
# predictions is a 2d array of the predicted target values
# print(str(predictions))

# Writing to a file and overwriting the content
file = open("bbc-performance.txt", "w")
file.write("(a) ***  MultinomialNB default values, try 1 ***\n\n")

# Creating the confusion matrix
confusionMatrix = sklearn.metrics.confusion_matrix(target_test, predictions)
# 2d array of the confusion matrix

file.write("(b)\n")
file.write("\tConfusion matrix:\n")

# Temporarily setting the system output to the file in order to copy the matrix format to the file
np.set_printoptions(threshold=sys.maxsize)
temp = sys.stdout
sys.stdout = file  # Makes the print() function write to the file with the corresponding format
print(confusionMatrix)

# Obtaining precision, recall and f1-measure of the predictions
precRecF1 = met.classification_report(target_test, predictions)
# the classification report returns a "report" or a dictionary (refer to online documentation)
print("\n(c)\n")
print("\tPrecision, Recall, and F1-measure: ")
print(precRecF1)

# Accuracy, macro-average F1 and weighted-average F1 of the predictions
accuracy = met.accuracy_score(target_test, predictions)
# returns a float

macroF1 = met.f1_score(target_test, predictions, average='macro')
# returns a float

weightedF1 = met.f1_score(target_test, predictions, average='weighted')
# returns a float

print("\n(d)\n")
print("\tAccuracy: ")
print("\t" + str(accuracy))
print("\tMacro-average F1: ")
print("\t" + str(macroF1))
print("\tWeighted-average F1: ")
print("\t" + str(weightedF1))

# Prior probabilities for each class
print("\n(e)\n")
print("\tPrior Probabilities: ")

ignore = "README.TXT"
classes = [e for e in os.listdir("BBC") if e not in ignore]

# Prior probability for each class
for i in range(5):
    print("\t" + classifications[i] + " class prior probability: " + str(instances[i]/totalDoc))

print("\n(f)\n")
print("\tVocabulary Size: ")

print("\t"+str(len(vectorizer.get_feature_names())))

print("\n(g)\n")
print("\t# Word Tokens in each class: ")

# ?
#businessContent =
#businessVectorizer = fe.text.CountVectorizer()
# businessVector = businessVectorizer.fit_transform(businessContent)
# Double for loop maybe?
# First for loop to iterate over each class, second for loop to iterate over each text file in a class

print("\n(h)\n")
print("\t# Word Tokens in the entire corpus: ")

totalWordTokens = sum(vectorizer.vocabulary_.values())
print("\t" + str(totalWordTokens))


print("\n(i)\n")
print("\t# and % of words with a frequency of zero in each class: ")

# ?

print("\n(j)\n")
print("\t# and % of words with a frequency of one in the entire corpus: ")

numWordsOneEntire = 0
for k in vectorizer.vocabulary_:
    if vectorizer.vocabulary_[k] == 1:
        numWordsOneEntire += 1
print("\tAmount: " + str(numWordsOneEntire))
print("\tPercentage: " + str(numWordsOneEntire / totalWordTokens * 100) + "%")

sys.stdout = temp  # Putting the system output back to normal


print(precRecF1)
print(accuracy)
print(macroF1)
print(weightedF1)


file.close()
