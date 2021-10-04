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

# Creating a bar chart
plt.bar(classifications, instances)
plt.xlabel('Classification')
plt.ylabel('Instances')
plt.title('BBC Distribution')
# plt.savefig("BBC-distribution.pdf")

# Loading the corpus
data = sklearn.datasets.load_files("BBC", encoding='latin1', load_content=True)

# Creates a dictionary of the vocabulary of the corpus
vectorizer = fe.text.CountVectorizer()
vectorizer.fit_transform(data.data)
vector = vectorizer.transform(data.data)
# ft = vectorizer.get_feature_names_out()  # Gets the feature names, output is an array[5]

# ! Proof that CountVectorizer() stores their words and instances properly !
# print(vectorizer.vocabulary_)  # Prints dictionary of the words and their count
# print(vector.toarray())  # Prints an array

doc_train, doc_test, target_train, target_test = me.train_test_split(vector.toarray(), data.target, test_size=0.2, train_size=0.8, random_state=None)

# ! Proof that the split is done properly !
print("Total number of document: ",
      len(os.listdir("BBC/business")) +
      len(os.listdir("BBC/entertainment")) +
      len(os.listdir("BBC/politics")) +
      len(os.listdir("BBC/sport")) +
      len(os.listdir("BBC/tech")))
print("The training set has: ", len(doc_train), "elements, it makes up", len(doc_train) / 2225 * 100,
      "% of the total set"
      "\nThe test set has: ", len(doc_test), "elements, it makes up", len(doc_test) / 2225 * 100, "% of the total set")

# Training a multinomial Naive Bayes Classifier on the training set
MultiNB = MultinomialNB()
MultiNB.fit(doc_train, target_train)

# Evaluating the MultiNB to the test set
predictions = MultiNB.predict(doc_test)
# print(predictions)  # This part sucks, makes me want to kill myself. What am I even doing?
# try print(str(predictions)) and dont kys :) - Ryan
# Writing to a file and overwriting the content
file = open("bbc-performance.txt", "w")
file.write("(a) ***  MultinomialNB default values, try 1 ***\n\n")

# Creating the confusion matrix
confusionMatrix = sklearn.metrics.confusion_matrix(target_test, predictions)
print(confusionMatrix)

file.write("(b)\n")
file.write("\tConfusion matrix:\n")

# Temporarily setting the system output to the file in order to copy the matrix format to the file
np.set_printoptions(threshold=sys.maxsize)
temp = sys.stdout
sys.stdout = file  # Makes the print() function write to the file with the corresponding format
print(confusionMatrix)

# Obtaining precision, recall and f1-measure of the predictions
precRecF1 = met.classification_report(target_test, predictions)
print("\n(c)\n")
print("\tPrecision, Recall, and F1-measure: ")
print(precRecF1)

# Accuracy, macro-average F1 and weighted-average F1 of the predictions
accuracy = met.accuracy_score(target_test, predictions)
macroF1 = met.f1_score(target_test, predictions, average='macro')
weightedF1 = met.f1_score(target_test, predictions, average='weighted')
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

for e in classes:
    print("\t" + e + " class prior probability : " + str(1/len(classes)))


print("\n(f)\n")
print("\tVocabulary Size: ")

print("\t"+str(len(vectorizer.get_feature_names_out())))


print("\n(g)\n")
print("\t# Word Tokens in each class: ")

# for e in os.listdir("BBC"):
#     vectorizer.get_feature_names_out()

sys.stdout = temp  # Putting the system output back to normal


print(precRecF1)
print(accuracy)
print(macroF1)
print(weightedF1)



file.close()
