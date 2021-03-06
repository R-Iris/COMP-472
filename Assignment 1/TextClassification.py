import matplotlib.pyplot as plt
import os
import sklearn.datasets
import sklearn.feature_extraction as fe
import sklearn.model_selection as me
import sklearn.metrics as met
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import sys
import math

# the sum of all values in question 7 (g) should add up to the answer for 7 (h)
# FIX ----------------------------- FIX -------------------------- FIX --------


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
# print(vector.toarray())
# vector = vectorizer.transform(data.data)
# Vector is the count of each word in the format of a 2d array
# ft = vectorizer.get_feature_names_out()  # Gets the feature names, output is an array[5]

# ! Proof that CountVectorizer() stores their words and instances properly !
# print(vectorizer.vocabulary_)  # Prints dictionary of the words and their count
# print(vector.toarray())  # Prints an array

doc_train, doc_test, target_train, target_test = me.train_test_split(vector.toarray(), data.target, test_size=0.2, train_size=0.8, random_state=None)
# train_test_split splits a list into 2, in this case, 2 arguments return 4 lists
# doc is the words and the count of their instances, target is their class (category, file directory)

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

classificationsIT = 0
for i in MultiNB.feature_count_:
    occurences = 0
    for j in i:
        occurences += j
    print("\t" + str(classifications[classificationsIT] + ": " + str(occurences)))
    classificationsIT += 1

print("\n(h)\n")
print("\t# Word Tokens in the entire corpus: ")

# vectorNPArray has, for each class, keys being each word of the vocabulary, and values being the instance count for
# each of those words
vectorNPArray = np.array(vector.toarray())
totalWordTokens = np.sum(vectorNPArray)
print("\t" + str(totalWordTokens))

print("\n(i)\n")
print("\t# and % of words with a frequency of zero in each class: ")

classITI = 0
for i in MultiNB.feature_count_:
    zeroCount = 0
    for j in i:
        if j == 0:
            zeroCount += 1
    print("\t" + str(classifications[classITI] + ": " + str(zeroCount) + ", %: ") +
          str(zeroCount/len(vectorizer.get_feature_names())*100))
    classITI += 1

print("\n(j)\n")
print("\t# and % of words with a frequency of one in the entire corpus: ")

# Unique words in entire corpus over the vocabulary size
uniqueWords = 0
for k in vectorizer.vocabulary_:
    if vectorizer.vocabulary_[k] == 1:
        print("\t Unique Word: " + str(k))
        uniqueWords += 1
print("\t Amount: " + str(uniqueWords) + " Frequency: %" + str(uniqueWords/len(vectorizer.get_feature_names())*100))

print("\n(k)\n")
print("\tour 2 favorite words (that are present in the vocabulary) and their log-prob: ")
# P("goldeneye") = log(# of times goldeneye is seen/total # of tokens corpus)
# P("nintendo") = log(# of times goldeneye is seen/total # of tokens corpus)

print("\t Log-prob of goldeneye: " + str(math.log((vectorizer.vocabulary_.get('goldeneye'))/totalWordTokens)))
print("\t Log-prob of nintendo: " + str(math.log((vectorizer.vocabulary_.get('nintendo'))/totalWordTokens)))

#
#
#
# Try number 2 for the Multinomial NB, default values

file.write("\n\n(a) ***  MultinomialNB default values, try 2 ***\n\n")

MultiNB2 = MultinomialNB()
MultiNB2.fit(doc_train, target_train)

predictions2 = MultiNB2.predict(doc_test)

confusionMatrix2 = sklearn.metrics.confusion_matrix(target_test, predictions2)

file.write("(b)\n")
file.write("\tConfusion matrix:\n")
print(confusionMatrix2)

precRecF1_2 = met.classification_report(target_test, predictions2)
print("\n(c)\n")
print("\tPrecision, Recall, and F1-measure: ")
print(precRecF1_2)
accuracy2 = met.accuracy_score(target_test, predictions2)
macroF1_2 = met.f1_score(target_test, predictions2, average='macro')
weightedF1_2 = met.f1_score(target_test, predictions2, average='weighted')
print("\n(d)\n")
print("\tAccuracy: ")
print("\t" + str(accuracy2))
print("\tMacro-average F1: ")
print("\t" + str(macroF1_2))
print("\tWeighted-average F1: ")
print("\t" + str(weightedF1_2))

print("\n(e)\n")
print("\tPrior Probabilities: ")
classes2 = [e for e in os.listdir("BBC") if e not in ignore]
for i in range(5):
    print("\t" + classifications[i] + " class prior probability: " + str(instances[i]/totalDoc))


print("\n(f)\n")
print("\tVocabulary Size: ")
print("\t"+str(len(vectorizer.get_feature_names())))

print("\n(g)\n")
print("\t# Word Tokens in each class: ")
classificationsIT_2 = 0
for i in MultiNB2.feature_count_:
    occurences = 0
    for j in i:
        occurences += j
    print("\t" + str(classifications[classificationsIT_2] + ": " + str(occurences)))
    classificationsIT_2 += 1

print("\n(h)\n")
print("\t# Word Tokens in the entire corpus: ")
vectorNPArray2 = np.array(vector.toarray())
totalWordTokens2 = np.sum(vectorNPArray2)
print("\t" + str(totalWordTokens2))

print("\n(i)\n")
print("\t# and % of words with a frequency of zero in each class: ")
classITI_2 = 0
for i in MultiNB2.feature_count_:
    zeroCount = 0
    for j in i:
        if j == 0:
            zeroCount += 1
    print("\t" + str(classifications[classITI_2] + ": " + str(zeroCount) + ", %: ") +
          str(zeroCount/len(vectorizer.get_feature_names())*100))
    classITI_2 += 1

print("\n(j)\n")
print("\t# and % of words with a frequency of one in the entire corpus: ")
uniqueWords2 = 0
for k in vectorizer.vocabulary_:
    if vectorizer.vocabulary_[k] == 1:
        print("\t Unique Word: " + str(k))
        uniqueWords2 += 1
print("\t Amount: " + str(uniqueWords2) + " Frequency: %" + str(uniqueWords2/len(vectorizer.get_feature_names())*100))

print("\n(k)\n")
print("\tour 2 favorite words (that are present in the vocabulary) and their log-prob: ")
print("\t Log-prob of goldeneye: " + str(math.log((vectorizer.vocabulary_.get('goldeneye'))/totalWordTokens2)))
print("\t Log-prob of nintendo: " + str(math.log((vectorizer.vocabulary_.get('nintendo'))/totalWordTokens2)))

#
#
#
# Try number 3 for the Multinomial NB, smoothing value 0.0001

file.write("\n\n(a) ***  MultinomialNB smoothing value of 0.0001, try 3 ***\n\n")

MultiNB3 = MultinomialNB(alpha=0.0001)
MultiNB3.fit(doc_train, target_train)

predictions3 = MultiNB3.predict(doc_test)

confusionMatrix3 = sklearn.metrics.confusion_matrix(target_test, predictions3)

file.write("(b)\n")
file.write("\tConfusion matrix:\n")
print(confusionMatrix3)

precRecF1_3 = met.classification_report(target_test, predictions3)
print("\n(c)\n")
print("\tPrecision, Recall, and F1-measure: ")
print(precRecF1_3)
accuracy3 = met.accuracy_score(target_test, predictions3)
macroF1_3 = met.f1_score(target_test, predictions3, average='macro')
weightedF1_3 = met.f1_score(target_test, predictions3, average='weighted')
print("\n(d)\n")
print("\tAccuracy: ")
print("\t" + str(accuracy3))
print("\tMacro-average F1: ")
print("\t" + str(macroF1_3))
print("\tWeighted-average F1: ")
print("\t" + str(weightedF1_3))

print("\n(e)\n")
print("\tPrior Probabilities: ")
classes3 = [e for e in os.listdir("BBC") if e not in ignore]
for i in range(5):
    print("\t" + classifications[i] + " class prior probability: " + str(instances[i]/totalDoc))


print("\n(f)\n")
print("\tVocabulary Size: ")
print("\t"+str(len(vectorizer.get_feature_names())))

print("\n(g)\n")
print("\t# Word Tokens in each class: ")
classificationsIT_3 = 0
for i in MultiNB3.feature_count_:
    occurences = 0
    for j in i:
        occurences += j
    print("\t" + str(classifications[classificationsIT_3] + ": " + str(occurences)))
    classificationsIT_3 += 1

print("\n(h)\n")
print("\t# Word Tokens in the entire corpus: ")
vectorNPArray3 = np.array(vector.toarray())
totalWordTokens3 = np.sum(vectorNPArray3)
print("\t" + str(totalWordTokens3))

print("\n(i)\n")
print("\t# and % of words with a frequency of zero in each class: ")
classITI_3 = 0
for i in MultiNB3.feature_count_:
    zeroCount = 0
    for j in i:
        if j == 0:
            zeroCount += 1
    print("\t" + str(classifications[classITI_3] + ": " + str(zeroCount) + ", %: ") +
          str(zeroCount/len(vectorizer.get_feature_names())*100))
    classITI_3 += 1

print("\n(j)\n")
print("\t# and % of words with a frequency of one in the entire corpus: ")
uniqueWords3 = 0
for k in vectorizer.vocabulary_:
    if vectorizer.vocabulary_[k] == 1:
        print("\t Unique Word: " + str(k))
        uniqueWords3 += 1
print("\t Amount: " + str(uniqueWords3) + " Frequency: %" + str(uniqueWords3/len(vectorizer.get_feature_names())*100))

print("\n(k)\n")
print("\tour 2 favorite words (that are present in the vocabulary) and their log-prob: ")
print("\t Log-prob of goldeneye: " + str(math.log((vectorizer.vocabulary_.get('goldeneye'))/totalWordTokens3)))
print("\t Log-prob of nintendo: " + str(math.log((vectorizer.vocabulary_.get('nintendo'))/totalWordTokens3)))


#
#
#
# Try number 4 for the Multinomial NB, smoothing value 0.9

file.write("\n\n(a) ***  MultinomialNB smoothing value of 0.9, try 4 ***\n\n")

MultiNB4 = MultinomialNB(alpha=0.9)
MultiNB4.fit(doc_train, target_train)

predictions4 = MultiNB4.predict(doc_test)

confusionMatrix4 = sklearn.metrics.confusion_matrix(target_test, predictions4)

file.write("(b)\n")
file.write("\tConfusion matrix:\n")
print(confusionMatrix4)

precRecF1_4 = met.classification_report(target_test, predictions4)
print("\n(c)\n")
print("\tPrecision, Recall, and F1-measure: ")
print(precRecF1_4)
accuracy4 = met.accuracy_score(target_test, predictions4)
macroF1_4 = met.f1_score(target_test, predictions4, average='macro')
weightedF1_4 = met.f1_score(target_test, predictions4, average='weighted')
print("\n(d)\n")
print("\tAccuracy: ")
print("\t" + str(accuracy4))
print("\tMacro-average F1: ")
print("\t" + str(macroF1_4))
print("\tWeighted-average F1: ")
print("\t" + str(weightedF1_4))

print("\n(e)\n")
print("\tPrior Probabilities: ")
classes4 = [e for e in os.listdir("BBC") if e not in ignore]
for i in range(5):
    print("\t" + classifications[i] + " class prior probability: " + str(instances[i]/totalDoc))


print("\n(f)\n")
print("\tVocabulary Size: ")
print("\t"+str(len(vectorizer.get_feature_names())))

print("\n(g)\n")
print("\t# Word Tokens in each class: ")
classificationsIT_4 = 0
for i in MultiNB4.feature_count_:
    occurences = 0
    for j in i:
        occurences += j
    print("\t" + str(classifications[classificationsIT_4] + ": " + str(occurences)))
    classificationsIT_4 += 1

print("\n(h)\n")
print("\t# Word Tokens in the entire corpus: ")
vectorNPArray4 = np.array(vector.toarray())
totalWordTokens4 = np.sum(vectorNPArray4)
print("\t" + str(totalWordTokens4))

print("\n(i)\n")
print("\t# and % of words with a frequency of zero in each class: ")
classITI_4 = 0
for i in MultiNB4.feature_count_:
    zeroCount = 0
    for j in i:
        if j == 0:
            zeroCount += 1
    print("\t" + str(classifications[classITI_4] + ": " + str(zeroCount) + ", %: ") +
          str(zeroCount/len(vectorizer.get_feature_names())*100))
    classITI_4 += 1

print("\n(j)\n")
print("\t# and % of words with a frequency of one in the entire corpus: ")
uniqueWords4 = 0
for k in vectorizer.vocabulary_:
    if vectorizer.vocabulary_[k] == 1:
        print("\t Unique Word: " + str(k))
        uniqueWords4 += 1
print("\t Amount: " + str(uniqueWords4) + " Frequency: %" + str(uniqueWords4/len(vectorizer.get_feature_names())*100))

print("\n(k)\n")
print("\tour 2 favorite words (that are present in the vocabulary) and their log-prob: ")
print("\t Log-prob of goldeneye: " + str(math.log((vectorizer.vocabulary_.get('goldeneye'))/totalWordTokens4)))
print("\t Log-prob of nintendo: " + str(math.log((vectorizer.vocabulary_.get('nintendo'))/totalWordTokens4)))

sys.stdout = temp  # Putting the system output back to normal

file.close()
