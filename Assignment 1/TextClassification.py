import matplotlib.pyplot as plt
import pandas as pd
import os

import sklearn.datasets
import sklearn.feature_extraction as fe
import sklearn.model_selection as me

classifications = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']
instances = [len(os.listdir("BBC/business")),
             len(os.listdir("BBC/entertainment")),
             len(os.listdir("BBC/politics")),
             len(os.listdir("BBC/sport")),
             len(os.listdir("BBC/tech"))]
plt.bar(classifications, instances)
plt.xlabel('Classification')
plt.ylabel('Instances')
plt.title('BBC Distribution')
plt.savefig("BBC-distribution.pdf")

#
data = sklearn.datasets.load_files("BBC", encoding='latin1', load_content=True)
vectorizer = fe.text.CountVectorizer()
vectorizer.fit(data.data)
print(vectorizer.vocabulary_)
# me.train_test_split(data, test_size=0.2, train_size=0.8, random_state=None)
