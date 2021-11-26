import gensim.downloader as api
import gensim.similarities as sim
import csv

googleW2VModel = api.load('word2vec-google-news-300')
with open("synonyms.csv", newline='') as synonymsCSV:
    reader = csv.reader(synonymsCSV, delimiter=',')
    next(reader)
    for line in reader:
        query = []
        couple1 = (line[0], line[2])
        query.append(couple1)
        couple2 = (line[0], line[3])
        query.append(couple2)
        couple3 = (line[0], line[4])
        query.append(couple3)
        couple4 = (line[0], line[5])
        query.append(couple4)
        # compare = sim.Similarity(query, googleW2VModel)



#corpus = sim.docsim.MatrixSimilarity()

