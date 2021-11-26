import gensim.downloader as api
import gensim.similarities as sim
import csv

googleW2VModel = api.load('word2vec-google-news-300')
with open("synonyms.csv", newline='') as synonymsCSV:
    reader = csv.reader(synonymsCSV, delimiter=',')
    next(reader)
    for line in reader:
        query = []
        maxCosine = 0
        bestSyn = ""
        questionWord = line[0]
        answerWords = line[2]

        query.append(questionWord,answerWords[0])
        query.append(questionWord,answerWords[0])
        query.append(questionWord,answerWords[0])
        query.append(questionWord,answerWords[0])
        


        couple1 = (line[0], line[2])
        query.append(couple1)
        couple2 = (line[0], line[3])
        query.append(couple2)
        couple3 = (line[0], line[4])
        query.append(couple3)
        couple4 = (line[0], line[5])
        query.append(couple4)

        for w1, w2 in query:
            try:
                wordNotPresent = googleW2VModel[w1]

            except KeyError:
                print("The word(s) does not appear in this model")

            currentCosine = googleW2VModel.similarity(w1, w2)
            if maxCosine < currentCosine:
                maxCosine = currentCosine
                bestSyn = w2

        print(w1 + " || " + bestSyn)

        if bestSyn == line[1]:
            print("Correct")
        else:
            print("Wrong")
            #print('%r\t%r\t%.2f' % (w1, w2, googleW2VModel.similarity(w1, w2)))



#corpus = sim.docsim.MatrixSimilarity()

