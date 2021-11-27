import gensim.downloader as api
print("Hi")
#googleW2VModel = api.load('word2vec-google-news-300')
#print(str(googleW2VModel))
#print(googleW2VModel["asdasd"])
# with open("synonyms.csv", newline='') as synonymsCSV:
#     reader = csv.reader(synonymsCSV, delimiter=',')
#     next(reader)
#     for line in reader:
#
#         label = ""
#
#         questionWord = line[0]
#
#         answerWords = []
#         answerWords[0] = line[2]
#         answerWords[1] = line[3]
#         answerWords[2] = line[4]
#         answerWords[3] = line[5]
#
#         query = []
#         for i in range(4):
#             if googleW2VModel[answerWords[i]]:
#                 query.append((questionWord, answerWords[i]))
#
#
#         try:
#             googleW2VModel[questionWord]
#
#         except KeyError:
#             label = "guess"
#
#         maxCosine = 0
#         bestSyn = ""
#
#         for w1, w2 in query:
#             try:
#                 wordNotPresent = googleW2VModel[w1]
#
#             except KeyError:
#                 print("The word(s) does not appear in this model")
#
#             currentCosine = googleW2VModel.similarity(w1, w2)
#             if maxCosine < currentCosine:
#                 maxCosine = currentCosine
#                 bestSyn = w2
#
#         print(w1 + " || " + bestSyn)
#
#         if bestSyn == line[1]:
#             print("Correct")
#         else:
#             print("Wrong")
#             #print('%r\t%r\t%.2f' % (w1, w2, googleW2VModel.similarity(w1, w2)))



#corpus = sim.docsim.MatrixSimilarity()

