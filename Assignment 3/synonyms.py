import gensim.downloader as api
import csv
import matplotlib.pyplot as plt


def synonymFinder(modelName='word2vec-google-news-300'):
    model = api.load(modelName)

    with open("synonyms.csv", newline='') as synonymsCSV:
        reader = csv.reader(synonymsCSV, delimiter=',')
        next(reader)
        correctCount = 0
        wrongCount = 0

        detailsCSV = open(f'{modelName}-details.csv', "w")
        for line in reader:

            label = ""

            questionWord = line[0]
            correctWord = line[1]
            answerWords = []
            answerWords.append(line[2])
            answerWords.append(line[3])
            answerWords.append(line[4])
            answerWords.append(line[5])

            query = []
            try:
                model[questionWord]
            except KeyError:
                print(f'The word {questionWord} does not appear in this model')
                label = "guess"

            for i in range(4):
                try:
                    model[answerWords[i]]
                    query.append((questionWord, answerWords[i]))
                except KeyError:
                    print(f'The word {answerWords[i]} does not appear in this model')
                    pass

            if not query:
                label = "guess"

            maxCosine = 0
            bestSyn = "None"
            if label != "guess":

                for w1, w2 in query:
                    currentCosine = model.similarity(w1, w2)
                    if maxCosine < currentCosine:
                        maxCosine = currentCosine
                        bestSyn = w2

                if bestSyn == correctWord:
                    label = "correct"
                    correctCount = correctCount + 1
                else:
                    label = "wrong"
                    wrongCount = wrongCount + 1
            # task 1 part 1. details csv
            detailsCSV.write(f'{questionWord},{correctWord},{bestSyn},{label}\n')
        detailsCSV.close()

        # task 1 part 2. analysis csv
        analysisCSV = open("analysis.csv", "a")
        if wrongCount == 0:
            wrongCount = 1
        analysisCSV.write(f'{modelName},{len(model)},{correctCount},{correctCount + wrongCount},'
                          f'{correctCount / float(correctCount + wrongCount)}\n')
        analysisCSV.close()

    # closing the synonym csv file
    synonymsCSV.close()


def graphMaker():
    with open("analysis.csv", newline='') as analysisFile:
        reader = csv.reader(analysisFile, delimiter=',')
        xaxisLabel = []
        accuracy = []

        for line in reader:
            xaxisLabel.append(line[0])
            accuracy.append(round(float(line[4]) * 100, 2))

    analysisFile.close()

    plt.bar(xaxisLabel, accuracy)
    plt.xticks(fontsize=9, rotation=7.5, wrap=True)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Distribution')
    plt.savefig("accuracy-distribution.pdf")

    # Needs to be compared to a random baseline(?)
    # Needs to be compared to a human gold-standard -> we need to wait for this


if __name__ == '__main__':
    # part 1
    # synonymFinder('word2vec-google-news-300')
    # part 2    1:
    # synonymFinder('glove-twitter-50')
    # synonymFinder('glove-wiki-gigaword-50')
    # # part 2    2:
    # synonymFinder('glove-wiki-gigaword-100')
    # synonymFinder('glove-wiki-gigaword-300')
    graphMaker()
