import sys
import numpy as np
from nltk.corpus import movie_reviews as mr
from sklearn.naive_bayes import MultinomialNB as mnb


def seedsToSet(inFile):
    wordSet = set()
    fin = open(inFile)
    for line in fin.readlines():
        line = line.strip().lower()
        if '_' not in line:
            wordSet.add(line)
    fin.close()
    return wordSet

def main():
    gSeedFin = sys.argv[1]
    bSeedFin = sys.argv[2]
    gSeeds = seedsToSet(gSeedFin)
    bSeeds = seedsToSet(bSeedFin)
    allSeeds = gSeeds.union(bSeeds)
    allSeedsList = list(allSeeds)
    
    trainText = np.zeros((1900, len(allSeeds)))
    trainScore = np.zeros(1900)
    for row, fileid in enumerate(mr.fileids('neg')[:900]):	
        words = set(mr.words(fileid))
        words = words.intersection(allSeeds)
        for word in words:
            column = allSeedsList.index(word)
            trainText[row][column] = 1
    for row, fileid in enumerate(mr.fileids('pos')[:900]):	
        words = set(mr.words(fileid))
        words = words.intersection(allSeeds)
        for word in words:
            column = allSeedsList.index(word)
            trainText[-(row+1)][column] = 1
        trainScore[-(row+1)] = 1

    testText = np.zeros((200, len(allSeeds)))
    testScore = np.zeros(200)
    for row, fileid in enumerate(mr.fileids('neg')[-100:]):	
        words = set(mr.words(fileid))
        words = words.intersection(allSeeds)
        for word in words:
            column = allSeedsList.index(word)
            testText[row][column] = 1
    for row, fileid in enumerate(mr.fileids('pos')[-100:]):	
        words = set(mr.words(fileid))
        words = words.intersection(allSeeds)
        for word in words:
            column = allSeedsList.index(word)
            testText[-(row+1)][column] = 1
        testScore[-(row+1)] = 1

    clf = mnb()
    clf.fit(trainText, trainScore)
    print("Accuracy: " + str(clf.score(testText, testScore)*100) + "%")

if __name__ == "__main__":
    main()