from tutorial_util import load_relationships, load_seed_words
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import numpy as np

pos_words, neg_words = load_seed_words('pos_words.txt', 'neg_words.txt')
and_relationships = load_relationships('and_relationships.txt')
but_relationships = load_relationships('but_relationships.txt')

distance_matrix = {}

for new_word in and_relationships:
    if new_word not in distance_matrix:
        distance_matrix[new_word] = {}

    for seed_word in and_relationships[new_word]:
        if seed_word not in distance_matrix[new_word]:
            distance_matrix[new_word][seed_word] = 0

        if seed_word in pos_words:
            distance_matrix[new_word][seed_word] += 1

        if seed_word in neg_words:
            distance_matrix[new_word][seed_word] -= 1
        
for new_word in but_relationships:
    if new_word not in distance_matrix:
        distance_matrix[new_word] = {}

    for seed_word in but_relationships[new_word]:
        if seed_word not in distance_matrix[new_word]:
            distance_matrix[new_word][seed_word] = 0

        if seed_word in pos_words:
            distance_matrix[new_word][seed_word] -= 1

        if seed_word in neg_words:
            distance_matrix[new_word][seed_word] += 1

allSeeds = pos_words + neg_words
seedCt = len(allSeeds)
trainMatrix = np.zeros((seedCt, seedCt))
trainScores = np.zeros(seedCt)
for rowA, word in enumerate(pos_words):
    if word in distance_matrix:
        columns = distance_matrix[word]
        for seedWord in columns.keys():
            columnNum = allSeeds.index(seedWord)
            trainMatrix[rowA][columnNum] = columns[seedWord]

rowA += 1
for rowB, word in enumerate(neg_words):
    if word in distance_matrix:
        columns = distance_matrix[word]
        for seedWord in columns.keys():
            columnNum = allSeeds.index(seedWord)
            trainMatrix[rowA+rowB][columnNum] = columns[seedWord]
    trainScores[rowA+rowB] = 1

np.fill_diagonal(trainMatrix, 1)

clf = GaussianNB()
clf.fit(trainMatrix, trainScores)
#print(clf.score(trainMatrix,trainScores))

for word in distance_matrix.keys():
    if word not in allSeeds:
        row = np.zeros(seedCt)
        columns = distance_matrix[word]
        for seedWord in columns.keys():
            columnNum = allSeeds.index(seedWord)
            row[columnNum] = columns[seedWord]
        row = [list(row)]
        scores = clf.predict_proba(row)[0]
        diff = scores[0] - scores[1]
        threshold = 0.1
        if diff > threshold:
            pos_words.append(word)
        elif diff < (threshold*(-1)):
            neg_words.append(word)


print('done!')
