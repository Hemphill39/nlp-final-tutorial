from tutorial_util import load_relationships, load_seed_words

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

#Import the following to classify the new seed words
from sklearn.naive_bayes import GaussianNB
import numpy as np

#Create a square matrix of size seedCt x seedCt to insert training samples
allSeeds = pos_words + neg_words
seedCt = len(allSeeds)
trainMatrix = np.zeros((seedCt, seedCt))
#Create vector of length seedCt to label training features
trainScores = np.zeros(seedCt)

#Populate train matrix with positive seed words
for rowA, word in enumerate(pos_words):
    if word in distance_matrix:
        columns = distance_matrix[word]
        for seedWord in columns.keys():
            columnNum = allSeeds.index(seedWord)
            trainMatrix[rowA][columnNum] = columns[seedWord]

#Populate train matrix with negative seed words
rowA += 1
for rowB, word in enumerate(neg_words):
    if word in distance_matrix:
        columns = distance_matrix[word]
        for seedWord in columns.keys():
            columnNum = allSeeds.index(seedWord)
            trainMatrix[rowA+rowB][columnNum] = columns[seedWord]
    trainScores[rowA+rowB] = 1

#Ensure every seed word correlates to itself
np.fill_diagonal(trainMatrix, 1)

#Train a naive bayes classifier on the seed words
clf = GaussianNB()
clf.fit(trainMatrix, trainScores)

#Parse through each new word
for word in distance_matrix.keys():
    if word not in allSeeds:
        row = np.zeros(seedCt)
        columns = distance_matrix[word]
        #Load in each seed word's features into a vector of length == seedCt
        for seedWord in columns.keys():
            columnNum = allSeeds.index(seedWord)
            row[columnNum] = columns[seedWord]
        row = [list(row)]
        #Classify the new word and determine if it is positive, negative or neutral based on its score
        scores = clf.predict_proba(row)[0]
        diff = scores[0] - scores[1]
        threshold = 0.1
        if diff > threshold:
            pos_words.append(word)
        elif diff < (threshold*(-1)):
            neg_words.append(word)

#Print out the new and old seeds
with open("new_pos_words.txt",'w') as fout:
    for word in pos_words:
        fout.write(str(word) + "\n")

with open("new_neg_words.txt",'w') as fout:
    for word in neg_words:
        fout.write(str(word) + "\n")

print('done!')
