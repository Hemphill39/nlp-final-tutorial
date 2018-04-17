from nltk.corpus import sentiwordnet as swn
import random

pos_synsets = []
neg_synsets = []

for synset in swn.all_senti_synsets():
    if synset.pos_score() > 0.70:
        pos_synsets.append(synset)

    elif synset.neg_score() > 0.70:
        neg_synsets.append(synset)

# reduce the negative word list to around 500
neg_synsets = [ neg_synsets[i] for i in sorted(random.sample(range(len(neg_synsets)), 536)) ]

print('num positive ' + str(len(pos_synsets)))
print('num negative ' + str(len(neg_synsets)))

for syn in pos_synsets[:10]:
    print(str(syn))

for syn in neg_synsets[:10]:
    print(str(syn))

pos_words = []
neg_words = []

for syn in pos_synsets:
    word = str(syn)[1:].split('.')[0]
    pos_words.append(word)

for syn in neg_synsets:
    word = str(syn)[1:].split('.')[0]
    neg_words.append(word)


pos_word_file = open('pos_words.txt', 'w')
neg_word_file = open('neg_words_file.txt', 'w')

for word in pos_words:
    pos_word_file.write("%s\n" % word)

for word in neg_words:
    neg_word_file.write("%s\n" % word)