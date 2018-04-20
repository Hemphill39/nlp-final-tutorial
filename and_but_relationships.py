from nltk.corpus import brown, movie_reviews
from nltk.text import TokenSearcher
from tutorial_util import load_seed_words
import nltk
# import nltk  nltk.download('brown') nltk.download('movie_reviews')

def write_word_pair(f, word1, word2):
    f.write(word1 + ' ' + word2 + '\n')

pos_words, neg_words = load_seed_words('pos_words.txt', 'neg_words.txt')
all_words = pos_words + neg_words

# resource to do regex
# http://www.nltk.org/book/ch03.html
texts = []
reviews = nltk.Text(movie_reviews.words())
brown_reviews = nltk.Text(brown.words())
texts.append(reviews)
texts.append(brown_reviews)


and_relationships = open('and_relationships.txt', 'w')
but_relationships = open('but_relationships.txt', 'w')

for text in texts:

    tok_search = TokenSearcher(text)
    and_sequences = tok_search.findall('<\w*> <and> <\w*>')

    # theres not that many but sequences
    but_sequences = tok_search.findall('<\w*> <but> <\w*>')
    # include sequences like ...'apple, but cat...' and 'apple but cat'
    # but_occurences = tok_search.findall('<\w*><,>? <but> <.*>')

    for and_seq in and_sequences:
        word1 = and_seq[0]
        word2 = and_seq[2]

        # check if its in the positive words
        if word1 in pos_words or word2 in pos_words:
            # write to a positive file in format 'seed_word new_word'
            seed_word = ''
            new_word = ''
            if word1 in pos_words:
                seed_word = word1
                new_word = word2
                write_word_pair(and_relationships, seed_word, new_word)
            if word2 in pos_words:
                seed_word = word2
                new_word = word1
                write_word_pair(and_relationships, seed_word, new_word)
        elif word1 in neg_words or word2 in neg_words:
            # write to a positive file in format 'seed_word new_word'
            seed_word = ''
            new_word = ''
            if word1 in neg_words:
                seed_word = word1
                new_word = word2
                write_word_pair(and_relationships, seed_word, new_word)
            if word2 in neg_words:
                seed_word = word2
                new_word = word1
                write_word_pair(and_relationships, seed_word, new_word)

    for but_seq in but_sequences:
        word1 = but_seq[0]
        word2 = but_seq[2]

        # check if its in the positive words
        if word1 in pos_words or word2 in pos_words:
            # write to a positive file in format 'seed_word new_word'
            seed_word = ''
            new_word = ''
            if word1 in pos_words:
                seed_word = word1
                new_word = word2
                write_word_pair(but_relationships, seed_word, new_word)
            elif word2 in pos_words:
                seed_word = word2
                new_word = word1
                write_word_pair(but_relationships, seed_word, new_word)

        elif word1 in neg_words or word2 in neg_words:
            # write to a positive file in format 'seed_word new_word'
            seed_word = ''
            new_word = ''
            if word1 in neg_words:
                seed_word = word1
                new_word = word2
                write_word_pair(but_relationships, seed_word, new_word)
            elif word2 in neg_words:
                seed_word = word2
                new_word = word1
                write_word_pair(but_relationships, seed_word, new_word)

and_relationships.close()
but_relationships.close()


