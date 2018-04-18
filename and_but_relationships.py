from nltk.corpus import brown, movie_reviews
from nltk.text import TokenSearcher
from tutorial_util import load_seed_words
import nltk
# import nltk  nltk.download('brown') nltk.download('movie_reviews')

def write_word_pair(f, word1, word2):
    f.write(word1 + ' ' + word2 + '\n')

pos_words, neg_words = load_seed_words('pos_words.txt', 'neg_words.txt')

# resource to do regex
# http://www.nltk.org/book/ch03.html
reviews = nltk.Text(movie_reviews.words())
tok_search = TokenSearcher(reviews)
and_sequences = tok_search.findall('<\w*> <and> <\w*>')

# theres not that many but sequences
but_sequences = tok_search.findall('<\w*> <but> <\w*>')
# include sequences like ...'apple, but cat...' and 'apple but cat'
# but_occurences = tok_search.findall('<\w*><,>? <but> <.*>')

and_relationships = open('and_relationships.txt', 'w')
but_relationships = open('but_relationships.txt', 'w')

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
        elif word2 in pos_words:
            seed_word = word2
            new_word = word1
        else:
            print("neither word in positive set")

        write_word_pair(and_relationships, seed_word, new_word)
    elif word1 in neg_words or word2 in neg_words:
        # write to a positive file in format 'seed_word new_word'
        seed_word = ''
        new_word = ''
        if word1 in neg_words:
            seed_word = word1
            new_word = word2
        elif word2 in neg_words:
            seed_word = word2
            new_word = word1
        else:
            print("neither word in negative set")

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
        elif word2 in pos_words:
            seed_word = word2
            new_word = word1
        else:
            print("neither word in positive set")

        write_word_pair(but_relationships, seed_word, new_word)
    elif word1 in neg_words or word2 in neg_words:
        # write to a positive file in format 'seed_word new_word'
        seed_word = ''
        new_word = ''
        if word1 in neg_words:
            seed_word = word1
            new_word = word2
        elif word2 in neg_words:
            seed_word = word2
            new_word = word1
        else:
            print("neither word in negative set")

        write_word_pair(but_relationships, seed_word, new_word)


and_relationships.close()
but_relationships.close()


