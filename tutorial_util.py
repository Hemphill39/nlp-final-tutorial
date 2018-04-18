
def load_seed_words(pos_file, neg_file):
    pos_words = []
    neg_words = []
    with open(pos_file, 'r') as f:
        for line in f:
            pos_words.append(line.replace('\n', ''))
    
    with open(neg_file, 'r') as f:
        for line in f:
            neg_words.append(line.replace('\n', ''))

    return pos_words, neg_words

# return a dictionary with new_words as a key and the seed words as the values
def load_relationships(relation_file):
    relationships = {}

    with open(relation_file, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            seed_word, new_word = line.split(' ')[0], line.split(' ')[1]
            if new_word not in relationships:
                relationships[new_word] = []                
            relationships[new_word].append(seed_word)

    return relationships
            
