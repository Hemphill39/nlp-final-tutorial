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

print('done!')
