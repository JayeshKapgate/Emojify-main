import pickle
import numpy as np
import os
# Load the GloVe word embeddings from the text file
glove_file = open('{}/Emojify-main/glove.6B.50d.txt'.format(os.getcwd()), encoding='utf8')
embedding_matrix = {}
for line in glove_file:
    values = line.split()
    word = values[0]
    emb = np.array(values[1:], dtype='float')
    embedding_matrix[word] = emb

# Save the embedding matrix to a pickle file
with open('glove.6B.50d.pkl', 'wb') as f:
    pickle.dump(embedding_matrix, f)