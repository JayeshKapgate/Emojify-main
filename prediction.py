import numpy as np
import pickle
import os
import emoji
import streamlit as st

st.title("TEXT TO EMOJI")


# Load the trained model
with open('{}/model.pkl'.format(os.getcwd()), 'rb') as f:
    model = pickle.load(f)

# Load the emoji dictionary
emoji_dict = {
    0: ':grinning_face:',
    1: ':pensive_face:',
    2: ':angry_face:',
    3: ':anguished_face:'
}

# Load the GloVe word embeddings
with open('{}/glove.6B.50D.pkl'.format(os.getcwd()), 'rb') as f:
    embedding_matrix = pickle.load(f)

# Define a function to preprocess the input text
def preprocess_text(text):
    max_len = 10
    embedding_data = np.zeros((1, max_len + 2, 50))
    words = text.split()
    for i in range(min(len(words), max_len)):
        word = words[i].lower()
        if embedding_matrix.get(word) is not None:
            embedding_data[0][i+1] = embedding_matrix[word]
    return embedding_data

# Define a function to predict the emoji emotion
def predict_emoji(text):
    # Preprocess the input text
    X = preprocess_text(text)
    # Make the prediction using the trained model
    y_pred = model.predict(X,verbose=0)
    # Return the corresponding emoji emotion
    # return emoji_dict[np.argmax(y_pred)]
    return emoji.emojize(emoji_dict[np.argmax(y_pred)])

# Prompt the user to enter a text message
# text = input("Enter a text message: ")
# # Predict the emoji emotion and print it

# print(text,emoji)



with st.container():
   text = st.text_input( "Enter the text",)
#    if input:
#         st.write(text)
   result =st.button("Submit")
   emoj = predict_emoji(text)
   if result:
       st.write(text,emoj)