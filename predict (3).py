import sys
import pickle
import numpy as np
import os

# Load the pre-trained model from the .pkl file
with open('{}/Emojify-main/model.pkl'.format(os.getcwd()), 'rb') as f:
    model = pickle.load(f)

# Read the data from standard input and preprocess it
#text = sys.stdin.read()
text=input("Enter text")
   
# Use the model to predict the disease
predictions = model.predict([text],verbose=0)
emoj = np.argmax(predictions)

# Print the predicted disease index (e.g. 0 for 'healthy' or 1 for 'diseased')
print(emoj)
