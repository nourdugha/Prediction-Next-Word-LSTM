import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model
model = load_model("next_word_lstm_model.h5")

# Load the tokenizer
with open("next_word_lstm_tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)


## Prdiction funciton

def predict_next_word(model, tokenizer, text, max_sequences_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequences_len:
        # token_list[-14:]  captures the end part of the list (including the last 14).
        token_list = token_list[-(max_sequences_len):]
    token_list = pad_sequences([token_list],maxlen=max_sequences_len,padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted,axis=1)
    for word , index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

# Streamlit app
st.title("Next word prediction with LSTM")
input_stream = st.text_input("Enter the sequence of words","to be or not to be")
if st.button("Predict"):
    max_sequences_len = model.input_shape[1]
    next_word = predict_next_word(model,tokenizer,input_stream,max_sequences_len)
    st.write(f"Next word is {next_word}")
