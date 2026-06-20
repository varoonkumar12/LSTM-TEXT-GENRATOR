import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set up page config
st.set_page_config(page_title="LSTM Text Generator", layout="centered")
st.title("📜 LSTM Text Generation Interface")
st.write("Enter a seed phrase and let the trained LSTM network continue the text.")

# Load saved artifacts safely
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model("lstm_text_generator.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open("config.pkl", "rb") as f:
            config = pickle.load(f)
        return model, tokenizer, config["max_sequence_len"]
    except FileNotFoundError:
        st.error("Model artifacts not found! Run 'train.py' first to generate the files.")
        return None, None, None

model, tokenizer, max_sequence_len = load_artifacts()

# Helper function for temperature sampling
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    # Prevent log(0) errors by adding epsilon
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if model is not None:
    # User Inputs Layout
    seed_text = st.text_input("Enter Seed Text:", "to be or not to be")
    
    col1, col2 = st.columns(2)
    with col1:
        next_words = st.slider("Words to Generate", min_value=5, max_value=50, value=20)
    with col2:
        temperature = st.slider("Temperature (Creativity)", min_value=0.1, max_value=1.5, value=0.7, step=0.1)

    if st.button("Generate Text"):
        generated_text = seed_text
        output_placeholder = st.empty()
        
        with st.spinner("Generating sequences..."):
            for _ in range(next_words):
                # Tokenize and pad the seed phrase
                token_list = tokenizer.texts_to_sequences([generated_text])[0]
                token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
                
                # Predict probabilities
                predicted_probs = model.predict(token_list, verbose=0)[0]
                
                # Apply temperature adjustment
                output_word_index = sample_with_temperature(predicted_probs, temperature)
                
                # Decode index to word
                output_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == output_word_index:
                        output_word = word
                        break
                
                if not output_word:
                    break
                    
                generated_text += " " + output_word
                # Real-time update in Streamlit app
                output_placeholder.markdown(f"**Generated Output:**\n> {generated_text}")
                
        st.success("Generation Complete!")