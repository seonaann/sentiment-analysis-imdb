import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model
model = tf.keras.models.load_model("sentiment_model_imdb.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 250

def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded)[0][0]
    label = "Positive 😊" if prediction > 0.5 else "Negative 😠"
    return f"{label} ({prediction:.2f})"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter a movie review..."),
    outputs="text",
    title="🎬 IMDB Sentiment Analyzer",
    description="Enter a review to see if it's Positive or Negative."
)

demo.launch()
