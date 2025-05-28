# 🎬 IMDB Movie Review Sentiment Analysis

This project uses a neural network to predict **sentiment (positive or negative)** from movie reviews using the **IMDB dataset**. It was built using **TensorFlow**, **Keras**, and **NLP preprocessing techniques**.

---

## 📌 Project Overview

- **Goal**: Classify movie reviews as Positive (1) or Negative (0)
- **Dataset**: IMDB reviews (50,000 examples)
- **Type**: Binary text classification
- **Frameworks**: TensorFlow, Keras
- **Model**: Embedding + GlobalAveragePooling1D + Dense layers

---

## 📊 Dataset Info

- 25,000 training reviews
- 25,000 testing reviews
- Balanced dataset: 50% positive, 50% negative
- Loaded via: `tensorflow_datasets.imdb_reviews`

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- Tokenizer & pad_sequences
- Matplotlib for visualization

---

## 🧠 Model Architecture

```
Embedding → GlobalAveragePooling1D → Dense (ReLU) → Dense (Sigmoid)
```

- **Embedding layer** to learn word vector representations
- **GlobalAveragePooling** to reduce dimensionality
- **Dense layers** to classify sentiment

---

## 🧪 Training Results

- **Training Accuracy**: ~90%
- **Validation Accuracy**: ~85%
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

---

## 📈 Output

- Plots of training/validation accuracy and loss
- Sentiment prediction function for custom reviews:
```python
predict_sentiment("This movie was absolutely fantastic!")
```

---

## 💾 How to Run

1. Clone the repo
2. Run the notebook in Google Colab or Jupyter
3. Train the model and test predictions
4. Save model using: `model.save("sentiment_model_imdb.h5")`

---

## ✨ Author

**Seona Ann Tom**  

[LinkedIn](https://www.linkedin.com/in/seona-ann-tom-06351332a)  
[GitHub](https://github.com/seonaann)

---

*This project is part of my machine learning journey. Feedback is welcome!*