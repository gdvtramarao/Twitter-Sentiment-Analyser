# Twitter Sentiment Analysis

This project uses Machine Learning (ML) to perform **sentiment analysis** on Twitter posts (tweets). The goal is to classify a given tweet as **Positive**, **Negative**, **Neutral**, or **Irrelevant**. The machine learning model is trained using **Logistic Regression** on preprocessed data, and the prediction is performed through a simple **Streamlit** web application.

## Key Features

- **Data Preprocessing:** Cleaned and vectorized Twitter data using TF-IDF vectorization.
- **Sentiment Classification:** The sentiment of a tweet is classified into one of four categories: Positive, Negative, Neutral, or Irrelevant.
- **Interactive Web App:** Built with **Streamlit** to allow users to input text and get the sentiment prediction along with confidence scores.
- **Machine Learning Model:** **Logistic Regression** classifier for sentiment prediction, trained on labeled Twitter data.

# 🚀 Twitter Sentiment Analyzer with Live Demo  

[![Open in Streamlit](https://img.shields.io/badge/🔗%20Live%20Demo-Streamlit-red?logo=streamlit&style=for-the-badge)](https://gdvtramarao-sentiment-analyzer.streamlit.app/)

🔍 **Real-time Twitter sentiment analysis web app** built with **Logistic Regression** and **TF-IDF**.  
Classifies text into: **Positive 🟢, Negative 🔴, Neutral 🟡, or Irrelevant ⚪**.  

---

## ✨ Features  
- ⚡ Instant sentiment predictions on tweets & text  
- 📊 ML pipeline with TF-IDF + Logistic Regression  
- 🌐 Deployed on **Streamlit Cloud** for easy access  
- 🎨 Simple and clean UI  

---

## 📸 Demo Screenshot
<img width="1161" height="810" alt="image" src="https://github.com/user-attachments/assets/715b2aec-079c-41e2-a4c2-a4318b115ba7" />



## Technologies Used
- **Python**
- **Streamlit**: For building the web application.
- **Scikit-learn**: For machine learning, model building, and prediction.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical operations.
- **Pickle**: For model serialization and saving.

## Steps to Run the Project

### 1. Clone the Repository

Clone the project repository to your local machine using the following command:

```bash
git clone https://github.com/gdvtramarao/Twitter-Sentiment-Analyser.git
cd Twitter-Sentiment-Analyser
```


### Create a virtual environment and activate it

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. Install Required Dependencies
```bash
Install all required Python dependencies using the requirements.txt file.

pip install -r requirements.txt
```


### 4. Run Data Preprocessing and Model Training
### 4.1 Preprocessing Data

```bash


Run the features.py script to preprocess the training and validation datasets. This script will clean the data, apply TF-IDF vectorization, and save the processed data.

python src/features.py


This will:

Load the datasets (twitter_training.csv and twitter_validation.csv).

Apply text preprocessing (remove NaN values, apply TF-IDF).

Save the transformed datasets (X_train_tfidf.pkl, X_val_tfidf.pkl) and label encoder (label_encoder.pkl) in the outputs/ directory.
```

### 4.2 Train the Logistic Regression Model

```bash
Next, run the train_logreg.py script to train the Logistic Regression model.

python src/train_logreg.py


This will:

Load the processed training data (X_train_tfidf.pkl).

Train the Logistic Regression model on the processed data.

Save the trained model (logreg_model.pkl) and vectorizer (vectorizer.pkl) in the outputs/ directory.

```

### 5. Run the Streamlit Web Application

```bash

After the model is trained, run the Streamlit app to start the web interface for sentiment analysis.

streamlit run app.py


This will:

Launch the app in your browser (typically available at http://localhost:8501).

Allow you to enter text, and it will display the sentiment prediction (Positive, Negative, Neutral, or Irrelevant) along with confidence scores.
```

### 6. Test the Model

```bash
In the Streamlit app, you can test various phrases. You can also test the app with the provided example texts or type your own text. The app will show the predicted sentiment and the confidence score for each class.
```
