# SMS Spam Detection

This repository contains a machine learning project that focuses on detecting spam SMS messages using Natural Language Processing (NLP) techniques. The goal of the project is to classify SMS messages as either spam or ham (legitimate) based on their textual content.

The project demonstrates a complete machine learning workflow, starting from data preprocessing and exploratory data analysis to feature extraction, model training, and evaluation.

---

## Project Overview

Spam detection is a common real-world application of text classification. In this project, raw SMS text data is cleaned and transformed into a numerical format that machine learning algorithms can understand. Several classification models were tested, and the most suitable model was selected based on performance metrics.

The final implementation prioritizes simplicity, interpretability, and precision, making it suitable for academic purposes and learning.

---

## Dataset

- **File:** `spam.csv`  
- **Description:** Contains SMS messages labeled as spam or ham  
- **Target Column:** Indicates whether a message is spam (1) or ham (0)

---

## Methodology

The project follows these main steps:

1. Data loading and inspection  
2. Text preprocessing (lowercasing, tokenization, stopword removal, stemming)  
3. Feature extraction using TF-IDF  
4. Model training and evaluation  
5. Final model selection based on performance  

---

## Models Used

- Multinomial Naive Bayes  
- Logistic Regression  
- Support Vector Machine (SVM)

**Final Model:** Multinomial Naive Bayes

---

## Results

- Effective identification of spam and ham messages  
- High precision achieved by the final model  
- Visual analysis using word frequency plots and distributions  

---

## Technologies Used

- Python  
- Jupyter Notebook  
- pandas  
- numpy  
- NLTK  
- scikit-learn  
- Matplotlib  
- Seaborn
- Streamlit

---

## Dependencies and Requirements

- Python 3.8 or above  
- pandas  
- numpy  
- nltk  
- scikit-learn  
- matplotlib  
- seaborn  

Install the required dependencies using:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
