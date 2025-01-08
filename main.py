import pandas as pd
import numpy as np
import re
import emoji
import contractions
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

np.random.seed(42)
random.seed(42)
def data():
    birth_year_df = pd.read_csv(r"birth_year.csv")
    birth_year_df['post'] = birth_year_df['post'].apply(contractions.fix)
    birth_year_df['post'] = birth_year_df['post'].apply(
        lambda x: ' '.join(x.split())
    )
    birth_year_df['post'] = birth_year_df['post'].apply(
        lambda x: re.sub(r"(https|http)?:\S*", "", x)
    )
    #Adding new columns
    birth_year_df['num_characters'] = birth_year_df['post'].apply(len)
    birth_year_df['num_special_symbols'] = birth_year_df['post'].apply(lambda x: len(re.findall(r'[!?#%$@&<>-_]', x)))
    birth_year_df['num_emojis'] = birth_year_df['post'].apply(
        lambda x: len([char for char in x if char in emoji.EMOJI_DATA]))
    birth_year_df['age'] = 2025 - birth_year_df['birth_year']
    birth_year_df['age_range'] = (birth_year_df['age'] // 10) * 10


    return birth_year_df

def ridge_normal(birth_year_df):
# Splitting the data
    X = birth_year_df['post']
    y = birth_year_df['birth_year']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline for TF-IDF + Ridge Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=55000)),
        ('regressor', Ridge())
    ])

# Train the model
    pipeline.fit(X_train, y_train)

# Evaluate the model
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

def ridge_stylometry(birth_year_df):
    # Define a pipeline for TF-IDF + Ridge Regression + additional stylometric features
    X = birth_year_df[['post', 'num_characters', 'num_special_symbols', 'num_emojis']]
    y = birth_year_df['birth_year']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(max_features=55000), 'post'),  # Process the 'post' text
                ('numerical', FunctionTransformer(), ['num_characters', 'num_special_symbols', 'num_emojis'])
            ]
        )),
        ('regressor', Ridge())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

def plotting_features(birth_year_df, feature):
    feature_mean = birth_year_df.groupby('age_range')[feature].mean()
    plt.figure(figsize=(10, 6))
    plt.bar(feature_mean.index.astype(str), feature_mean.values, color='skyblue')
    plt.title(f"Average {feature} by Age Range")
    plt.xlabel("Age Range")
    plt.ylabel(f"Average {feature}")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def main():
    #Run these first to create the cleaner dataset, NOT THE PREPROCESSED ONE
    #birth_year_df = data()
    #birth_year_df.to_csv("birth_year_cleaned.csv")
    birth_year_df = pd.read_csv("birth_year_cleaned.csv")
    birth_year_df.dropna(inplace=True)
    if (int(input("Choose model (1 for normal ridge, 2 for ridge with stylometric features: "))) == 1:
        ridge_normal(birth_year_df)
    else:
        ridge_stylometry(birth_year_df)
    #For plotting feature(num_characters, num_special_symbols, num_emojis
    #plotting_features(birth_year_df, feature)



main()