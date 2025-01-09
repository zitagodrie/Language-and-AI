# Language-and-AI

## Setup of the Environment
We used Python 3.12 with the packages listed in the `requirements.txt` file. Use the following commands in your terminal to install the right packages in Python 3.12:

1. `python -m venv venv`
2. win: `.\venv\Scripts\activate` / macOS: `source venv/bin/activate`
3. `pip install -r requirements.txt`

## Organization / Steps
### 1. Model selection
In the folder model selection you can find the notebook with the different models we tried as an experiment (ridge regression, Random Forest, BERT). The accompanying results from the runs can be found in the excel file. This is a standalone file in which we picked the model

### 2. Preprocessing
After the model selection, we started with the pipeline for training the final model. This starts of with the preprocessing, which is done in `preprocessing.ipynb`. This notebook outputs 2 pickle files, which contains dataframes cleaned in different ways. These files are stored in the `preprocessed_data` directory. You should keep it in this location to load the data into the next notebook. 
This notebook outputs 2 dataframes with various levels of processing:
1. `df_original.pkl` - Basic preprocessing applied including tokenization, cleaning, stylometry feature engineering and the new feature 'age range'
2. `df_normalized.pkl` - In addtion to the basic preprocessing, this dataframe only contains 1 text per author and it is sampled to have equal amount of instances for each age range.

### 3. Training
- Setting up pipelines for different machine learning models, including Ridge Regression, Random Forest, and Multinomial Naive Bayes.
- Performing 5-fold cross-validation to evaluate model performance using Mean Absolute Error (MAE).
- Collecting predictions from all the models. (This is also stored in pickle files in the `predictions_data` directory. It is named after the dataframe that is used in the notebook)

### 4. Evaluation
- Conducting a quantitative analysis of the model predictions using the quantative_analysis function.
- Calculating accuracy, precision, recall, and F1-score for each age range.
- Performing a paired t-test to compare the performance of models on cleaned vs. uncleaned data.
- Plotting the actual and predicted age range distributions using count plots to visually compare the performance of the models.
- Displaying the plots side by side for easy comparison.