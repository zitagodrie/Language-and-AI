# Language-and-AI

## Setup of the Environment
We used Python 3.12 with the packages listed in the `requirements.txt` file. Use the following commands in your terminal to install the right packages in Python 3.12:

1. `python -m venv venv`
2. win: `.\venv\Scripts\activate` / macOS: `source venv/bin/activate`
3. `pip install -r requirements.txt`

## Organization / Steps
### 1. Model selection
In the folder model selection you can find the notebook with the different models we tried (ridge regression, Random Forest and BERT). The accompanying results from the runs can be found in the excel file. This is a standalone file in which we picked the model

### 2. Preprocessing
After the model selection, we started with the pipeline for training the final model. This starts of with the preprocessing, which is done in `preprocessing.ipynb`. This notebook outputs a pickle file, which contains the cleaned dataframe. This file is stored in the main directory. You should keep it in this location to load the data into the next notebook.

### 3. Stylometry
...

### 4. Training and Quantative evaluation