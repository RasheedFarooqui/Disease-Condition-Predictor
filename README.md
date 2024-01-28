# Disease-Condition-Predictor
Data Cleaning:

Reads a dataset containing drug reviews and relevant information.
Removes irrelevant columns, handles missing values, and drops duplicate rows.
Converts column names to lowercase for consistency.
Exploratory Data Analysis (EDA):

Analyzes the distribution of drugs, conditions, ratings, and useful counts.
Visualizes the most common drugs, conditions, rating distributions, and useful count distributions.
Creates word clouds to visualize the most frequent words in reviews for specific conditions.
Text Preprocessing:

Cleans and preprocesses the review text by removing special characters, converting to lowercase, tokenizing, removing stopwords, and lemmatizing.
Model Building:

Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features.
Splits the dataset into training and testing sets.
Implements several classification models:
Logistic Regression
Support Vector Classifier (SVC)
Random Forest Classifier
Decision Tree Classifier
Evaluation:

Evaluates each model's performance using accuracy score and confusion matrix.
Visualizing Results:

Visualizes the performance of different models using a bar plot.
App Structure:

Saves the trained SVC model, preprocessed dataframe, and TF-IDF vectorizer using pickle for later use.
