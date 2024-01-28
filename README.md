# Disease-Condition-Predictor
## About Dataset
> Patient reviews on specific drugs along with related conditions and a 10 star patient rating reflecting overall patient satisfaction. (This dataset was used for the Winter 2018 Kaggle University Club Hackathon)

## Problem Domain
> Predicting disease conditions from customer reviews using NLP and ML techniques and personalized healthcare recommendations

## Libraries used :
>
> Numpy
>
>   Pandas
>
>   Seaborn
>
>  Matplotlib
>
> scikit-Learn (classification and NLP Models)
>
>  NLTK

## Project Outline

* I divided the project into  six parts:

  - Data Cleaning
  - Exploratory Data Analysis
  - Text Preprocessing
  - Model Building (TF-IDF and Classification Models)
  - Visualizing Results
  - Model's Web App development
 
### 1. Data Cleaning
  - Filtered Relevant columns
  - Handled missing values and duplicated rows
  - Lower casing COlumn names
  - Checking data types and indexing

### 2. Exploratory Data Analysis
  > Visualized:

  - Most Common Drugs
  - Most Common Conditions
  - Rating Distribution and Distribution of Useful Count
  - Filtering Most Common Conditions
  - Average Ratings of Most Common Drugs
  - Average Useful Count of Most Common Drugs
  - Drugs with Highest Ratings and Useful Counts
  - Word Clouds for Different Conditions


 ### 3. Text Preprocessing
  - Filtering Relevant Conditions
  - Text Cleaning (lower casing)
  - Text Tokenization and Lemmatization
  - Building Corpus
  - Replacing Original Reviews with Cleaned Reviews


 ### 4. Model Building
  - 4.1 TF-IDF Model
  - 4.2 Classification Models


     * Logistic Regression
     * Support Vector Classifier
     * Random Forest Classifier
     * Decision Tree Classifie
   
 ### 5. Visualizing Result
  - Achieved an accuracy score of approximately 86.46% with Logistic Regression
  - Achieved an accuracy score of approximately 90.26% with SVC (Highest)
  - Achieved an accuracy score of approximately 86.68% with Random Forest Classifier
  - Achieved an accuracy score of approximately 83.21% with Descision Tree Classifier


### 6. Model Web App develpment:

* Used pickle to dump and load dataframe and SVC model to a new IDE

* Used Streamlit Library To Create a Web App which takes the input from user and displays the predicted Disease Condition for the given Input



