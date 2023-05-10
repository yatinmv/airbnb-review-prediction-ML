
#Airbnb Review Prediction using ML

## Project Description

The goal of this project is to predict the various rating aspects of Airbnb listings using machine learning algorithms. The different rating aspects we consider are cleanliness, location, communication, value, check-in, accuracy and the overall rating. 

We perform data cleaning and exploratory data analysis to understand the dataset and prepare it for machine learning algorithms. We then use kNN and Random Forest machine learning models to make predictions.

## Data Description

The dataset used for this project is from Airbnb listings in Amsterdam. It includes information such as the listing's price, location, and reviews from past customers. The dataset is preprocessed to remove any missing values, outliers and duplicates.

## Methods Used

- Data Cleaning
- Exploratory Data Analysis
- Machine Learning Algorithms: kNN and Random Forest

## Results and Evaluation

The following table shows the different values of hyperparameters we obtained from our cross-validation:

| Predicted Value | Hyperparameters | k   | maxFeatures | nEstimators |
|----------------|----------------|-----|-------------|-------------|
| cleanliness    | 9              | 5   | 20          |             |
| location       | 9              | 5   | 50          |             |
| communication  | 9              | 5   | 20          |             |
| value          | 9              | 10  | 50          |             |
| check-in       | 7              | 1   | 20          |             |
| accuracy       | 9              | 1   | 50          |             |
| rating         | 7              | 10  | 10          |             |

The following table shows the mean squared error in predicting the various aspects for different machine learning models. For our baseline model, we choose a model which predicts the mean of the rating value. 

| Predicted Value         | KNN     | Random Forest | Baseline Model |
|------------------------|---------|---------------|----------------|
| cleanliness             | 0.1324  | 0.1301        | 0.1416         |
| location                | 0.0905  | 0.0765        | 0.0900         |
| communication           | 0.0828  | 0.0971        | 0.0889         |
| value                   | 0.1356  | 0.1361        | 0.1276         |
| check-in                | 0.1037  | 0.1031        | 0.0991         |
| accuracy                | 0.1281  | 0.1238        | 0.1298         |
| rating                  | 0.1245  | 0.1372        | 0.1356         |

The results obtained by our models are mixed. Out of the three models, for some aspects the baseline model has a better performance and for other aspects one of our models have a better performance. Out of the two models we have used, there is no clear winner. Sometimes the kNN model performs better and sometimes the Random Forest model does better.
