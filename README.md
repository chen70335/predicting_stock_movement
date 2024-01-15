"""
Emma Truong, Hai Dang Nguyen, Nathan Chen
CSE 163 AE

Instructions on running stock_data_collection.py, machine_learning.py,
and graph.py
"""

# New library download required:
pip install yahoo_fin
pip install yahoo_fin --upgrade --no-cache-dir


# stock_data_collection.py
Running main.py returns three seperate csv datasets:
    1. stock_prediction_tech.csv
    2. stock_prediction_health.csv
    3. stock_prediction_fin.csv
-Possible Errors Encountered:
    In function get_stock_movement, we found that the yahoo_fin api
    method 'get_earnings_history' has limited time to access, and for
    every 7 tickers that uses the method, the function has to stop for
    around 2 minutes before continuing on to the next set of tickers.
    Therefore, the three main lists of tickers need to be broken down
    in order to create the dataset. However, the end result of the three
    datasets is also provided in the folder already.


# machine_learning.py
Running main.py currently returns the machine learning algorithm
accuracy of four different models performed on the combined dataset.
However, there are functions including logistic_reg_kbest(df),
logistic_reg_rfe(df), and machine_learning_models(df, normalization_type)
that the client can use to interpret different results.

-Four usable dataframes:
    1. df_tech
    2. df_health
    3. df_fin
    4. df_combined (including all three above)

-machine_learning_models(df, normalization_type):
    Takes in a dataset and a normalization_type (parameters include
    'None', 'Simple Normalization', and 'Standard Scaler'). Returns
    four models' accuracy (Decision Tree, Random Forest, Logistic Regression,
    Support Vector Machine) with their respective types of normalization

-logistic_reg_kbest(df):
    Takes in a dataset then return the accuracy of logistic regression
    before and after SelectKBest feature selection. In order to find the
    highest accuracy using SelectKBest, the number of K was manually
    implemented until it resulted in the highest accuracy.

-logistic_reg_rfe(df):
    Takes in a dataset then return the accuracy of logistic regression
    before and after recursive feature elimination feature selection.
 
 
# graph.py
<img width="335" alt="comparing_industry" src="https://github.com/chen70335/predicting_stock_movement/assets/101837218/545e7b43-8023-4662-9a12-943c5c6a669f">

Takes in 4 seperate datasets including technology, healthcare, financial,
and combined. Produces bar graphs that show the accuracy differences between
different models, different use of data normalization, and different use of
feature selection techniques. Also, produces bar graphs on ranked features for
each of the four datasets.

-comparing_models(df):
    Specifically only take in the df_combined dataset to produce a bar graph that shows
    differences in accuracy of four types of models, as well as differences in models that
    use no data normalization, simple normalization, or standard scaler. This function calls
    no_normalize(df), simple_normalize(df), and standard_scaler(df) in order to compute the
    results for the differences in use of data normalization.

-ranking_features_combined(df):
    Specifically only take in the df_combined dataset to produce a bar graph
    that produces visualization of the ranked features for the combined dataset
    using simple normalization, logistic regression, and SelectKBest.

-comparing_industry(df_list):
    Takes in a list of the four datasets then return a bar graph representing
    every single dataset's accuracy results using simple normalization and the
    differences in using no feature selection technique, recursive feature elimination,
    or SelectKBest. This function also calls fs_accuracy(df_list) which computes the
    accuracy for all 12 results (3 for each dataset * 4 total datasets)
