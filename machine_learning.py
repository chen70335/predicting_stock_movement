"""
Emma Truong, Hai Dang Nguyen, Nathan Chen
CSE 163 AE

This file implements the functions necessary to create the machine learning
models for research questions 1-2. Checked against flake8 errors.
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, mean_squared_error


def main():
    df_fin = pd.read_csv('stock_prediction_financial.csv')
    df_fin = df_fin.drop(columns=['Unnamed: 0.1', 'Unnamed: 0.2',
                                  'Unnamed: 0'], axis=1)

    df_health = pd.read_csv('stock_prediction_healthcare.csv')
    df_health = df_health.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'],
                               axis=1)

    df_tech = pd.read_csv('stock_prediction_tech.csv')
    df_tech = df_tech.dropna()
    df_tech = df_tech.drop(columns=['Unnamed: 0.1', 'Unnamed: 0.2',
                                    'Unnamed: 0', 'Unnamed: 0.1.1',
                                    'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1',
                                    'Unnamed: 0.1.1.1.1.1'],
                           axis=1)

    df_combined = pd.concat([df_fin, df_health, df_tech])
    machine_learning_models(df_combined, 'Standard Scaler')


def machine_learning_models(df, normalization_type):
    """
    Takes in a dataset and a normalization_type (parameters include
    'None', 'Simple Normalization', and 'Standard Scaler'). Returns
    four models' accuracy (Decision Tree, Random Forest, Logistic Regression,
    Support Vector Machine) with their respective types of normalization.
    """
    # Data/feature normalization
    if normalization_type == 'Simple Normalization':
        df = normalize_column(df)
        features = df.drop('Stock Movement', axis=1)
        labels = df['Stock Movement']
        features_train, features_test, labels_train, labels_test =\
            train_test_split(features, labels, test_size=0.3, random_state=2)
    elif normalization_type == 'Standard Scaler':
        df = df.drop(columns=['Release_Date', 'Ticker'])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        sc = StandardScaler()
        features = df.drop('Stock Movement', axis=1)
        labels = df['Stock Movement']
        features_train, features_test, labels_train, labels_test =\
            train_test_split(features, labels, test_size=0.3, random_state=2)
        features_train = sc.fit_transform(features_train)
        features_train = pd.DataFrame(data=features_train,
                                      columns=features.columns)
        features_test = sc.transform(features_test)
        features_test = pd.DataFrame(data=features_test,
                                     columns=features.columns)
    else:
        df = df.drop(columns=['Release_Date', 'Ticker'])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        features = df.drop('Stock Movement', axis=1)
        labels = df['Stock Movement']
        features_train, features_test, labels_train, labels_test =\
            train_test_split(features, labels, test_size=0.3, random_state=2)
    # ML: DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    predict = model.predict(features_test)
    print('DecisionTree Accuracy: ', accuracy_score(labels_test, predict))
    # ML: RandomForestClassifier
    model = RandomForestClassifier(n_estimators=25)
    model.fit(features_train, labels_train)
    predict_2 = model.predict(features_test)
    print('RandomForest Accuracy: ', accuracy_score(labels_test, predict_2))
    # ML: Support Vector Model
    model = svm.SVC()
    model.fit(features_train, labels_train)
    predict_3 = model.predict(features_test)
    print('SVM Accuracy: ', accuracy_score(labels_test, predict_3))
    model = LogisticRegression()
    model.fit(features_train, labels_train)
    predict = model.predict(features_test)
    print('Logistic Regression Accuracy: ',
          accuracy_score(labels_test, predict))


def logistic_reg_kbest(df):
    """
    Takes in a dataset then return the accuracy of logistic regression
    before and after SelectKBest feature selection. In order to find the
    highest accuracy using SelectKBest, the number of K was manually
    implemented until it resulted in the highest accuracy.
    """
    df = normalize_column(df)
    features = df.drop('Stock Movement', axis=1)
    labels = df['Stock Movement']
    ftr, fte, ltr, lte =\
        train_test_split(features, labels, test_size=0.3, random_state=2)
    model = LogisticRegression()
    model.fit(ftr, ltr)
    predict = model.predict(fte)
    print('Logistic Regression Accuracy: ', accuracy_score(lte, predict))
    k = 3
    # python logreg.py
    fs = SelectKBest(score_func=f_classif, k=k)
    ftr_selected = fs.fit_transform(ftr, ltr)
    selected_feature = fs.fit(ftr, ltr)
    ftr_selected_df = pd.DataFrame({'Features': list(ftr.columns),
                                    'Scores': selected_feature.scores_})
    ftr_selected_df = ftr_selected_df.sort_values(by='Scores', ascending=False)
    fte_selected = fs.transform(fte)
    model_2 = model.fit(ftr_selected, ltr)
    predict_2 = model_2.predict(fte_selected)
    print('Logistic Regression Accuracy After KBest: ',
          accuracy_score(lte, predict_2))
    print('Top', k, 'Features:', fs.get_feature_names_out())
    sns.catplot(data=ftr_selected_df, x='Features', y='Scores', kind='bar')
    plt.ylabel('Scores')
    plt.title('Ranking Features based on ANOVA F-value')
    # plt.xticks(rotation=45)
    plt.savefig('ranking_features.png', bbox_inches='tight')


def logistic_reg_rfe(df):
    """
    Takes in a dataset then return the accuracy of logistic regression
    before and after recursive feature elimination feature selection.
    """
    df = normalize_column(df)
    features = df.drop('Stock Movement', axis=1)
    labels = df['Stock Movement']
    ftr, fte, ltr, lte =\
        train_test_split(features, labels, test_size=0.3, random_state=2)
    model = LogisticRegression()
    model.fit(ftr, ltr)
    predict = model.predict(fte)
    print('Logistic Regression Accuracy: ', accuracy_score(lte, predict))
    rfe = RFE(estimator=model, step=1)
    ftr_selected = rfe.fit_transform(ftr, ltr)
    selected_feature = rfe.fit(ftr, ltr)
    ftr_selected_df = pd.DataFrame({'Feature': list(ftr.columns),
                                    'Scores': selected_feature.ranking_})
    print(ftr_selected_df.sort_values(by='Scores', ascending=True))
    fte_selected = rfe.transform(fte)
    model_2 = model.fit(ftr_selected, ltr)
    predict_2 = model_2.predict(fte_selected)
    print('Logistic Regression Accuracy After RFE: ',
          accuracy_score(lte, predict_2))
    print('Top Features:', rfe.get_feature_names_out())


def normalize_column(df):
    """
    Returns simple normalization of the dataset.
    """
    # df.drop(columns='Stock Movement')
    df = df.drop(columns=['Release_Date', 'Ticker'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    for col in df.columns:
        min = np.min(df[col])
        max = np.max(df[col])
        df[col] = (df[col] - min)/(max-min)
    print(df.shape)
    return df

if __name__ == '__main__':
    main()
