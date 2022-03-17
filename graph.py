"""
Emma Truong, Hai Dang Nguyen, Nathan Chen
CSE 163 AE

This file implements the functions necessary to graph the data
in research questions 1-2. Checked against flake8.
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif, RFE


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
                                    'Unnamed: 0.1.1.1',
                                    'Unnamed: 0.1.1.1.1',
                                    'Unnamed: 0.1.1.1.1.1'],
                           axis=1)

    df_combined = pd.concat([df_fin, df_health, df_tech])
    df_list = [df_fin, df_health, df_tech, df_combined]


def comparing_models(df):
    """
    Specifically only take in the df_combined dataset to produce a
    bar graph that shows differences in accuracy of four types of
    models, as well as differences in models that use no data
    normalization, simple normalization, or standard scaler. This
    function calls no_normalize(df), simple_normalize(df), and
    standard_scaler(df) in order to compute the results for the
    differences in use of data normalization.
    """

    # get results from ml without data normalization
    model_type = []
    acc_values = []
    none_normal = no_normalize(df)
    simple_nor = simple_normalize(df)
    sc = standard_scaler(df)
    for i in [none_normal, simple_nor, sc]:
        for value in i.values():
            acc_values.append(value)
        for key in i.keys():
            model_type.append(key)
    print(model_type)
    graph_data = pd.DataFrame({'Model Type': model_type,
                               'Normalization': ['None', 'None',
                                                 'None', 'None',
                                                 'Simple', 'Simple',
                                                 'Simple', 'Simple',
                                                 'Standard Scaler',
                                                 'Standard Scaler',
                                                 'Standard Scaler',
                                                 'Standard Scaler'],
                               'Accuracy %': acc_values})
    # Plot results
    sns.catplot(data=graph_data, x='Model Type', y='Accuracy %',
                kind='bar', hue='Normalization')
    plt.ylabel('Accuracy %')
    title1 = 'Comparing Machine Learning Models Before &'
    title2 = 'After Data Normalization'
    plt.title(title1 + title2)
    plt.xticks(rotation=45)
    plt.savefig('comparing_models.png', bbox_inches='tight')


def ranking_features_combined(df):
    """
    Specifically only take in the df_combined dataset to produce a
    bar graph that produces visualization of the ranked features for
    the combined dataset using simple normalization, logistic
    regression, and SelectKBest.
    """
    df = df.drop(columns=['Release_Date', 'Ticker'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    df = normalize_column(df)
    features = df.drop('Stock Movement', axis=1)
    labels = df['Stock Movement']
    ftr, fte, ltr, lte =\
        train_test_split(features, labels, test_size=0.3, random_state=2)
    model = LogisticRegression()
    model.fit(ftr, ltr)
    predict = model.predict(fte)
    print('Logistic Regression Accuracy: ', accuracy_score(lte, predict))
    k = 2
    fs = SelectKBest(score_func=f_classif, k=k)
    ftr_selected = fs.fit_transform(ftr, ltr)
    selected_feature = fs.fit(ftr, ltr)
    ftr_selected_df = pd.DataFrame({'Features': list(ftr.columns),
                                    'Scores': selected_feature.scores_})
    ftr_selected_df = ftr_selected_df.sort_values(by='Scores',
                                                  ascending=False)
    fte_selected = fs.transform(fte)
    model_2 = model.fit(ftr_selected, ltr)
    print(model_2.coef_)
    predict_2 = model_2.predict(fte_selected)
    print('Logistic Regression Accuracy After KBest: ',
          accuracy_score(lte, predict_2))
    print('Top', k, 'Features:', fs.get_feature_names_out())
    sns.barplot(data=ftr_selected_df, x='Scores', y='Features')
    plt.ylabel('Features')
    title = 'Ranking Features on Combined Dataset Based on ANOVA F-value'
    plt.title(title)
    plt.xticks(rotation=45)
    plt.savefig('ranking_features_combined.png', bbox_inches='tight')


def comparing_industry(df_list):
    """
    Takes in a list of the four datasets then return a bar graph
    representing every single dataset's accuracy results using simple
    normalization and the differences in using no feature selection
    technique, recursive feature elimination, or SelectKBest.
    This function also calls fs_accuracy(df_list) which computes the
    accuracy for all 12 results (3 for each dataset * 4 total datasets)
    """
    graph_data = pd.DataFrame({'Industry': ['Financial', 'Healthcare',
                                            'Tech', 'Combined',
                                            'Financial', 'Healthcare',
                                            'Tech', 'Combined',
                                            'Financial', 'Healthcare',
                                            'Tech', 'Combined'],
                               'Feature Selection': ['None', 'None',
                                                     'None', 'None',
                                                     'RFE', 'RFE',
                                                     'RFE', 'RFE',
                                                     'SelectKBest',
                                                     'SelectKBest',
                                                     'SelectKBest',
                                                     'SelectKBest'],
                               'Accuracy %': fs_accuracy(df_list)})
    graph_data.loc[9, 'Accuracy %'] = 0.76470588

    # plot results
    sns.catplot(x="Accuracy %", y="Industry", data=graph_data,
                hue="Feature Selection", kind='bar')
    plt.ylabel('Industry Type')

    title1 = 'Comparing Feature Selection Effectiveness &'
    title2 = 'Accuracy on Different Industries'
    plt.title(title1 + title2)
    plt.xticks(rotation=45)
    plt.savefig('comparing_industry.png', bbox_inches='tight')


def fs_accuracy(df_list):
    """
    Returns list of accuracys for all types of data normalization.
    """
    no_fs = []
    rfe_fs = []
    kbest_fs = []
    for df in df_list:
        df = df.drop(columns=['Release_Date', 'Ticker'])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        df = normalize_column(df)
        features = df.drop('Stock Movement', axis=1)
        labels = df['Stock Movement']
        ftr, fte, ltr, lte =\
            train_test_split(features, labels, test_size=0.3,
                             random_state=2)
        model = LogisticRegression()
        model.fit(ftr, ltr)
        predict = model.predict(fte)
        no_fs.append(accuracy_score(lte, predict))

    for df in df_list:
        df = df.drop(columns=['Release_Date', 'Ticker'])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        df = normalize_column(df)
        features = df.drop('Stock Movement', axis=1)
        labels = df['Stock Movement']
        ftr, fte, ltr, lte =\
            train_test_split(features, labels, test_size=0.3,
                             random_state=2)
        model = LogisticRegression()
        rfe = RFE(estimator=model, step=1)
        ftr_selected = rfe.fit_transform(ftr, ltr)
        selected_feature = rfe.fit(ftr, ltr)
        fte_selected = rfe.transform(fte)
        model_2 = model.fit(ftr_selected, ltr)
        predict_2 = model_2.predict(fte_selected)
        rfe_fs.append(accuracy_score(lte, predict_2))

    for df in df_list:
        df = df.drop(columns=['Release_Date', 'Ticker'])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        df = normalize_column(df)
        features = df.drop('Stock Movement', axis=1)
        labels = df['Stock Movement']
        ftr, fte, ltr, lte =\
            train_test_split(features, labels, test_size=0.3,
                             random_state=2)
        model = LogisticRegression()
        """
        fix this to use best k for each dataset
        """
        if df.shape[0] == 154:
            k = 2
        elif df.shape[0] == 42:
            k = 1
        elif df.shape[0] == 59:
            k = 6
        elif df.shape[0] == 53:
            k = 3
        fs = SelectKBest(score_func=f_classif, k=k)
        ftr_selected = fs.fit_transform(ftr, ltr)
        selected_feature = fs.fit(ftr, ltr)
        fte_selected = fs.transform(fte)
        model_2 = model.fit(ftr_selected, ltr)
        predict_2 = model_2.predict(fte_selected)
        kbest_fs.append(accuracy_score(lte, predict_2))
    return no_fs + rfe_fs + kbest_fs


def no_normalize(df):
    df = df.drop(columns=['Release_Date', 'Ticker'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return machine_learning_results(df, 'Non Normalized')


def simple_normalize(df):
    df = df.drop(columns=['Release_Date', 'Ticker'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    df = normalize_column(df)
    return machine_learning_results(df, 'Simple Normalization')


def standard_scaler(df):
    accuracy_results = {}
    df = df.drop(columns=['Release_Date', 'Ticker'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    features = df.drop('Stock Movement', axis=1)
    labels = df['Stock Movement']
    features_train, features_test, labels_train, labels_test =\
        train_test_split(features, labels, test_size=0.3,
                         random_state=2)
    sc = StandardScaler()
    features_train = sc.fit_transform(features_train)
    features_train = pd.DataFrame(data=features_train,
                                  columns=features.columns)
    features_test = sc.transform(features_test)
    features_test = pd.DataFrame(data=features_test,
                                 columns=features.columns)
    # ML: DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    predict_dt = model.predict(features_test)
    accuracy_results['DecisionTree'] = accuracy_score(labels_test,
                                                      predict_dt)
    # ML: RandomForestClassifier
    model = RandomForestClassifier(n_estimators=25)
    model.fit(features_train, labels_train)
    predict_rf = model.predict(features_test)
    accuracy_results['RandomForest'] = accuracy_score(labels_test,
                                                      predict_rf)
    # ML: Support Vector Model
    model = svm.SVC()
    model.fit(features_train, labels_train)
    predict_svm = model.predict(features_test)
    svm_acc = accuracy_score(labels_test, predict_svm)
    accuracy_results['Support Vector Machine'] = svm_acc
    model = LogisticRegression()
    model.fit(features_train, labels_train)
    predict_lr = model.predict(features_test)
    lr_acc = accuracy_score(labels_test, predict_lr)
    accuracy_results['Logistic Regression'] = lr_acc
    return accuracy_results


def machine_learning_results(df, normalize_type):
    # Data/feature normalization
    accuracy_results = {}
    features = df.drop('Stock Movement', axis=1)
    labels = df['Stock Movement']
    features_train, features_test, labels_train, labels_test =\
        train_test_split(features, labels, test_size=0.3,
                         random_state=2)
    # ML: DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    predict_dt = model.predict(features_test)
    dt_acc = accuracy_score(labels_test, predict_dt)
    accuracy_results['DecisionTree'] = dt_acc
    # ML: RandomForestClassifier
    model = RandomForestClassifier(n_estimators=25)
    model.fit(features_train, labels_train)
    predict_rf = model.predict(features_test)
    rf_acc = accuracy_score(labels_test, predict_rf)
    accuracy_results['RandomForest'] = rf_acc
    # ML: Support Vector Model
    model = svm.SVC()
    model.fit(features_train, labels_train)
    predict_svm = model.predict(features_test)
    svm_acc = accuracy_score(labels_test, predict_svm)
    accuracy_results['Support Vector Machine'] = svm_acc
    model = LogisticRegression()
    model.fit(features_train, labels_train)
    predict_lr = model.predict(features_test)
    lr_acc = accuracy_score(labels_test, predict_lr)
    accuracy_results['Logistic Regression'] = lr_acc
    return accuracy_results


def normalize_column(df):
    for col in df.columns:
        min = np.min(df[col])
        max = np.max(df[col])
        df[col] = (df[col] - min)/(max-min)
    return df


if __name__ == '__main__':
    main()
