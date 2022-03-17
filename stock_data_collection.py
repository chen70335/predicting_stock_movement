"""
Emma Truong, Hai Dang Nguyen, Nathan Chen
CSE 163 AE

This file includes the functions necessary to create dataframes of
various stock data for research questions 1-2. Checked against
flake8 errors.
"""


import yahoo_fin.stock_info as si
import pandas as pd
import numpy as np


def main():
    """
    Returns three seperate csv datasets of stocks in tech, healthcare,
    and financial industry.
    """
    tech_tickers = ['AAPL', 'NVDA', 'FB', 'RBLX', 'MSFT', 'GOOG', 'AMZN',
                    'TSM', 'DIS', 'AVGO', 'CSCO', 'VZ', 'ADBE', 'ACN', 'CMCSA',
                    'ORCL', 'INTC', 'CRM', 'QCOM', 'AMD', 'T', 'NFLX', 'TXN',
                    'TMUS', 'SPGI', 'INTU', 'SAP', 'SONY', 'AMAT', 'IBM',
                    'SHOP', 'ATVI', 'SNAP', 'DELL', 'EA', 'ZM', 'TWTR', 'SPOT',
                    'GRMN', 'DOCU', 'PAYC', 'PINS', 'GDDY',
                    'LOGI', 'ZG', 'PTON']
    health_tickers = ['A', 'ABBV', 'ABC', 'ABMD', 'UNH', 'JNJ', 'PFE',
                      'LLY', 'ABT', 'TMO', 'MRK', 'DHR', 'BMY', 'MDT',
                      'CVS', 'AMGN', 'ANTM', 'ISRG', 'SYK', 'ZTS', 'HCA',
                      'BDX', 'CI', 'GILD', 'EW', 'REGN', 'BSX', 'VRTX',
                      'HUM', 'MRNA', 'ILMN', 'CNC', 'IDXX', 'BAX', 'MCK',
                      'IQV', 'DXCM', 'RMD', 'ALGN', 'MTD', 'BIIB', 'WST',
                      'CERN', 'LH', 'ZBH', 'STE', 'PKI', 'COO', 'WAT',
                      'HOLX', 'CTLT', 'BIO', 'TFX', 'DGX', 'TECH', 'INCY',
                      'CAH', 'CRL', 'VTRS', 'HSIC', 'UHS', 'XRAY',
                      'DVA', 'OGN']
    fin_tickers = ['MTB', 'HIG', 'NTRS', 'KEY', 'RF', 'HBAN', 'RJF',
                   'CINF', 'CFG', 'BRO', 'SYF', 'SBNY', 'BEN', 'MKTX',
                   'CBOE', 'CMA', 'RE', 'LNC', 'AIZ', 'ZION', 'GL', 'PBCT',
                   'IVZ', 'BRK-B', 'JPM', 'BAC', 'WFC', 'MS', 'SCHW', 'AXP',
                   'C', 'GS', 'BLK', 'SPGI', 'CB', 'CME', 'USB', 'MMC', 'PNC',
                   'TFC', 'ICE', 'AON', 'PGR', 'MCO', 'COF', 'MET', 'AIG',
                   'TRV', 'BK', 'PRU', 'AFL', 'MSCI', 'ALL', 'AJG', 'TROW',
                   'AMP', 'SIVB', 'DFS', 'FITB', 'STT', 'FRC', 'NDAQ']

    df_list = data_collection(tech_tickers, health_tickers, fin_tickers)
    df_list = financial_data_analysis(df_list)
    df_tech = df_list[0]
    df_health = df_list[1]
    df_fin = df_list[2]

    df_tech = get_rev_earn(df_tech, tech_tickers)
    df_health = get_rev_earn(df_health, health_tickers)
    df_fin = get_rev_earn(df_fin, fin_tickers)

    df_tech = get_new_features(df_tech, tech_tickers)
    df_health = get_new_features(df_health, health_tickers)
    df_fin = get_new_features(df_fin, fin_tickers)

    df_tech.to_csv('stock_prediction_tech.csv')
    df_health.to_csv('stock_prediction_healthcare.csv')
    df_fin.to_csv('stock_prediction_financial.csv')


def data_collection(tech_tickers, health_tickers, fin_tickers):
    """
    Collects stock data in the given lists of tickers from three
    different Industries, including both financial data and stock
    movement which will be used as the labels for machine learning
    models. Requires installation of yahoo_fin API.
    """
    df_tech = get_financial_statements(tech_tickers)
    df_health = get_financial_statements(health_tickers)
    df_fin = get_financial_statements(fin_tickers)
    """
    For the below get_stock_movement functions, tickers need to be broken down
    into lists of 7 or less and ran seperately with at least 2 minutes in
    between in order to complete data collection.
    """
    df_tech = get_stock_movement(df_tech, tech_tickers)
    df_health = get_stock_movement(df_health, health_tickers)
    df_fin = get_stock_movement(df_fin, fin_tickers)
    df_list = [df_tech, df_health, df_fin]
    return df_list


def financial_data_analysis(df_list):
    """
    This dataset takes in dataframe collected using Yahoo Finance to
    calculate ratios & measurements for machine learning features.
    Returned as a separate dataset.
    """
    analyzed_df_list = []
    for df in df_list:
        df['EPS'] = df['Qtr Earnings'] / df['Outstanding Shares']
        df['Gross Profit Margin %'] = df['grossProfit'] / df['Qtr Revenue']
        df['Revenue Per Share'] = df['Qtr Revenue'] / df['Outstanding Shares']
        df['Return on Assets %'] = df['netIncome'] / df['totalAssets']
        total_se = 'totalStockholderEquity'
        df['Return on Equity %'] = df['netIncome'] / df[total_se]
        qtr_rev = 'Qtr Revenue'
        prev_rev = 'Prev Qtr Revenue'
        df['Qtr Revenue Growth %'] = (df[qtr_rev] - df[prev_rev]) / df[qtr_rev]
        qtr_earn = 'Qtr Earnings'
        prev_earn = 'Prev Qtr Earnings'
        earn_growth = 'Qtr Earnings Growth %'
        df[earn_growth] = (df[qtr_earn] - df[prev_earn]) / df[qtr_earn]
        ocf = 'totalCashFromOperatingActivities'
        fcf = 'totalCashFromFinancingActivities'
        icf = 'totalCashflowsFromInvestingActivities'
        df['Total Cash Per Share'] = df[ocf] / df['Outstanding Shares']
        curr_liab = 'totalCurrentLiabilities'
        df['Current Liability Coverage %'] = df[ocf] / df[curr_liab]
        df['Long Term Debt Coverage %'] = df[ocf] / df['longTermDebt']
        total_cf = df[ocf] + df[fcf] + df[icf]
        df['Cash Generating Power %'] = df[ocf] / total_cf
        df = df.loc[:, ('Ticker', 'Gross Profit Margin %', 'Revenue Per Share',
                        'EPS', 'Return on Assets %', 'Return on Equity %',
                        'Qtr Revenue Growth %', 'Qtr Earnings Growth %',
                        'Total Cash Per Share', 'Current Liability Coverage %',
                        'Long Term Debt Coverage %', 'Cash Generating Power %',
                        'Release_Date', 'Stock Movement')]
        df = df.dropna()
        analyzed_df_list.append(df)
    return analyzed_df_list


def get_rev_earn(df, tickers):
    """
    After the analysis, while testing the ratios we found that
    revenue & earnings growth numbers were incorrect, therefore,
    we developed a seperate function only to get the revenue and
    earnings measures, while simultaneously adding more features
    as part of attempting to increase model accuracy.
    """
    df = df.dropna()
    dict_vals = {}
    for value in df['Ticker'].values:
        dict_vals[value] = np.nan
    print(df.shape)
    for ticker in tickers:
        mask = df['Ticker'] == ticker
        shares = si.get_quote_data(ticker)['sharesOutstanding']
        qtr_rev_earn = 'quarterly_revenue_earnings'
        rev = si.get_earnings(ticker)[qtr_rev_earn].loc[:, 'revenue']
        earn = si.get_earnings(ticker)[qtr_rev_earn].loc[:, 'earnings']
        df.loc[mask, 'EPS'] = earnings[3] / shares
        df.loc[mask, 'Revenue Per Share'] = rev[3] / shares
        df.loc[mask, 'Qtr Revenue Growth %'] = (rev[3] - rev[2]) / rev[2]
        df.loc[mask, 'Qtr Earnings Growth %'] = (earn[3] - earn[2]) / earn[2]
    return df


def get_new_features(df, tickers):
    """
    Using the existing dataset and its tickers, returnthe same dataset
    with added features including P/E ratio, O/S ratio, 9 month revenue
    growth, and 9 month earnings growth.
    """
    dict_vals = {}
    for value in df['Ticker'].values:
        dict_vals[value] = np.nan
    df = df.dropna()
    print(df.shape)
    for ticker in tickers:
        if ticker in dict_vals.keys():
            mask = df['Ticker'] == ticker
            release_date = df.loc[mask, 'Release_Date'].iloc[0]
            price = si.get_data(ticker, start_date=release_date)['close'][0]
            shares = si.get_quote_data(ticker)['sharesOutstanding']
            qtr_rev_earn = 'quarterly_revenue_earnings'
            rev = si.get_earnings(ticker)[qtr_rev_earn].loc[:, 'revenue']
            earn = si.get_earnings(ticker)[qtr_rev_earn].loc[:, 'earnings']
            df.loc[mask, 'P/E Ratio'] = price / (earn[3] / shares)
            df.loc[mask, 'P/S Ratio'] = price / (rev[3] / shares)
            rev_growth = '9 Month Revenue Growth %'
            df.loc[mask, rev_growth] = (rev[3] - rev[0]) / rev[0]
            earn_growth = '9 Month Earnings Growth %'
            df.loc[mask, earn_growth] = (earn[3] - earn[0]) / earn[0]
    return df


def get_financial_statements(tickers):
    """
    Import financial information from Yahoo Finance using yahoo_fin,
    returns a dataset for the given list of tickers that includes financial
    data from all three financial statements
    """

    df_list = []
    for ticker in tickers:
        income_statement = si.get_income_statement(ticker, yearly=False)
        income_statement = pd.DataFrame(income_statement.iloc[:, 0])
        income_statement = income_statement.transpose()
        qtr_rev_earn = 'quarterly_revenue_earnings'
        revenues = si.get_earnings(ticker)[qtr_rev_earn].iloc[:, -2]
        earnings = si.get_earnings(ticker)[qtr_rev_earn].iloc[:, -1]
        income_statement['Qtr Revenue'] = revenues[3]
        income_statement['Prev Qtr Revenue'] = revenues[2]
        income_statement['Qtr Earnings'] = earnings[3]
        income_statement['Prev Qtr Earnings'] = earnings[2]
        out_shares = 'Outstanding Shares'
        shares_out = 'sharesOutstanding'
        income_statement[out_shares] = si.get_quote_data(ticker)[shares_out]
        income_statement = income_statement.loc[:, ('grossProfit', 'netIncome',
                                                    'operatingIncome',
                                                    'Qtr Revenue',
                                                    'Prev Qtr Revenue',
                                                    'Outstanding Shares',
                                                    'Qtr Earnings',
                                                    'Prev Qtr Earnings')]
        income_statement['Ticker'] = ticker

        balance_sheet = si.get_balance_sheet(ticker, yearly=False)
        balance_sheet = pd.DataFrame(balance_sheet.iloc[:, 0])
        balance_sheet = balance_sheet.transpose()
        if 'longTermDebt' not in balance_sheet.columns:
            balance_sheet['longTermDebt'] = 0
        balance_sheet = balance_sheet.loc[:, ('cash', 'totalAssets',
                                              'totalCurrentAssets',
                                              'totalLiab',
                                              'totalCurrentLiabilities',
                                              'longTermDebt',
                                              'totalStockholderEquity')]
        balance_sheet['Ticker'] = ticker

        cash_flow = si.get_cash_flow(ticker, yearly=False)
        cash_flow = pd.DataFrame(cash_flow.iloc[:, 0])
        cash_flow = cash_flow.transpose()
        cash_flow = cash_flow.loc[:, ('totalCashflowsFromInvestingActivities',
                                      'totalCashFromFinancingActivities',
                                      'totalCashFromOperatingActivities')]
        cash_flow['Ticker'] = ticker

        data_frames = [balance_sheet, cash_flow, income_statement]
        data = reduce(lambda left, right: pd.merge(left, right, on=['Ticker'],
                                                   how='outer'), data_frames)

        data['Quarter'] = '4Q2021'
        data.set_index(data.pop('Ticker'), inplace=True)
        data.reset_index(inplace=True)
        df_list.append(data)
    df = pd.concat(df_list)
    df['Release_Date'] = np.nan
    df['Stock Movement'] = np.nan
    return df


def get_stock_movement(df, tickers):
    """
    Find latest earnings release dates and the stock price movement
    from the day of release's closing price to the next day's closing price.
    This function requires the size of the parameter tickers to be around 7
    at a time, because the yahoo_fin API does not support the
    'get_earnings_history' function continuously, and therefore extracting
    data requires more time waiting in between each subset of tickers.
    """
    for ticker in tickers:
        mask_ticker = df['Ticker'] == ticker
        qtr_results = 'quarterly_results'
        latest_eps = si.get_earnings(ticker)[qtr_results].iloc[-1]['actual']
        release_date = None
        earnings_hist = si.get_earnings_history(ticker)
        for earnings in earnings_hist:
            if latest_eps == earnings['epsactual']:
                release_date = earnings['startdatetime'][:10]
                break
        df.loc[mask_ticker, 'Release_Date'] = release_date
        price = si.get_data(ticker, start_date=release_date)
        if price['close'][1] > price['close'][0]:
            df.loc[mask_ticker, 'Stock Movement'] = int(1)
        else:
            df.loc[mask_ticker, 'Stock Movement'] = int(0)
    return df


if __name__ == '__main__':
    main()
