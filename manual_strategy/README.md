# Manual Strategy
## Introduction

Develop a trading strategy using my intuition and Technical Analysis, and test it against a stock using a market simulator. 

## Data Details, Dates and Rules

    Use the symbol JPM (see the Data section below for download link)
    The in sample/development period is January 1, 2008 to December 31 2009.
    The out of sample/testing period is January 1, 2010 to December 31 2011.
    Starting cash is 100,000 dollars.
    Allowable positions are: 1000 shares long, 1000 shares short, 0 shares.
    Benchmark: The performance of a portfolio starting with 100,000 cash, investing in 1000 shares of JPM and holding that position.
    There is no limit on leverage.
    Transaction costs for RuleBasedStrategy: Commission: 9.95, Impact: 0.005.
    Transaction costs for BestPossibleStrategy: Commission: 0.00, Impact: 0.00.

Code

    indicators.py - Implements indicators as functions that operate on Pandas series. The "main" code generates the charts that illustrate the indicators in the notebook rule_based_test.ipynb.

    marketsim.py - Accepts dataframes of trades and retrieve statistics showing performances of the portfolio vs. benchmark.

    best_strategy.py - This strategy assumes that we can see the future. It creates a set of trades that represents the best a strategy could possibly do during a period. This is to give an idea of an upper bound on performance. It implements testPolicy() which returns a trades dataframe. The main part of this code calls marketsim.py as necessary to generate stats and plots.

    rule_based_strategy.py: It incorporates technical indicators from indicators.py into trading_strategy() to create order signals. It implements testPolicy() which reads the order signals and returns a trades dataframe. The main part of this code calls marketsim.py as necessary to generate stats and plots.

    rule_based_test.ipynb: This is an analysis that calls testPolicy() from RuleBasedStrategy.py and relevant functions from marketsim.py to plot and compare in-sample and out-of-sample performances of the portfolio vs. benchmark.

## Setup

You need Python 2.7.x or 3.x, and the following packages: pandas, numpy, and matplotlib.
Data

## Run

To run any script file, use:

To run any IPython Notebook, use:

jupyter notebook <notebook_name.ipynb>