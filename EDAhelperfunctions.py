import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def basic_eda(df):
    '''Takes a dataframe and displays many useful statistics about the input
    dataframe including giving a view of the dataframe, the datatype of each
    column, whether each column has any NA values, whether there are any
    duplicate rows, the column names, and summary statistics for each column.'''

    print("The dataframe:\n", df.head())
    print("Info:\n")
    df.info()
    print("Number of duplicates:\n", df.duplicated().sum())
    print("Column Names:\n", df.columns)
    print("Describe:\n", df.describe(include = [np.number]))
    print("Describe:\n", df.describe(include = ['O']))
    print("Data Types:\n", df.dtypes)

def plot_feature(df, col, target_col):
    '''Takes a dataframe, the column to plot, and the target column.Creates a
    plot of each feature plotting the distribution of the feature and
    then plotting the feature against the target variable to see if it seems to
    have predictive value'''

    fig = plt.figure(figsize = (16, 8))
    plt.subplot(1,2,1)
    if df[col].dtypes == 'int64' or df[col].dtypes == 'float64':
        sns.distplot(df[col], bins = 20)
        plt.subplot(1,2,2)
        sns.scatterplot(x = col, y = target_col, data = df, alpha = 0.2)
        plt.xticks(rotation = 90, fontsize = 'small')
        plt.show()
        fig.savefig(col+'feature.png')
    else:
        target_means = df.groupby(col)[target_col].mean()
        level = list(target_means.sort_values().index)
        df[col] = df[col].astype('category')
        df[col].cat.reorder_categories(level, inplace = True)
        sns.countplot(x = col, data = df)
        plt.xticks(rotation = 90)
        plt.subplot(1,2,2)
        sns.boxplot(x = col, y = target_col, data = df)
        plt.xticks(rotation = 90)
        fig.savefig(col+'feature.png')

def corr_map_pair_plot(df, target_col, exclude_cols):
    '''Takes a dataframe, the target column, and any columns to exclude. Creates
    two plots, a correlation matrix that shows the correlation between all
    features and a pairplot showing the relationship between features. For
    categorical features, the mean of the target for each group replaces the
    category in order to compute correlations and plot the feature. This means
    that feature to feature comparison for categorical features may be
    unreliable.'''

    df2 = df.copy()
    for col in df2.columns:
        if (hasattr(df2[col], 'cat')) & (col not in exclude_cols):
            cats = list(df2[col].cat.categories)
            map_dict = {}
            for cat in cats:
                map_dict[cat] = df2.loc[df[col] == cat, target_col].mean()
            df2[col] = pd.to_numeric(df2[col].map(map_dict))
    fig = plt.figure(figsize = (15, 10))
    sns.heatmap(df2.corr(), annot = True)

    ##This is a workaround as seaborn and matplotlib are not playing nice.
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    ##

    plt.xticks(rotation = 45)
    fig.savefig('corr_matrix.png')
    plt.show()
    fig = plt.figure(figsize = (15,10))
    sns.pairplot(data = df2, kind = 'scatter', plot_kws = {'alpha':0.1})
    fig.savefig('pairplot.png')
    plt.show()
