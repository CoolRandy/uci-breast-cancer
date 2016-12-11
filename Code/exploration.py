# -*- coding: utf-8 -*-


# 1) Loading the Data

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)

# path to "data.csv" file (change to your directory)
path = '/Users/Hannes/Desktop/City University/Machine Learning/Coursework/ML_Nazareth_Draxl/ML_Nazareth_Draxl/Data/data.csv'
df = pd.read_csv(path, header=0)

# print the first 5 samples
df.head()

# 2) Heatmap correlation matrix 

# First we have to encode the categorical "diagnosis" column.
df.diagnosis = pd.factorize(df.diagnosis)[0]

# We drop the unecessary Unnamed 32 and id column.
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Then we calculate the correlation matrix.
corr = df.corr()

# Now lets plot it as a heatmap.
hm = sns.heatmap(corr, cbar=True, annot=False, yticklabels=
                df.columns, xticklabels=df.columns)
plt.show()
# Additionally we plot the first column in the correlation matrix
# representing the point biserial correlation between the 30 features
# and the target starting from the highest negative correlation.
corr.iloc[:, 0].nsmallest(n=31)


# Almost all features are negatively correlated with the target 
# (diagnosis), e.g. the point biserial correlation between 'concave points_worst' 
# and 'diagnosis' is -0.7935, a strong negative correlation.

# 3) Feature Characteristics:

# Scatterplot between two features which are highly correlated
sns.jointplot("radius_worst", "perimeter_worst", data=df,
                   kind="scatter", space=0, color="b", size=5, ratio=3)
plt.show()

# We observe a very high correlation between these two features. The 
# pearson correlation coefficient is 0.99, representing an almost 
# perfect positive linear correlation between 'radius_worst' and 
# 'perimeter_worst'. The p-value equals 0, meaning that the H0 Hypothesis 
# (no correlation) can be discareded at any given significance level. 
# Similar strong behaviour can be observed in other feature correlations 
# as well. These strong correlations are a sign of lots of redundancy in the data. 

# Plot the classdistribution of these two features
sns.pairplot(df, size=2, vars=["perimeter_worst", "radius_worst"], 
	hue='diagnosis')
plt.show()

# The scatterplots and distributions depict that higher feature 
# values are more likely to be of the malignant class represented by the 
# blue color. This pattern can be observed throughout almost all features. 

# 4) Summary statistics per Class

# define two different dataframes consisting of either malignant or beningn samples.
df_malignant = df[df.diagnosis == 0]
df_benign = df[df.diagnosis == 1]

# 30 features would be to much to include in the summary statistic, as such we focussed
# on the features that have the highest correlation with the target variable. 

# Define the 5 most highly correlated features with the target
features = ['concave points_worst', 'perimeter_worst', 'concave points_mean',
            'radius_worst', 'perimeter_mean']


# Summary statistic of malignant Class
df_malignant[features].describe()
# Summary statistic of benign Class
df_benign[features].describe()

