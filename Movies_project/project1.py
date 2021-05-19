import numpy as np
import pandas as pd


# read csv file
train = pd.read_csv('train_ratings.csv')
train = train.drop('timestamp', axis=1)

# change IDs to str because ints cannot be the dictionary keys
train['movieId'] = train['movieId'].transform(lambda x: 'movieId_' + str(x))
train['userId'] = train['userId'].transform(lambda x: 'userId_' + str(x))

# pivot long dataframe into matrix that we are interested in
Z = train.pivot(index='userId', columns='movieId', values='rating')

# values that are replacing NaNs in Z
# values is a dict which corresponds to 'columnname': value to replace
# so every NaN in a column is replaced by the same value, for now
values_movies = {movieId: Z.loc[:, movieId].mean() for movieId in Z.columns}
Z1 = Z.fillna(values_movies)

# if you want to replace NaN based on users' mean you have to transpose the matrix
# to get back to original dimensions of Z you have to transpose it back
# so in this case if user didn't rate a movie we assume that they
# would rate every movie similar
values_users = {userId: Z.T.loc[:, userId].mean() for userId in Z.T.columns}
Z2 = Z.T.fillna(values_users).T

cov_matrix = Z.T.cov()

# printing the output to see how replacing NaNs went out
print(cov_matrix)
