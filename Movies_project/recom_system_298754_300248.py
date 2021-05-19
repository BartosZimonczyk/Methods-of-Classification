import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
import sklearn.model_selection as ms
import argparse
import warnings
from numba import jit, njit, vectorize


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description='Method to complete Z matrix with given algorithm \
                for given train and test sets. \
                Returns RMSE for given algorithm.',
    )
parser.add_argument(
    '--train',
    required=True,
    help='.csv file with train set'
    )
parser.add_argument(
    '--test',
    required=True,
    help='.csv file with test set'
    )
parser.add_argument(
    '--alg',
    required=True,
    help='Name of algorithm to use. UPPERCASE'
)
parser.add_argument(
    '--result',
    required=True,
    help='Name of the file to save RMSE into.'
)

args = parser.parse_args()


train = pd.read_csv(args.train)
train = train.drop('timestamp', axis=1)

# change IDs to str because ints cannot be the dictionary keys
train['movieId'] = train['movieId'].transform(lambda x: 'movieId_' + str(x))
train['userId'] = train['userId'].transform(lambda x: 'userId_' + str(x))

# pivot long dataframe into matrix that we are interested in
Z = train.pivot(index='userId', columns='movieId', values='rating')

# train with subtracted validation set and validation set for some methods
train = pd.read_csv(args.train)
train = train.drop('timestamp', axis=1)

train_without_valid, train_with_valid = ms.train_test_split(train, test_size=0.1, stratify=train['userId'])
train_without_valid.loc[:,'movieId'] = train_without_valid['movieId'].transform(lambda x: 'movieId_' + str(x))
train_without_valid.loc[:,'userId'] = train_without_valid['userId'].transform(lambda x: 'userId_' + str(x))
Z_without_valid = train_without_valid.pivot(index='userId', columns='movieId', values='rating')
train_with_valid.loc[:,'movieId'] = train_with_valid['movieId'].transform(lambda x: 'movieId_' + str(x))
train_with_valid.loc[:,'userId'] = train_with_valid['userId'].transform(lambda x: 'userId_' + str(x))
Z_with_valid = train_with_valid.pivot(index='userId', columns='movieId', values='rating')

# test set
test = pd.read_csv(args.test).drop('timestamp', axis=1)
test['movieId'] = test['movieId'].transform(lambda x: 'movieId_' + str(x))
test['userId'] = test['userId'].transform(lambda x: 'userId_' + str(x))
test = test.pivot(index='userId', columns='movieId', values='rating')

# full set
Z_full = pd.read_csv('ratings.csv').drop('timestamp', axis=1)
Z_full['movieId'] = Z_full['movieId'].transform(lambda x: 'movieId_' + str(x))
Z_full['userId'] = Z_full['userId'].transform(lambda x: 'userId_' + str(x))
Z_full = Z_full.pivot(index='userId', columns='movieId', values='rating')

###
# Functions that transform train and test set into matrices of full dimensions 

@njit
def new_into_full(Z, Z_full, replace_column):
    full_matrix = np.zeros_like(Z_full)
    full_matrix[:] = np.nan

    i = 0
    for j in range(Z_full.shape[1]):
        if replace_column[j]:
            full_matrix[:, j] = Z[:, i]
            i += 1
    
    return full_matrix

def transform_into_full_matrix(Z, Z_full):
    Z_columns = Z.columns.to_series().transform(lambda x: int(x[8:])).to_numpy()
    Z_full_columns = Z_full.columns.to_series().transform(lambda x: int(x[8:])).to_numpy()
    replace_column = np.isin(Z_full_columns, Z_columns)

    transformed_Z = new_into_full(Z.to_numpy(copy=True),
                                  Z_full.to_numpy(copy=True),
                                  replace_column,)
    
    transformed_Z = pd.DataFrame(data=transformed_Z,
                                 index=Z_full.index,
                                 columns=Z_full.columns,)
    
    return transformed_Z

# Making test set with full dimensions
T_big = transform_into_full_matrix(test, Z_full)
Z_valid_big = transform_into_full_matrix(Z_with_valid, Z_full)

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

Z3 = (0.9 * Z1 + 1.1 * Z2) / 2

# Z1, Z2, Z3 for splitted set onto train/valid
Z1_without_valid = Z_without_valid.fillna(values_movies)
Z2_without_valid = Z_without_valid.T.fillna(values_users).T
Z3_without_valid = (0.9 * Z1_without_valid + 1.1 * Z2_without_valid) / 2


def RMSE(Z_big, T_big):
    m = T_big.count().sum()
    diff_sq = Z_big.sub(T_big) ** 2
    s = diff_sq.sum().sum() / m

    return np.sqrt(s)


def SVD1(Z_svd, Z_full, T_big):
    svd = TruncatedSVD(n_components=Z_full.shape[0] // 60, random_state=2137)
    svd.fit(Z_svd)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(Z_svd)/svd.singular_values_
    H = np.dot(Sigma2, VT)
    approx_Z = np.dot(W, H)
    Z_svd = pd.DataFrame(data=approx_Z, index=Z_svd.index, columns=Z_svd.columns)
    # Z_svd = np.around(Z_svd * 2) / 2
    # print(Z_svd)

    return RMSE(transform_into_full_matrix(Z_svd, Z_full), T_big)

def my_NMF(Z_nmf, Z_full, T_big):
    nmf = NMF(n_components=Z_full.shape[0] // 60, init='random', random_state=2137, max_iter=1000)
    W = nmf.fit_transform(Z_nmf)
    H = nmf.components_
    approx_Z = np.dot(W, H)
    Z_nmf = pd.DataFrame(data=approx_Z, index=Z_nmf.index, columns=Z_nmf.columns)

    return RMSE(transform_into_full_matrix(Z_nmf, Z_full), T_big)

### Place to copy-paste function that results in minimized RMSE on validation set for SVD1 and NMF
#
#
#
#

### Place to copy-paste SVD2 that is trained and validated on train set :)
#
#
#
#
def are_different(X, Y, eps=1e-5):
    diff = X - Y
    diff = np.linalg.norm(diff)
    return diff > eps

# this one takes Z splitted for train and valid tests
def SVD2_all_tuned(Z_current, Z_valid_big, Z_full, T_big, Z=Z_without_valid, eps=1e-5):
    rmses = [100000]
    i = 1

    svd = TruncatedSVD(n_components=Z_full.shape[0] // 60, random_state=2137)

    Z_previous = Z_current.copy()
    svd.fit(Z_current)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(Z_current)/svd.singular_values_
    H = np.dot(Sigma2, VT)
    approx_Z = np.dot(W, H)
    Z_current = pd.DataFrame(data=approx_Z, index=Z_current.index, columns=Z_current.columns)
    Z_current = Z_current * Z.isna() + Z.fillna(0)
    rmses.append(RMSE(transform_into_full_matrix(Z_current, Z_full), Z_valid_big))

    while are_different(Z_previous, Z_current, eps) and rmses[-1] < rmses[-2]:
        print(f'Current iteration: {i}')
        print(f'Current RMSE on validation set: {rmses[-1]}')
        i += 1

        Z_previous = Z_current.copy()
        svd.fit(Z_current)
        Sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_
        W = svd.transform(Z_current)/svd.singular_values_
        H = np.dot(Sigma2, VT)
        approx_Z = np.dot(W, H)
        Z_current = pd.DataFrame(data=approx_Z, index=Z_current.index, columns=Z_current.columns)
        Z_current = Z_current * Z.isna() + Z.fillna(0)
        rmses.append(RMSE(transform_into_full_matrix(Z_current, Z_full), Z_valid_big))

    return rmses[1:-1], RMSE(transform_into_full_matrix(Z_current, Z_full), T_big)

def SVD2_tuned(Z_current, Z_valid_big, Z_full, T_big, Z=Z_without_valid, eps=1e-5):
    return SVD2_all_tuned(Z_current, Z_valid_big, Z_full, T_big, Z=Z, eps=eps)[1]


# SGD
@njit
def SGD_loss_and_gradient(Z, W, H, i, j, l=0.0):
    # we will use this calculation few times, so to reduce time, we 
    # will store it in a variable
    diff = (Z[i, j] - np.dot(W[i, :], H[:, j]))

    # the gradients of Wi and Hj vecotrs
    W_grad = -2 * diff * H[:, j] + 2 * l * W[i, :]
    H_grad = -2 * diff * W[i, :] + 2 * l * H[:, j]

    # the loss for our problem
    loss = diff ** 2

    return W_grad, H_grad, loss

@njit
def SGD(Z, W, H,
        alpha=1e-2,
        alpha_decay = 1,
        multiply_alpha_after_epochs = 10,
        momentum = 0,
        decay = 0,
        l=0.0,
        num_epoch=1000):
    ## prep for performing SGD
    Z_mask = Z != 0
    coords = np.where(Z_mask)

    # this is list of pairs that are non empty in Z
    coords = np.array([(coords[0][i], coords[1][i]) for i in range(len(coords[0]))])

    # in order to use numba we have to convert everything to proper types
    W = W.astype(np.float32)
    H = H.astype(np.float32)

    # adding velocity parameter to the model
    W_velocity = np.zeros_like(W)
    H_velocity = np.zeros_like(H)

    # list of all losses achived during trainging
    losses = np.zeros(num_epoch)
    for epoch in range(num_epoch):
        # shuffling coords, core step in SGD
        np.random.shuffle(coords)

        # reseting loss for each epoch
        loss = 0

        for iter_ in range(coords.shape[0]):
            i = coords[iter_][0]
            j = coords[iter_][1]

            W_grad, H_grad, current_loss = SGD_loss_and_gradient(Z, W , H, i, j, l=l)

            # Implementation of weight decay on weights, by changing the gradients
            W_grad += decay * W[i, :]
            H_grad += decay * H[:, j]

            # Updating velocity of search with momentum
            W_velocity[i, :] = W_velocity[i, :] * momentum + alpha * W_grad
            H_velocity[:, j] = H_velocity[:, j] * momentum + alpha * H_grad

            # going with matrices along gradient with given learning rate = alpha
            W[i, :] = W[i, :] - W_velocity[i, :]
            H[:, j] = H[:, j] - H_velocity[:, j]
            
            # update of loss function for given coords, standarized
            loss = loss + current_loss / coords.shape[0]
    
        # applying alpha decay
        # the longer we search, the shorter is the learning rate
        if epoch % multiply_alpha_after_epochs == 0:
            alpha = alpha * alpha_decay
        losses[epoch] = loss
    
    return W, H, losses

def SGD_convinent(Z, Z_full, T_big,
                  r=15,
                  alpha=0.01,
                  alpha_decay=1,
                  multiply_alpha_after_epochs=10,
                  momentum = 0,
                  decay=0,
                  l=0.0,
                  num_epoch=1000,
                  init_mean=0,
                  init_scale=1,
                  ):

    # we have to start with some prediction of H and W, but since giving a lot
    # information at the start results in going far away from minimum, we are
    # just initializng random matrices
    W = np.random.normal(loc=init_mean, scale=init_scale, size=(Z.shape[0], r))
    H = np.random.normal(loc=init_mean, scale=init_scale, size=(r, Z.shape[1]))

    Z_mean = Z.mean().mean()
    Z_std = np.nanstd(Z.to_numpy().flatten())

    W, H, losses = SGD(((Z - Z_mean)/Z_std).fillna(0).to_numpy(copy=True),
                       W, H,
                       alpha=alpha,
                       alpha_decay=alpha_decay,
                       multiply_alpha_after_epochs=multiply_alpha_after_epochs,
                       momentum=momentum,
                       decay=decay,
                       l=l,
                       num_epoch=num_epoch,
                       )
    
    Z_predicted = (W @ H) * Z_std + Z_mean
    Z_predicted = pd.DataFrame(data=Z_predicted,
                               index=Z.index,
                               columns=Z.columns)
    Z_predicted_big = transform_into_full_matrix(Z_predicted, Z_full)
    rmse = RMSE(Z_predicted_big, T_big)

    return rmse, losses[-1]

if args.alg == 'NMF':
    final_rmse = np.array([my_NMF(Z3, Z_full, T_big)])
    np.savetxt(args.result, final_rmse)

elif args.alg == 'SVD1':
    final_rmse = np.array([SVD1(Z3, Z_full, T_big)])
    np.savetxt(args.result, final_rmse)

elif args.alg == 'SVD2':
    final_rmse = np.array([SVD2_tuned(Z3_without_valid, Z_valid_big, Z_full, T_big)])
    np.savetxt(args.result, final_rmse)

elif args.alg == 'SGD':
    final_rmse = np.array([SGD_convinent(Z, Z_full, T_big,
              r=10,
              alpha=0.01,
              alpha_decay=0.95,
              multiply_alpha_after_epochs=10,
              momentum=0.0,
              decay=0.0000,
              l=0.1,
              num_epoch=200,
              init_mean=0,
              init_scale=0.01,)[0]])
    np.savetxt(args.result, final_rmse)
