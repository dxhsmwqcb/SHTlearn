import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
import numpy as np
import pickle

# 读取MovieLens 20M数据集
ratings = pd.read_csv('ml-20m/ratings.csv')
ratings = ratings[['userId', 'movieId', 'rating']]

## 预处理数据
def preprocess_data(data, user_col, item_col, rating_col):
    users = data[user_col].astype('category').cat.codes
    items = data[item_col].astype('category').cat.codes
    ratings = data[rating_col].values
    
    num_users = users.max() + 1
    num_items = items.max() + 1
    
    matrix = lil_matrix((num_users, num_items))
    for user, item, rating in zip(users, items, ratings):
        matrix[user, item] = rating
    
    return matrix.tocsr(), users, items

matrix, users, items = preprocess_data(ratings, 'userId', 'movieId', 'rating')

# 拆分数据集
def split_data(matrix, test_size=0.2):
    train_matrix = matrix.copy().tolil()
    test_matrix = lil_matrix(matrix.shape)
    
    non_zero_indices = matrix.nonzero()
    non_zero_pairs = list(zip(non_zero_indices[0], non_zero_indices[1]))
    np.random.shuffle(non_zero_pairs)
    
    num_test_samples = int(test_size * len(non_zero_pairs))
    test_samples = non_zero_pairs[:num_test_samples]
    
    for user, item in test_samples:
        test_matrix[user, item] = matrix[user, item]
        train_matrix[user, item] = 0
    
    return train_matrix.tocsr(), test_matrix.tocoo()

train_matrix, test_matrix = split_data(matrix, test_size=0.2)

# 保存数据
def save_data(train_matrix, test_matrix, dataset_name):
    with open(f'{dataset_name}_trnMat.pkl', 'wb') as f:
        pickle.dump(train_matrix, f)
    with open(f'{dataset_name}_tstMat.pkl', 'wb') as f:
        pickle.dump(test_matrix, f)

save_data(train_matrix, test_matrix, 'movielens_20m')
