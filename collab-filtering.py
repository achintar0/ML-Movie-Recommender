import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


# load USER, MOVIES AND RATINGS as a df
user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
item_cols = ['movie_id', 'title', 'release_date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

users = pd.read_csv('ml-100k/u.user', sep="|", names=user_cols, encoding='latin-1')
movies = pd.read_csv('ml-100k/u.item', sep="|", names=item_cols, encoding='latin-1')
ratings = pd.read_csv('ml-100k/u.data', sep="\t", names=rating_cols, encoding='latin-1')

ratings = ratings.drop('timestamp', axis=1)

# plot age count distribution
sns.displot(data=users, x='age')
plt.title("Age Distribution", fontsize=12)
plt.figure(figsize=(16,8))
plt.show()

movie_ratings_count = ratings.merge(movies, on="movie_id")
movie_ratings_count['title'].value_counts()[0:10]

# getting ready for train and test split.
X = ratings.copy()
y = ratings['user_id']

# split train and test datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, stratify=y, random_state=42)

# compute the RMSE (root mean squared error).
def RMSE(y_true, y_predict):
    return np.sqrt(mean_squared_error(y_true, y_predict))

def score(CF_Model):
    #create tuple pairs based on the test set.
    id_pairs = zip(X_test['user_id'], X_test['movie_id'])

    #get the prediction set 
    y_predict = np.array([CF_Model(user, movie) for (user, movie) in id_pairs])

    y_true = np.array(X_test['rating'])

    return RMSE(y_true, y_predict)

# USER-BASED FILTERING
# pivot_table rating matrix
rating_matrix = X_train.pivot_table(index='user_id', columns='movie_id', values='rating')
rating_matrix.head()

# User-Based Collab-Filtering using MEAN ratings.
def CF_user_mean(user_id, movie_id):
    if movie_id in rating_matrix:
        mean_rating = rating_matrix[movie_id].mean()
    else:
        mean_rating = 3.0

    return mean_rating

# Weighted MEAN ratings.

#create a dummy ratings matrix, because cosine_similarity does not work with NaN values.
dum_rating_matrix = rating_matrix.copy().fillna(0)

cos_sim_mtx = cosine_similarity(dum_rating_matrix, dum_rating_matrix)

# convert into pandas df
cos_sim_mtx = pd.DataFrame(cos_sim_mtx, index=rating_matrix.index, columns=rating_matrix.index)

cos_sim_mtx.head(10)

def CF_user_weighted_mean(user_id, movie_id):
    if movie_id in rating_matrix:
        similarity_scores = cos_sim_mtx[user_id]
        movie_ratings = rating_matrix[movie_id]

        idx = movie_ratings[movie_ratings.isnull()].index

        movie_ratings = movie_ratings.dropna()

        similarity_scores = similarity_scores.drop(idx)

        wmean_rating = np.dot(similarity_scores, movie_ratings) / similarity_scores.sum()
    else:
        wmean_rating =3.0
    
    return wmean_rating

score(CF_user_weighted_mean)



