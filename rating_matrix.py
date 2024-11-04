import numpy as np
import pandas as pd

# Load training rating data and create training rating matrix
with pd.HDFStore('ratingDF_tr.h5') as store:
    df_rating_tr = store['df_rating_tr']

user_id = np.array(df_rating_tr['ncodpers'].unique())
prod_id = np.sort(np.array(df_rating_tr['prodIdx'].unique()))

# Create rating matrix for training data
rating_mat_tr = pd.DataFrame(0, index=user_id, columns=prod_id)
rating_mat_tr = rating_mat_tr.add(df_rating_tr.pivot(index='ncodpers', columns='prodIdx', values='rating').fillna(0), fill_value=0)

# Save the training rating matrix
with pd.HDFStore('rating_mat_tr.h5') as store:
    store['rating_mat'] = rating_mat_tr

print('Training set complete')

# Load validation rating data and create validation rating matrix
with pd.HDFStore('ratingDF_val.h5') as store:
    df_rating_val = store['df_rating_val']

# Create rating matrix for validation data
rating_mat_val = pd.DataFrame(0, index=user_id, columns=prod_id)
rating_mat_val.update(df_rating_val.pivot(index='ncodpers', columns='prodIdx', values='rating').fillna(0))

# Save the validation rating matrix
with pd.HDFStore('rating_mat_val.h5') as store:
    store['rating_mat'] = rating_mat_val

print('Validation set complete')
