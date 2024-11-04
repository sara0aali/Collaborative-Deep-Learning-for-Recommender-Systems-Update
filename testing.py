import numpy as np
import h5py
from scipy import stats
import auto_fun as auto

alpha = 40

# Load training and validation subset
with h5py.File('rating_tr_numpy.h5', 'r') as hf:
    rating_tr = hf['rating'][:].astype(int)

with h5py.File('rating_val_numpy.h5', 'r') as hf:
    rating_val = hf['rating'][:].astype(int)

# Load u and v matrices
with h5py.File('u_40_40+100.h5', 'r') as hf:
    u = hf['u'][:]
with h5py.File('v_40_40+100.h5', 'r') as hf:
    v = hf['v'][:]

# Define preference and confidence matrices
p = (rating_tr > 0).astype(int)
c = 1 + alpha * rating_tr

# Calculate predicted ratings
r_pred = np.dot(u, v.T)

# Mask old choices to keep only new choices
rating_val[rating_tr > 0] = 0
m = (p > 0)  # Mask matrix

# Calculate rank and total counts for percentile-ranking
rank = 0
total = 0
for i in range(rating_val.shape[0]):
    prod = rating_val[i]
    prod_predict = np.ma.masked_array(r_pred[i], mask=m[i])
    if np.sum(prod) > 0:
        total += np.count_nonzero(prod)
        rank += sum(stats.percentileofscore(prod_predict[~prod_predict.mask], prod_predict[j]) for j in np.where(prod > 0)[0])

# Print results
print("Total rated items:", total)
if total > 0:
    print("Percentile ranking score:", 100 - rank / total)
