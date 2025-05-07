import numpy as np
import h5py
from scipy import stats

# Hyperparameters
alpha = 40
lambda_u = 0.01
lambda_v = 0.01
learning_rate = 0.001

# Load training and validation subsets
with h5py.File('rating_tr_numpy.h5', 'r') as hf:
    rating_tr = hf['rating'][:].astype(int)

with h5py.File('rating_val_numpy.h5', 'r') as hf:
    rating_val = hf['rating'][:].astype(int)

# Load latent factors
with h5py.File('u_40_40+100.h5', 'r') as hf:
    u = hf['u'][:]
with h5py.File('v_40_40+100.h5', 'r') as hf:
    v = hf['v'][:]

# Define preference and confidence matrices
p = (rating_tr > 0).astype(int)
c = 1 + alpha * rating_tr

# Perform mini-batch SGD
for i in range(u.shape[0]):
    for j in range(v.shape[0]):
        if p[i, j] > 0:
            err_ij = rating_tr[i, j] - np.dot(u[i], v[j])
            grad_u = -2 * c[i, j] * err_ij * v[j] + 2 * lambda_u * u[i]
            grad_v = -2 * c[i, j] * err_ij * u[i] + 2 * lambda_v * v[j]
            u[i] -= learning_rate * grad_u
            v[j] -= learning_rate * grad_v

# Predict ratings
r_pred = np.dot(u, v.T)

# Evaluate only new interactions
rating_val[rating_tr > 0] = 0
m = (p > 0)

# Percentile ranking
rank = 0
total = 0
for i in range(rating_val.shape[0]):
    prod = rating_val[i]
    prod_predict = np.ma.masked_array(r_pred[i], mask=m[i])
    if np.sum(prod) > 0:
        total += np.count_nonzero(prod)
        rank += sum(stats.percentileofscore(prod_predict[~prod_predict.mask], prod_predict[j])
                    for j in np.where(prod > 0)[0])

print("Total rated items:", total)
if total > 0:
    print("Percentile ranking score:", 100 - rank / total)

# Explainability functions
def explain_recommendation(user_index, item_index, u, v):
    user_factors = u[user_index]
    item_factors = v[item_index]
    score = np.dot(user_factors, item_factors)
    contributions = user_factors * item_factors
    explanations = [f"Dimension {i+1}: Contribution {c:.2f}" for i, c in enumerate(contributions)]
    return score, explanations

def save_explanations_to_file(user_index, item_index, u, v, file_name="explanations.txt"):
    score, explanations = explain_recommendation(user_index, item_index, u, v)
    with open(file_name, "w") as f:
        f.write(f"Recommendation Score for User {user_index} and Item {item_index}: {score:.2f}\n")
        f.write("Contributions:\n")
        f.write("\n".join(explanations))
    print(f"Explanations saved to {file_name}")

# Example usage
user_id = 10
item_id = 5
score, explanations = explain_recommendation(user_id, item_id, u, v)
print(f"Recommendation Score for User {user_id} and Item {item_id}: {score:.2f}")
print("Contributions:")
print("\n".join(explanations))
save_explanations_to_file(user_id, item_id, u, v)
