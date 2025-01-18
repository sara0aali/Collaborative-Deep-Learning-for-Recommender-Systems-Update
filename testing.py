import numpy as np
import h5py
from scipy import stats

# Hyperparameters
alpha = 40
lambda_u = 0.01  # L2 regularization term for user matrix
lambda_v = 0.01  # L2 regularization term for item matrix
learning_rate = 0.001  # Learning rate for gradient descent

# Load training and validation subsets
with h5py.File('rating_tr_numpy.h5', 'r') as hf:
    rating_tr = hf['rating'][:].astype(int)

with h5py.File('rating_val_numpy.h5', 'r') as hf:
    rating_val = hf['rating'][:].astype(int)

# Load user and item latent matrices
with h5py.File('u_40_40+100.h5', 'r') as hf:
    u = hf['u'][:]
with h5py.File('v_40_40+100.h5', 'r') as hf:
    v = hf['v'][:]

# Define preference and confidence matrices
p = (rating_tr > 0).astype(int)
c = 1 + alpha * rating_tr

# Gradient descent for matrix factorization
for i in range(u.shape[0]):
    for j in range(v.shape[1]):
        if p[i, j] > 0:
            err_ij = rating_tr[i, j] - np.dot(u[i, :], v[:, j])
            u[i, :] += learning_rate * (err_ij * v[:, j] - lambda_u * u[i, :])
            v[:, j] += learning_rate * (err_ij * u[i, :] - lambda_v * v[:, j])

# Calculate predicted ratings
r_pred = np.dot(u, v.T)

# Mask old choices to keep only new choices
rating_val[rating_tr > 0] = 0
m = (p > 0)  # Mask matrix

# Percentile ranking calculation
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

# Explainability function
def explain_recommendation(user_index, item_index, u, v):
    """
    Provides an explanation for why an item is recommended to a user.
    """
    user_factors = u[user_index]
    item_factors = v[item_index]
    score = np.dot(user_factors, item_factors)  # Compute recommendation score

    # Calculate contribution of each dimension to the score
    contributions = user_factors * item_factors
    explanations = [f"Dimension {i+1}: Contribution {c:.2f}" for i, c in enumerate(contributions)]

    return score, explanations

# Example usage
user_id = 10  # User index
item_id = 5   # Item index
score, explanations = explain_recommendation(user_id, item_id, u, v)

print(f"Recommendation Score for User {user_id} and Item {item_id}: {score:.2f}")
print("Contributions:")
print("\n".join(explanations))

# Save explanations to a file
def save_explanations_to_file(user_index, item_index, u, v, file_name="explanations.txt"):
    """
    Save explanation of a recommendation to a file.
    """
    score, explanations = explain_recommendation(user_index, item_index, u, v)
    with open(file_name, "w") as f:
        f.write(f"Recommendation Score for User {user_index} and Item {item_index}: {score:.2f}\n")
        f.write("Contributions:\n")
        f.write("\n".join(explanations))
    print(f"Explanations saved to {file_name}")

# Save an example explanation
save_explanations_to_file(user_id, item_id, u, v)
