import h5py
import numpy as np
import auto_functions as auto
import time

# Hyperparameters
INPUT_LAYER = 314
LEARNING_RATE = 0.001 / 50 * 2
EPOCH_NUM = 50
mu, sigma = 0, 0.1
l = 40
HIDDEN_UNIT1 = l
alpha = 40
l2_u = 100.0
l2_v = 100.0
lambda_reg = 0.01  # L2 Regularization
batch = 500
ratio_l = 400.0
ratio_u = 0.5
l1 = 0.01

# Define data dictionary
diction = [
    ('ind_empleado', 5), ('pais_residencia', 24), ('sexo', 3), ('ind_nuevo', 2), ('indrel', 2),
    ('indrel_1mes', 4), ('tiprel_1mes', 4), ('indresi', 2), ('indext', 2), ('conyuemp', 3),
    ('canal_entrada', 158), ('indfall', 2), ('cod_prov', 53), ('ind_actividad_cliente', 2),
    ('segmento', 4), ('antiguedad_binned', 10), ('age_binned', 24), ('renta_binned', 10)
]

def compute_cumulative_lengths(diction):
    """Compute cumulative lengths for dictionary fields."""
    lenList = [tup[1] for tup in diction]
    return [sum(lenList[:i + 1]) for i in range(len(lenList))]


def load_data():
    """Load user information and rating matrix."""
    with h5py.File('user_infor.h5', 'r') as hf:
        xtrain = hf['infor'][:]

    with h5py.File('rating_tr_numpy.h5', 'r') as hf:
        rating_mat = hf['rating'][:]

    return xtrain, rating_mat


def initialize_weights_and_biases():
    """Initialize weights, biases, and latent matrices."""
    W1, b1, c1 = auto.initialization(INPUT_LAYER, [HIDDEN_UNIT1], mu, sigma)
    u = np.random.rand(rating_mat.shape[0], l)
    v = np.random.rand(rating_mat.shape[1], l)
    return W1, b1, c1, u, v


def update_u(u, v, p, c, l2_u, lambda_reg):
    """Update user matrix (u)."""
    v2 = v.T.dot(v)
    for i in range(p.shape[0]):
        nonzero_list = np.nonzero(p[i, :])[0]
        v_nonzero = v[nonzero_list, :]
        c_diag_nonzero = np.diag(c[i, nonzero_list] - 1)
        temp_u = np.dot(c[i, nonzero_list], v_nonzero)
        u[i, :] = np.dot(temp_u, np.linalg.pinv(l2_u * np.identity(l) + np.dot(v_nonzero.T, c_diag_nonzero.dot(v_nonzero)) + v2 + lambda_reg * u[i, :]))
    return u


def update_v(v, u, p, c, l2_v, lambda_reg):
    """Update item matrix (v)."""
    u2 = u.T.dot(u)
    for j in range(p.shape[1]):
        nonzero_list = np.nonzero(p[:, j])[0]
        c_diag_nonzero = np.diag(c[nonzero_list, j] - 1)
        u_nonzero = u[nonzero_list, :]
        temp_v = np.dot(c[nonzero_list, j], u_nonzero)
        v[j, :] = np.dot(temp_v, np.linalg.pinv(l2_v * np.identity(l) + np.dot(u_nonzero.T, c_diag_nonzero.dot(u_nonzero)) + u2 + lambda_reg * v[j, :]))
    return v


def save_results(u, v, W1, b1, c1, hidden):
    """Save matrices and autoencoder outputs to files."""
    with h5py.File('u_40_mono_40+100_try_auto.h5', 'w') as hf:
        hf.create_dataset("u", data=u)
    with h5py.File('v_40_mono_40+100_try_auto.h5', 'w') as hf:
        hf.create_dataset("v", data=v)
    with h5py.File('W1_40_mono_40+100_try.h5', 'w') as hf:
        hf.create_dataset("W1", data=W1)
    with h5py.File('b1_40_mono_40+100_try.h5', 'w') as hf:
        hf.create_dataset("b1", data=b1)
    with h5py.File('c1_40_mono_40+100_try.h5', 'w') as hf:
        hf.create_dataset("c1", data=c1)
    with h5py.File('h_40_mono_40+100_auto_try.h5', 'w') as hf:
        hf.create_dataset("hidden", data=hidden)


def main(denoise=True):
    accList = compute_cumulative_lengths(diction)
    xtrain, rating_mat = load_data()
    W1, b1, c1, u, v = initialize_weights_and_biases()

    # Define preference and confidence matrices
    p = np.zeros(rating_mat.shape)
    p[rating_mat > 0] = 1
    c = 1 + alpha * rating_mat

    iteration = 100
    print('Start training...')
    for iterate in range(iteration):
        start = time.time()
        u = update_u(u, v, p, c, l2_u, lambda_reg)
        print(f'U update complete in {time.time() - start:.2f}s')

        start = time.time()
        v = update_v(v, u, p, c, l2_v, lambda_reg)
        print(f'V update complete in {time.time() - start:.2f}s')

        loss = np.linalg.norm(p - np.dot(u, v.T))
        print(f'Iteration {iterate + 1}/{iteration}, Loss: {loss:.4f}')

        W1, b1, c1 = auto.autoEncoder_mono(ratio_l, ratio_u, batch, W1, xtrain, u, b1, c1, accList, EPOCH_NUM, LEARNING_RATE, l1, denoise)
        hidden = auto.getoutPut_mono(W1, b1, xtrain, accList)
        u = hidden
        print(f'Updated loss after autoencoder: {np.linalg.norm(p - np.dot(u, v.T)):.4f}')

    save_results(u, v, W1, b1, c1, hidden)

if __name__ == "__main__":
    main(denoise=True)
