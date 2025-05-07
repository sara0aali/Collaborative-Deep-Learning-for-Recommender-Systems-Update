import h5py
import numpy as np
from auto_functions import initialization, autoEncoder_mono, getoutPut_mono

# Hyperparameters
INPUT_LAYER = 314
HIDDEN_UNIT1 = 40
LEARNING_RATE = 0.001 / 5
EPOCH_NUM = 100
mu, sigma = 0, 0.1
l = 40
alpha = 40
l2_u = 100.0
l2_v = 100.0
lambda_u = 0.01
lambda_v = 0.01
batch = 500
ratio_l = 10.0
ratio_u = 1000000.0


def main(denoise=True):
    # Feature configuration
    diction = [('ind_empleado', 5), ('pais_residencia', 24), ('sexo', 3), ('ind_nuevo', 2), ('indrel', 2),
               ('indrel_1mes', 4), ('tiprel_1mes', 4), ('indresi', 2), ('indext', 2), ('conyuemp', 3),
               ('canal_entrada', 158), ('indfall', 2), ('cod_prov', 53), ('ind_actividad_cliente', 2),
               ('segmento', 4), ('antiguedad_binned', 10), ('age_binned', 24), ('renta_binned', 10)]
    lenList = [tup[1] for tup in diction]
    accList = [sum(lenList[:i + 1]) for i in range(len(lenList))]

    # Load data
    with h5py.File('user_infor.h5', 'r') as hf:
        xtrain = hf['infor'][:]
    with h5py.File('rating_tr_numpy.h5', 'r') as hf:
        rating_mat = hf['rating'][:]

    # Initialize
    W1, b1, c1 = initialization(INPUT_LAYER, [HIDDEN_UNIT1], mu, sigma)
    u = np.random.normal(0, 0.1, (rating_mat.shape[0], l))
    v = np.random.normal(0, 0.1, (rating_mat.shape[1], l))

    # Preference & Confidence
    p = (rating_mat > 0).astype(int)
    c = 1 + alpha * rating_mat

    print('Start training with mini-batch SGD...')

    for iterate in range(10):
        for i in range(rating_mat.shape[0]):
            for j in range(rating_mat.shape[1]):
                if p[i, j] > 0:
                    err_ij = rating_mat[i, j] - np.dot(u[i], v[j])
                    grad_u = -2 * c[i, j] * err_ij * v[j] + 2 * lambda_u * u[i]
                    grad_v = -2 * c[i, j] * err_ij * u[i] + 2 * lambda_v * v[j]
                    u[i] -= LEARNING_RATE * grad_u
                    v[j] -= LEARNING_RATE * grad_v

        print(f"Iteration {iterate + 1} complete")
        loss = np.linalg.norm(p - np.dot(u, v.T)) + l2_u * np.linalg.norm(u) + l2_v * np.linalg.norm(v)
        print(f"Loss: {loss:.4f}")

        # Autoencoder step
        W1, b1, c1 = autoEncoder_mono(ratio_l, ratio_u, batch, W1, xtrain, u, b1, c1, accList, EPOCH_NUM, LEARNING_RATE, denoise)
        hidden = getoutPut_mono(W1, b1, xtrain, accList)
        u = hidden
        print(f"Loss after autoencoder: {np.linalg.norm(p - np.dot(u, v.T)):.4f}")

    # Save
    with h5py.File('u_40_mono_40+100_auto.h5', 'w') as hf:
        hf.create_dataset("u", data=u)
    with h5py.File('v_40_mono_40+100_auto.h5', 'w') as hf:
        hf.create_dataset("v", data=v)
    with h5py.File('W1_40_mono_40+100.h5', 'w') as hf:
        hf.create_dataset("W1", data=W1)
    with h5py.File('b1_40_mono_40+100.h5', 'w') as hf:
        hf.create_dataset("b1", data=b1)
    with h5py.File('c1_40_mono_40+100.h5', 'w') as hf:
        hf.create_dataset("c1", data=c1)

    # Predictions & Evaluation
    predictions = np.dot(u, v.T)
    np.save('predictions_mono.npy', predictions)
    mae = np.mean(np.abs(predictions - rating_mat))
    rmse = np.sqrt(np.mean((predictions - rating_mat) ** 2))
    with open('evaluation_metrics_mono.txt', 'w') as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"RMSE: {rmse}\n")
    print("Saved mono model outputs and evaluations.")


if __name__ == '__main__':
    main(denoise=True)
