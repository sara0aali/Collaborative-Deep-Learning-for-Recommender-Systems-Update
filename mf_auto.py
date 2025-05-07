import numpy as np
import h5py
from auto_functions import initialization, autoEncoder, getoutPut

# Hyperparameters
INPUT_LAYER = 314
HIDDEN_UNIT1 = 40
HIDDEN_UNIT2 = 30
LEARNING_RATE = 0.001 / 5
EPOCH_NUM = 100
mu, sigma = 0, 0.1
l = 30
alpha = 40
l2_u = 100.0
l2_v = 100.0
batch = 500
ratio_l = 10.0
ratio_u = 1000000.0
HIDDEN_UNIT = [HIDDEN_UNIT1, HIDDEN_UNIT2]

def main(denoise=True):
    diction = [('ind_empleado', 5), ('pais_residencia', 24), ('sexo', 3), ('ind_nuevo', 2), ('indrel', 2),
               ('indrel_1mes', 4), ('tiprel_1mes', 4), ('indresi', 2), ('indext', 2), ('conyuemp', 3),
               ('canal_entrada', 158), ('indfall', 2), ('cod_prov', 53), ('ind_actividad_cliente', 2),
               ('segmento', 4), ('antiguedad_binned', 10), ('age_binned', 24), ('renta_binned', 10)]
    lenList = [tup[1] for tup in diction]
    accList = [sum(lenList[:i+1]) for i in range(len(lenList))]

    with h5py.File('user_infor.h5', 'r') as hf:
        xtrain = hf['infor'][:]
    with h5py.File('rating_tr_numpy.h5', 'r') as hf:
        rating_mat = hf['rating'][:]

    W1, W2, b1, b2, c1, c2 = initialization(INPUT_LAYER, HIDDEN_UNIT, mu, sigma)
    u = np.random.normal(0, 0.1, (rating_mat.shape[0], l))
    v = np.random.normal(0, 0.1, (rating_mat.shape[1], l))

    p = (rating_mat > 0).astype(int)
    c = 1 + alpha * rating_mat

    learning_rate = 0.005
    lambda_u = 0.01
    lambda_v = 0.01

    print('Start training...')
    for iterate in range(10):
        for i in range(rating_mat.shape[0]):
            for j in range(rating_mat.shape[1]):
                if p[i, j] > 0:
                    err_ij = rating_mat[i, j] - np.dot(u[i], v[j])
                    grad_u = -2 * c[i, j] * err_ij * v[j] + 2 * lambda_u * u[i]
                    grad_v = -2 * c[i, j] * err_ij * u[i] + 2 * lambda_v * v[j]
                    u[i] -= learning_rate * grad_u
                    v[j] -= learning_rate * grad_v
        print(f"Iteration {iterate+1} complete")
        loss = np.linalg.norm(p - np.dot(u, v.T)) + l2_u * np.linalg.norm(u) + l2_v * np.linalg.norm(v)
        print(f"Loss: {loss:.4f}")

        W1, W2, b1, b2, c1, c2 = autoEncoder(ratio_l, ratio_u, batch, W1, W2, xtrain, u, b1, b2, c1, c2, accList, EPOCH_NUM, LEARNING_RATE, denoise)
        hidden = getoutPut(W1, W2, b1, b2, xtrain, accList)
        u = hidden
        print(f"Loss after autoencoder: {np.linalg.norm(p - np.dot(u, v.T)):.4f}")

    with h5py.File('u_final_auto.h5', 'w') as hf:
        hf.create_dataset("u", data=u)
    with h5py.File('v_final_auto.h5', 'w') as hf:
        hf.create_dataset("v", data=v)
    with h5py.File('W1_final.h5', 'w') as hf:
        hf.create_dataset("W1", data=W1)
    with h5py.File('b1_final.h5', 'w') as hf:
        hf.create_dataset("b1", data=b1)
    with h5py.File('c1_final.h5', 'w') as hf:
        hf.create_dataset("c1", data=c1)
    with h5py.File('W2_final.h5', 'w') as hf:
        hf.create_dataset("W2", data=W2)
    with h5py.File('b2_final.h5', 'w') as hf:
        hf.create_dataset("b2", data=b2)
    with h5py.File('c2_final.h5', 'w') as hf:
        hf.create_dataset("c2", data=c2)

    predictions = np.dot(u, v.T)
    np.save('predictions.npy', predictions)
    mae = np.mean(np.abs(predictions - rating_mat))
    rmse = np.sqrt(np.mean((predictions - rating_mat) ** 2))
    with open('evaluation_metrics.txt', 'w') as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"RMSE: {rmse}\n")
    print("Files saved: u_final_auto.h5, v_final_auto.h5, predictions.npy, evaluation_metrics.txt")

if __name__ == '__main__':
main(denoise=True)

# ذخیره ماتریس‌های نهایی
with h5py.File('u_final_auto.h5', 'w') as hf:
    hf.create_dataset("u", data=u)
with h5py.File('v_final_auto.h5', 'w') as hf:
    hf.create_dataset("v", data=v)

# ذخیره پیش‌بینی‌های مدل
predictions = np.dot(u, v.T)
np.save('predictions.npy', predictions)

# ذخیره متریک‌های ارزیابی (در صورت وجود ground_truth)
ground_truth = rating_mat  # فرض کنید ماتریس واقعی همان rating_mat است
mae = np.mean(np.abs(predictions - ground_truth))
rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))

with open('evaluation_metrics.txt', 'w') as f:
    f.write(f"MAE: {mae}\n")
    f.write(f"RMSE: {rmse}\n")

print("Files saved: u_final_auto.h5, v_final_auto.h5, predictions.npy, evaluation_metrics.txt")

