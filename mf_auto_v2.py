import numpy as np
import h5py
import auto_functions as auto
import time

# تعریف پارامترهای مدل
INPUT_LAYER = 314
HIDDEN_UNIT1 = 200
HIDDEN_UNIT2 = 40
LEARNING_RATE = 0.001 / 20
EPOCH_enc = 30  # تعداد تکرارها برای اتوانکودر
EPOCH_mf = 10   # تعداد تکرارها برای به‌روزرسانی MF
mu, sigma = 0, 0.1
l = HIDDEN_UNIT2
alpha = 40
l2_u = 100.0  # مقدار Regularization برای U
l2_v = 600.0  # مقدار Regularization برای V
lambda_reg = 0.01  # مقدار L2 Regularization
batch = 500
ratio_l = 1000.0
ratio_u = 2.0
l1 = 0.01

def main(denoise=True):
    # تعریف ساختار ویژگی‌ها
    diction = [('ind_empleado', 5), ('pais_residencia', 24), ('sexo', 3), ('ind_nuevo', 2), ('indrel', 2), 
               ('indrel_1mes', 4), ('tiprel_1mes', 4), ('indresi', 2), ('indext', 2), ('conyuemp', 3), 
               ('canal_entrada', 158), ('indfall', 2), ('cod_prov', 53), ('ind_actividad_cliente', 2), 
               ('segmento', 4), ('antiguedad_binned', 10), ('age_binned', 24), ('renta_binned', 10)]
    lenList = [tup[1] for tup in diction]
    accList = [sum(lenList[:i+1]) for i in range(len(lenList))]

    # خواندن اطلاعات کاربر
    with h5py.File('user_infor.h5', 'r') as hf:
        xtrain = hf['infor'][:]
    
    # خواندن ماتریس رتبه‌بندی
    with h5py.File('rating_tr_numpy.h5', 'r') as hf:
        rating_mat = hf['rating'][:]
    
    # مقداردهی اولیه وزن‌ها و بایاس‌ها
    W1, W2, b1, b2, c1, c2 = auto.initialization(INPUT_LAYER, [HIDDEN_UNIT1, HIDDEN_UNIT2], mu, sigma)
    u = np.random.rand(rating_mat.shape[0], l)
    v = np.random.rand(rating_mat.shape[1], l)

    # تعریف ماتریس‌های ترجیح و اطمینان
    p = np.zeros(rating_mat.shape)
    p[rating_mat > 0] = 1
    c = 1 + alpha * rating_mat

    iteration = 30
    print('شروع آموزش...')
    for iterate in range(iteration):
        # به‌روزرسانی‌های Matrix Factorization
        for iter_mf in range(EPOCH_mf):
            # به‌روزرسانی v
            start = time.time()
            u2 = u.T.dot(u)
            for j in range(rating_mat.shape[1]):
                nonzero_list = np.nonzero(p[:, j])[0]
                c_diag_nonzero = np.diag(c[nonzero_list, j] - 1)
                u_nonzero = u[nonzero_list, :]
                temp_v = np.dot(c[nonzero_list, j], u_nonzero)
                v[j, :] = np.dot(temp_v, np.linalg.pinv(l2_v * np.identity(l) + np.dot(np.dot(u_nonzero.T, c_diag_nonzero), u_nonzero) + u2 + lambda_reg * v[j, :]))
            print('به‌روزرسانی v تکمیل شد - زمان:', time.time() - start)

            # به‌روزرسانی u
            start = time.time()
            v2 = v.T.dot(v)
            for i in range(rating_mat.shape[0]):
                nonzero_list = np.nonzero(p[i, :])[0]
                v_nonzero = v[nonzero_list, :]
                c_diag_nonzero = np.diag(c[i, nonzero_list] - 1)
                temp_u = np.dot(c[i, nonzero_list], v_nonzero)
                u[i, :] = np.dot(temp_u, np.linalg.pinv(l2_u * np.identity(l) + np.dot(np.dot(v_nonzero.T, c_diag_nonzero), v_nonzero) + v2 + lambda_reg * u[i, :]))
            print('به‌روزرسانی u تکمیل شد - زمان:', time.time() - start)

            # محاسبه و چاپ خسارت
            print('خسارت MF:', np.linalg.norm(p - np.dot(u, v.T)))
        
        # به‌روزرسانی‌های Autoencoder
        W1, W2, b1, b2, c1, c2 = auto.autoEncoder(ratio_l, ratio_u, batch, W1, W2, xtrain, u, b1, b2, c1, c2, accList, EPOCH_enc, LEARNING_RATE, l1, denoise=True)
        hidden = auto.getoutPut(W1, W2, b1, b2, xtrain, accList)

        # چاپ نرم‌ها برای نظارت
        print('Norm u:', np.sum(np.square(u)))
        print('Norm difference:', np.sum(np.square(u - hidden)))
        u = hidden
        print('خسارت کلی پس از autoencoder:', np.linalg.norm(p - np.dot(u, v.T)))

    # ذخیره ماتریس‌ها در فایل‌ها
    with h5py.File('u_200_40bi.h5', 'w') as hf:
        hf.create_dataset("u", data=u)
    with h5py.File('v_200_40bi.h5', 'w') as hf:
        hf.create_dataset("v", data=v)
    with h5py.File('W1_200_40bi.h5', 'w') as hf:
        hf.create_dataset("W1", data=W1)
    with h5py.File('b1_200_40bi.h5', 'w') as hf:
        hf.create_dataset("b1", data=b1)
    with h5py.File('c1_200_40bi.h5', 'w') as hf:
        hf.create_dataset("c1", data=c1)
    with h5py.File('W2_200_40bi.h5', 'w') as hf:
        hf.create_dataset("W2", data=W2)
    with h5py.File('b2_200_40bi.h5', 'w') as hf:
        hf.create_dataset("b2", data=b2)
    with h5py.File('c2_200_40bi.h5', 'w') as hf:
        hf.create_dataset("c2", data=c2)

    return hidden

main(denoise=True)
