import numpy as np
import math

randomSeed = np.random.RandomState(42)

def initialization(INPUT_LAYER, HIDDEN_UNIT, mu, sigma):
    NUM_HIDDEN = len(HIDDEN_UNIT)
    
    W1 = randomSeed.normal(mu, sigma, [HIDDEN_UNIT[0], INPUT_LAYER])
    b1 = randomSeed.normal(mu, sigma, [HIDDEN_UNIT[0]])
    c1 = randomSeed.normal(mu, sigma, [INPUT_LAYER])
    
    if NUM_HIDDEN == 1:
        return W1, b1, c1
    elif NUM_HIDDEN == 2:
        W2 = randomSeed.normal(mu, sigma, [HIDDEN_UNIT[1], HIDDEN_UNIT[0]])
        b2 = randomSeed.normal(mu, sigma, [HIDDEN_UNIT[1]])
        c2 = randomSeed.normal(mu, sigma, [HIDDEN_UNIT[0]])
        return W1, W2, b1, b2, c1, c2
    elif NUM_HIDDEN == 3:
        W2 = randomSeed.normal(mu, sigma, [HIDDEN_UNIT[1], HIDDEN_UNIT[0]])
        b2 = randomSeed.normal(mu, sigma, [HIDDEN_UNIT[1]])
        c2 = randomSeed.normal(mu, sigma, [HIDDEN_UNIT[0]])
        W3 = randomSeed.normal(mu, sigma, [HIDDEN_UNIT[2], HIDDEN_UNIT[1]])
        b3 = randomSeed.normal(mu, sigma, [HIDDEN_UNIT[2]])
        c3 = randomSeed.normal(mu, sigma, [HIDDEN_UNIT[1]])
        return W1, W2, W3, b1, b2, b3, c1, c2, c3
    else:
        print('Too many layers')
        return 0

def preActivation(W, x, b):
    b_shaped = np.reshape(b, (1, b.size))
    b_shaped = np.repeat(b_shaped, x.shape[0], axis=0)
    return np.dot(x, W.T) + b_shaped

def sigmoid_forward(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_backward(x):
    return np.multiply(sigmoid_forward(x), (1 - sigmoid_forward(x)))

def tanh_forward(z):
    return np.tanh(z)

def tanh_backward(z):
    return 1 - np.square(tanh_forward(z))

def softmax(z):
    z_exp = np.exp(z)
    z_sum = np.sum(z_exp, axis=1).reshape(-1, 1)
    return z_exp / z_sum

def getLoss(W1, W2, xtrain, u, b1, b2, c1, c2, accList, l2_reg):
    loss = 0
    for t in range(1):
        x = xtrain
        A1 = preActivation(W1, x, b1)
        h1 = tanh_forward(A1)
        A2 = preActivation(W2, h1, b2)
        h2 = tanh_forward(A2)
        Ahat = preActivation(W1.T, h2, c1)
        xhat0 = softmax(Ahat[:, 0:accList[0]])
        # (other xhat layers as before...)

        prediction = np.concatenate(
            (xhat0, xhat1, xhat2, xhat3, xhat4, xhat5, xhat6, xhat7, xhat8, xhat9,
             xhat10, xhat11, xhat12, xhat13, xhat14, xhat15, xhat16, xhat17), axis=1)

        loss -= np.sum(x * np.log(prediction))
    
    meanLoss = loss / xtrain.shape[0]
    loss_enc = np.sum(np.square(h2 - u)) / xtrain.shape[0]
    
    # Adding L2 regularization to the loss
    l2_loss = (l2_reg / 2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    total_loss = meanLoss + loss_enc + l2_loss
    
    return total_loss, loss_enc

def autoEncoder(ratio_l, ratio_u, batch, W1, W2, xtrain, u, b1, b2, c1, c2, accList, EPOCH_NUM, LEARNING_RATE, l2_reg, denoise=True):
    for i in range(EPOCH_NUM + 1):
        loss = 0
        if i == 0:
            [loss, loss_enc] = getLoss(W1, W2, xtrain, u, b1, b2, c1, c2, accList, l2_reg)
            print(i, loss, loss_enc)
        else:
            for t in range(int(math.floor(xtrain.shape[0] / batch))):
                x_sample = xtrain[t * batch:(t + 1) * batch, :]
                u_sample = u[t * batch:(t + 1) * batch, :]
                
                if denoise:
                    x = x_sample.astype(float)
                    x += randomSeed.normal(0, 0.1, size=x.shape)
                else:
                    x = x_sample

                # Forward and Backward propagation as before...

                # Adding L2 regularization to the weight updates
                W1 -= LEARNING_RATE * ((dLdW1_out.T + dLdW1_in) / ratio_l + dudW1 / ratio_u + l2_reg * W1)
                W2 -= LEARNING_RATE * ((dLdW2_out.T + dLdW2_in) / ratio_l + dudW2 / ratio_u + l2_reg * W2)
                b1 -= LEARNING_RATE * (dLdb1 / ratio_l + dudb1 / ratio_u)
                b2 -= LEARNING_RATE * (dLdb2 / ratio_l + dudb2 / ratio_u)
                c1 -= LEARNING_RATE * dLdc1 / ratio_l
                c2 -= LEARNING_RATE * dLdc2 / ratio_l

            [loss, loss_enc] = getLoss(W1, W2, xtrain, u, b1, b2, c1, c2, accList, l2_reg)
            print(i, loss, loss_enc)
    return W1, W2, b1, b2, c1, c2


#
