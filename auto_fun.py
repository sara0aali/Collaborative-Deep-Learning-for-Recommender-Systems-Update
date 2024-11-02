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
    else:
        print('too many layers')
        return None

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
    z_sum = np.sum(z_exp, axis=1)
    z_sum = np.reshape(z_sum, (z_sum.size, 1))
    z_sum = np.repeat(z_sum, z.shape[1], axis=1)
    return z_exp / z_sum

def getLoss(W1, W2, xtrain, u, b1, b2, c1, c2, accList, lambda_reg):
    loss = 0
    for _ in range(1):  # Single iteration for simplicity
        x = xtrain

        A1 = preActivation(W1, x, b1)
        h1 = tanh_forward(A1)
                
        A2 = preActivation(W2, h1, b2)
        h2 = tanh_forward(A2)
        
        A3 = preActivation(W2.T, h2, c2)
        h3 = tanh_forward(A3)

        Ahat = preActivation(W1.T, h3, c1)
        
        xhat = np.concatenate([softmax(Ahat[:, accList[i]:accList[i+1]]) for i in range(len(accList)-1)], axis=1)
        
        loss -= np.sum(x * np.log(xhat))
        
    meanLoss = loss / xtrain.shape[0]
    l2_loss = (lambda_reg / 2) * (np.sum(W1**2) + np.sum(W2**2))  # L2 regularization term
    total_loss = meanLoss + l2_loss
    return total_loss

def autoEncoder(ratio_l, ratio_u, batch, W1, W2, xtrain, u, b1, b2, c1, c2, accList, EPOCH_NUM, LEARNING_RATE, lambda_reg, denoise=True):
    beta = 0.9
    dW10 = np.zeros(W1.shape)
    dW20 = np.zeros(W2.shape)
    db10 = np.zeros(b1.shape)
    db20 = np.zeros(b2.shape)
    dc10 = np.zeros(c1.shape)
    dc20 = np.zeros(c2.shape)
    
    for i in range(EPOCH_NUM + 1):
        if i == 0:
            loss = getLoss(W1, W2, xtrain, u, b1, b2, c1, c2, accList, lambda_reg)
            print(i, loss)
        else:
            for t in range(int(math.floor(xtrain.shape[0] / batch))):
                x_sample = xtrain[t * batch:(t + 1) * batch, :]
                u_sample = u[t * batch:(t + 1) * batch, :]
                
                if denoise:
                    x = x_sample + randomSeed.normal(0, 0.1, size=x_sample.shape)
                else:
                    x = x_sample
                
                # Forward pass
                A1 = preActivation(W1, x, b1)
                h1 = tanh_forward(A1)
                
                A2 = preActivation(W2, h1, b2)
                h2 = tanh_forward(A2)
                
                A3 = preActivation(W2.T, h2, c2)
                h3 = tanh_forward(A3)

                Ahat = preActivation(W1.T, h3, c1)
                xhat = np.concatenate([softmax(Ahat[:, accList[i]:accList[i+1]]) for i in range(len(accList)-1)], axis=1)

                # Backpropagation with L2 Regularization
                dLdAhat = xhat - x_sample
                dLdW1_out = np.dot(dLdAhat.T, h3) + lambda_reg * W1  # L2 reg term
                dLdc1 = np.sum(dLdAhat, axis=0)
                dLdh3 = np.dot(dLdAhat, W1.T)
                
                dLdA3 = np.multiply(dLdh3, tanh_backward(A3))
                dLdW2_out = np.dot(dLdA3.T, h2) + lambda_reg * W2  # L2 reg term
                dLdc2 = np.sum(dLdA3, axis=0)
                dLdh2 = np.dot(dLdA3, W2.T)
                
                dLdA2 = np.multiply(dLdh2, tanh_backward(A2))
                dLdW2_in = np.dot(dLdA2.T, h1) + lambda_reg * W2  # L2 reg term
                dLdb2 = np.sum(dLdA2, axis=0)
                dLdh1 = np.dot(dLdA2, W2)
                
                dLdA1 = np.multiply(dLdh1, tanh_backward(A1))
                dLdW1_in = np.dot(dLdA1.T, x) + lambda_reg * W1  # L2 reg term
                dLdb1 = np.sum(dLdA1, axis=0)
                
                # Update weights with momentum
                W1 += -LEARNING_RATE * (dLdW1_in + beta * dW10)
                W2 += -LEARNING_RATE * (dLdW2_in + beta * dW20)
                b1 += -LEARNING_RATE * (dLdb1 + beta * db10)
                b2 += -LEARNING_RATE * (dLdb2 + beta * db20)
                c1 += -LEARNING_RATE * dLdc1
                c2 += -LEARNING_RATE * dLdc2
                
                dW10 = dLdW1_in
                dW20 = dLdW2_in
                db10 = dLdb1
                db20 = dLdb2
            
            loss = getLoss(W1, W2, xtrain, u, b1, b2, c1, c2, accList, lambda_reg)
            print(i, loss)
    
    return W1, W2, b1, b2, c1, c2

#متغیر lambda_reg به عنوان ضریب L2 Regularization اضافه شده است.
در محاسبه گرادیان‌ها (dLdW1_out, dLdW1_in, dLdW2_out, dLdW2_in)، عبارت + lambda_reg * W اضافه شده است تا L2 Regularization اعمال شود.
در محاسبه تابع هزینه (getLoss)، مقدار L2 Regularization نیز به صورت 
𝜆 2 ∑ 𝑊 2 2 λ ∑W 2
  به meanLoss اضافه شده است.
