import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = x_train[0:1000].reshape(1000, 28*28) / 255
labels = y_train[0:1000]
one_hot_encode = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_encode[i][l] = 1
labels = one_hot_encode

t_images = x_test[0:1000].reshape(1000, 28*28) / 255
t_labels = np.zeros((len(y_test), 10))

np.random.seed(1)
relu = lambda x: (x >= 0)*x
reluderivative = lambda x: x >= 0

alpha = 0.005
iterations = 360
px_per_img = 784
hidden_size = 40
n_labels = 10


weights_0 = 0.2 * np.random.random((px_per_img, hidden_size)) - 0.1
weights_1 = 0.2 * np.random.random((hidden_size, n_labels)) - 0.1

for j in range(iterations):
    error, correct_count = (0.0, 0)
    for i in range(px_per_img):
        layer_0 = images[i: i+1]
        
        layer_1 = relu(np.dot(layer_0, weights_0))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2  # The dropout mask is multiplied by 2, such that the layer_2 doesn't increase its sensitivity to layer_1. The reason i chose 2, is because 1/2 of the values are 0. -> ( 1 / (1/2) = 2 )
        
        layer_2 =np.dot(layer_1, weights_1) 
                       
        error += np.sum((labels[i: i+1] - layer_2)**2)
        correct_count += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        layer2_delta = (labels[i: i+1] - layer_2)
        
        layer1_delta = layer2_delta.dot(weights_1.T) * reluderivative(layer_1) 
        layer1_delta *= dropout_mask

        weights_0 += alpha * layer_0.T.dot(layer1_delta)
        weights_1 += alpha * layer_1.T.dot(layer2_delta)

# Every 10 iterations, we test the neural network. We then print the results of the training and the tests simultanously.
    if j%10 == 0: 
        t_error, t_correct_count = (0.0, 0)
        for i in range(len(t_images)):
            layer_0 = t_images[i: i+1]
            layer_1 = relu(np.dot(layer_0, weights_0)) 
            layer_2 =np.dot(layer_1, weights_1) 
                       
            t_error += np.sum((t_labels[i: i+1] - layer_2)**2)
            t_correct_count += int(np.argmax(t_labels[i: i+1]) == np.argmax(layer_2))



    
        sys.stdout.write("\n"+ \
                         " I:" + str(j) + \
                         " Test-Error:" + str(t_error/ float(len(t_images))) [0:5] +\
                         " Test-Accuracy:" + str(t_correct_count/ float(len(t_images))) +\
                         " Train-Error:" + str(error / float(len(images)))[0:5] + \
                         " Train-Accuracy:" + str(correct_count / float(len(images)))
                        )
    
