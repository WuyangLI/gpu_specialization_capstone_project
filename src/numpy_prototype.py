import numpy as np
from math import sqrt
from numpy import mean
from numpy.random import rand
import struct
from array import array
from os.path import join

# load MNIST dataset from file
def read_images_labels(images_filepath, labels_filepath, sample=True):        
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels_data = array("B", file.read())
        labels = []

        
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        
        images = []

    for i in range(size):
        label = int(labels_data[i])
        if sample and i % 10 != 0:
            continue
        labels.append([0.0] * label + [1.0] + [0.0] * (9 - label))   
        img = image_data[i * rows * cols:(i + 1) * rows * cols]
        images.append(img)            

    print(f"images total number {len(images)}")
    print(f"images dim {len(images[0])}")
    print(f"labels total number {len(labels)}")
    print(f"labels dim {len(labels[0])}")
    # Save the array to a text file
    def save_to_file(fp, array_to_save):
        with open(fp, 'w') as file:
            for row in array_to_save:
                file.write(' '.join(map(str, row)) + '\n')
            
    save_to_file(f"{images_filepath}_sample_{sample}.txt", images)
    save_to_file(f"{labels_filepath}_sample_{sample}.txt", labels)
    return images, labels

# initialize model weights
def xavier_init(dim1, dim2):
    # number of nodes in the previous layer
    lower, upper = -(1.0 / sqrt(dim1)), (1.0 / sqrt(dim1))
    # generate random numbers
    random_weights = rand(dim1, dim2)
    # scale to the desired range
    return lower + random_weights * (upper - lower)


## forward pass
def relu(x):
    return np.maximum(0, x)

def softmax(x, axis=-1):
    # Ensure numerical stability by subtracting the maximum value
    # of each row (for matrices) or the entire array (for vectors)
    # from each element.
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cross_entropy(y, p):
    # Ensure numerical stability by adding a small epsilon value
    epsilon = 1e-15
    # Apply the cross-entropy formula element-wise
    cross_entropy = -y * np.log(p + epsilon)
    # Calculate the mean cross-entropy across all elements
    mean_cross_entropy = np.mean(cross_entropy)
    return mean_cross_entropy

def forward_pass(x, y, w1, w2):
    s = np.dot(x, w1)
    z = relu(s)
    f = np.dot(z, w2)
    p = softmax(f)
    loss = cross_entropy(y, p)
    print("loss is: ", "{:.3f}".format(loss))
    return s, z, f, p

## backward pass
def backward_pass(x, s, z, f, p, y, w1, w2, lr):
    epsilon = 1e-15
    dp = - np.divide(y, p + epsilon)
    # derivative of sofmax(v) w.r.t v is sofmax(v) - sofmax(v)*sofmax(v)
    df = np.multiply(dp, p - np.multiply(p, p))
    dw2 = np.dot(np.transpose(z), df)
    dz = np.dot(df, np.transpose(w2))
    ds = np.multiply(dz, (s > 0).astype(float))
    dw1 = np.dot(np.transpose(x), ds)
    w2 = w2 - lr * np.clip(dw2, -1, 1)
    w1 = w1 - lr * np.clip(dw1, -1, 1)
    return w1, w2

def train_model(x, y, w1, w2, epoch_num, lr):
    for i in range(epoch_num):
        s, z, f, p = forward_pass(x, y, w1, w2)
        w1, w2 = backward_pass(x, s, z, f, p, y, w1, w2, lr)
    return w1, w2

def proof_of_concept():
    """
    this is the python prototype of training a two-layer MLP classifier for MNIST
    the model architecture is as follows:
        MLP W1
           |
    relu activation
           |
        MLP W2
           |
        softmax

    in algebra, the model prediction P is a function of input X and weights W1, W2, depicted as below:
        P = softmax(relu(X * W1) * W2)
    """
    batch_size = 6000
    test_batch_size = 1000
    pixel_len = 28 * 28
    hidden_size = 512
    class_num = 10

    # load train and test data
    train_img, train_label = read_images_labels('../data/archive/train-images.idx3-ubyte', '../data/archive/train-labels.idx1-ubyte')
    test_img, test_label = read_images_labels('../data/archive/t10k-images.idx3-ubyte', '../data/archive/t10k-labels.idx1-ubyte')

    # initialize model - two layer MLP
    w1 = xavier_init(pixel_len, hidden_size)
    w2 = xavier_init(hidden_size, class_num)
    w1, w2 = train_model(np.array(train_img), np.array(train_label), w1, w2, 10, 0.001)

    # test trained model
    _, _, _, test_p = forward_pass(np.array(test_img), np.array(test_label), w1, w2)
    test_prediction_idx = np.argmax(test_p, axis=1)
    test_groundtruth_idx = np.argmax(np.array(test_label), axis=1)
    accuracy = np.sum((test_prediction_idx == test_groundtruth_idx).astype(float)) / len(test_prediction_idx)
    print("accuracy of the model is: ", "{:.2f} %".format(accuracy*100.0))