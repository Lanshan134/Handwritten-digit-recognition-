import os
import urllib.request
import gzip
import numpy as np

def load_mnist():
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = {
        'train_images':'train-images-idx3-ubyte.gz',
        'train_labels':'train-labels-idx1-ubyte.gz',
        'test_images':'t10k-images-idx3-ubyte.gz',
        'test_labels':'t10k-labels-idx1-ubyte.gz'
    }

    os.makedirs('data', exist_ok=True)

    def download_and_parse_mnist_file(filename):
        filepath = os.path.join('data', filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        with gzip.open(filepath, 'rb') as f:
            if 'images' in filename:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
                data = data.reshape(-1, 28*28) / 255.0
            else:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            return data

    X_train = download_and_parse_mnist_file(files['train_images'])
    y_train = download_and_parse_mnist_file(files['train_labels'])
    X_test = download_and_parse_mnist_file(files['test_images'])
    y_test = download_and_parse_mnist_file(files['test_labels'])

    return X_train, y_train, X_test, y_test

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_mnist()
    print('MNIST dataset loaded:')
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)
