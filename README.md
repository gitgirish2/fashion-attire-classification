# Attire Recognition
This code helps you classify different clothing attires.

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Here's an example how the data looks (each class takes three-rows):

![](data/fashion_mnist.png)

<img src="data/embedding.gif" width="100%">


## To Serious Machine Learning Researchers

Seriously, we are talking about replacing MNIST. Here are some good reasons:

+ MNIST is too easy. Convolutional nets can achieve 99.7% on MNIST. Classic machine learning algorithms can also achieve 97% easily. Check out our side-by-side benchmark for Fashion-MNIST vs. MNIST, and read "Most pairs of MNIST digits can be distinguished pretty well by just one pixel."
+ MNIST is overused. In this April 2017 Twitter thread, Google Brain research scientist and deep learning expert Ian Goodfellow calls for people to move away from MNIST.
+ MNIST can not represent modern CV tasks, as noted in this April 2017 Twitter thread, deep learning expert/Keras author François Chollet.


## Labels

Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |


## Usage

### Loading data with Python (requires [NumPy](http://www.numpy.org/))

Use `utils/mnist_reader` from this repo:
```python
import mnist_reader

X_train, y_train = mnist_reader.load_mnist("__data__/fashion", kind="train")
X_test, y_test = mnist_reader.load_mnist("__data__/fashion", kind="t10k")
```

### Loading data with Tensorflow
Make sure you have downloaded the data and placed it in `__data__/fashion`. Otherwise, *Tensorflow will download and use the original MNIST.*

```python
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("__data__/fashion")

data.train.next_batch(BATCH_SIZE)
```

Note, Tensorflow supports passing in a source url to the `read_data_sets`. You may use: 
```python
data = input_data.read_data_sets("__data__/fashion", source_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/")
```


### Other Explorations of Fashion-MNIST
+ [Fashion-MNIST: Year in Review](https://hanxiao.github.io/2018/09/28/Fashion-MNIST-Year-In-Review/)
+ [Fashion-MNIST on Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=fashion-mnist&btnG=&oq=fas) 
