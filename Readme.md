<h1 align="center">House Price Prediction</h1>

<div align= "center"><img src="assets/home.png" width="250" height="250"/> 

<br/>

  <h4>This is a project in which we are going to create a machine learning model to make prediction of a digit which is writting by hand.</h4>
</div>

<br/>

# Table of Contents 

- [Table of Contents](#table-of-contents)
- [:warning: Frameworks and Libraries](#warning-frameworks-and-libraries)
- [:file_folder: Datasets](#file_folder-datasets)
  - [🔄 Source](#-source)
  - [📈 Visualising data](#-visualising-data)
  - [🔥 Probability Measure](#-probability-measure)
  - [🚇 Cross-Entropy](#-cross-entropy)
- [:book: Data Preprocessing](#book-data-preprocessing)
- [:link: Download](#link-download)
- [:key: Prerequisites](#key-prerequisites)
- [🚀&nbsp; Installation](#-installation)
- [:bulb: How to Run](#bulb-how-to-run)
- [📂 Directory Tree](#-directory-tree)
- [:key: Results](#key-results)
- [:clap: And it's done!](#clap-and-its-done)
- [:raising_hand: Citation](#raising_hand-citation)
- [:heart: Owner](#heart-owner)
- [:eyes: License](#eyes-license)

<br/>

# :warning: Frameworks and Libraries

- **[Pytorch](https://pytorch.org/):** An open source machine learning framework that accelerates the path from research prototyping to production deployment.
- **[Matplotlib](https://matplotlib.org/) :** Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
- **[Numpy](https://numpy.org/):**
  Caffe-based Single Shot-Multibox Detector (SSD) model used to detect faces
- **[Pandas](https://pandas.pydata.org/):**
  pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.

<br/>

# :file_folder: Datasets

## 🔄 Source

This dataset is a builtin dataset in pytorch MNIST dataset available from [MNIST database;s page](http://yann.lecun.com/exdb/mnist/). The dataset may also be downloaded directly by importing `torchvision.dataset`.

<br/>

## 📈 Visualising data

<p align="center">
  <img src="./assets/digit5.png" />
</p>

<br/>

## 🔥 Probability Measure

To convert the output rows into probabilities, we use the softmax function, which has the following formula:

**Equation of Softmax:**

$$\Large S(y_i) = \frac{e^{y_i}}{\sum_{j} e^{y_j}}$$

<br/>

<p align="center">
  <img src="./assets/softmax.png" />
</p>

**Graph of Softmax:**

<br/>

<p align="center">
  <img src="./assets/softmax-graph.png" />
</p>

<br/>

## 🚇 Cross-Entropy

Accuracy is often used as an evaluation metric for classification, but not as a loss function. A commonly used loss function for classification problems is the cross-entropy, which has the following formula:

**Equation of Cross Entropy:**

$$\Large D(\hat{y}, y) = -\sum_{j} y_j \ln{\hat{y}_j}$$

<br/>

<p align="center">
  <img src="./assets/entropy.png" />
</p>

**Graph of Cross Entropy:**

<br/>

<p align="center">
  <img src="./assets/entropy-graph.png" />
</p>

<br/>

# :book: Data Preprocessing

Data pre-processing is an important step for the creation of a machine learning
model. Initially, data may not be clean or in the required format for the model which
can cause misleading outcomes. In pre-processing of data, we transform data into our
required format. It is used to deal with noises, duplicates, and missing values of the
dataset. Data pre-processing has the activities like importing datasets, splitting
datasets, attribute scaling, etc. Preprocessing of data is required for improving the
accuracy of the model.

<br/>

# :link: Download

The dataset is now available [here](http://yann.lecun.com/exdb/mnist/) !

<br/>

# :key: Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](requirements.txt)

<br/>

# 🚀&nbsp; Installation

The Code is written in Python 3.7. If you don&rsquo;t have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

1. Clone the repo

```bash
git clone https://github.com/Chaganti-Reddy/Hand-Digit-Recognition.git
```

2. Change your directory to the cloned repo

```bash
cd Hand-Digit-Recognition
```

Before running the command copy the downloaded dataset folder to face-mask-detector folder...

3. Now, run the following command in your Terminal/Command Prompt to install the libraries required

```bash
python3 -m virtualenv my_env

source my_env/bin/activate

pip3 install -r requirements.txt

```

<br/>

# :bulb: How to Run

1. Open terminal. Go into the cloned project directory and type the following command:

```bash
python3 Digit-Recognition.py
```

<br/>

# 📂 Directory Tree

```
├── assets
│   ├── digit5.png
│   ├── entropy.png
│   ├── graph.png
│   ├── home.png
│   ├── softmax-graph.png
│   └── softmax.png
├── data
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
├── Digit-Recognition.ipynb
├── Digit-Recognition.py
├── LICENSE
├── mnist-logistic.pth
├── Readme.md
└── requirements.txt
```

<br/>

# :key: Results

 **Our Classification Model has successfully predicted the output with an excellent accuracy via <code>PyTorch</code>**

<br/>

<p align="center">
  <img src="./assets/digit7.png" />
</p>

<br/>

# :clap: And it's done!

Feel free to mail me for any doubts/query
:email: chagantivenkataramireddy1@gmail.comv

---

# :raising_hand: Citation

You are allowed to cite any part of the code or our dataset. You can use it in your Research Work or Project. Remember to provide credit to the Maintainer Chaganti Reddy by mentioning a link to this repository and her GitHub Profile.

Follow this format:

- Author's name - Chaganti Reddy
- Date of publication or update in parentheses.
- Title or description of document.
- URL.

# :heart: Owner

Made with :heart:&nbsp; by [Chaganti Reddy](https://github.com/Chaganti-Reddy/)

# :eyes: License

MIT © [Chaganti Reddy](https://github.com/Chaganti-Reddy/Hand-Digit-Recognition/blob/main/LICENSE)
