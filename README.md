# Fashion-MNIST Image Classifier

This project benchmarks four machine learning models on the **Fashion-MNIST dataset**:
- Logistic Regression
- Random Forest
- Multi-layer Perceptron (MLP)
- Convolutional Neural Network (CNN, PyTorch)

The goal is to compare classical ML and neural network approaches on a standard computer vision dataset.

---

## Dataset

This project uses the **Fashion-MNIST dataset** created by Zalando Research.  
The dataset is available on [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist).  

Please download the following CSV files from Kaggle and place them in the project root directory:
- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

These files are too large to include directly in this repository, so they must be downloaded separately.

---

## CNN Experiment

In addition to the classical models, a **Convolutional Neural Network (CNN)** was implemented in **PyTorch** to better capture spatial patterns in the images.

- Architecture: convolutional + pooling layers, followed by fully connected layers  
- Training achieved **92–94% validation accuracy**  
- Model evaluation included loss/accuracy curves and confusion matrix analysis for class-wise errors  

This demonstrates the improvement possible with deep learning compared to traditional ML baselines.
