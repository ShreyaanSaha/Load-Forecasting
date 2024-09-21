# Power System Load Forecasting Automation using Artificial Neural Network

This project focuses on automating power system load forecasting using advanced machine learning techniques. The primary goal is to develop a robust forecasting model to predict future power loads accurately, which is crucial for efficient energy management and planning.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Optimization](#training-and-optimization)
- [Performance Evaluation](#performance-evaluation)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Load forecasting is a critical component of power system planning and operation. Accurate forecasts enable utilities to make informed decisions regarding generation, transmission, and distribution. This project leverages Artificial Neural Networks (ANN) to create a reliable load forecasting model.

## Dataset

The dataset used in this project contains historical power load data. It includes various features that influence power consumption, such as:

- Date and time
- Weather conditions (temperature, humidity, etc.)
- Historical load data

## Data Preprocessing

### Outlier Removal

Outliers in the dataset were identified and removed using two techniques:

1. **Boxplot Analysis**: Visual inspection to detect and remove outliers.
2. **Z-score Method**: Statistical method to remove data points that are more than three standard deviations away from the mean.

### Data Scaling

Data was standardized to have a mean of 0 and a standard deviation of 1. This was achieved using the following formula:

\[ Z = \frac{(X - \mu)}{\sigma} \]

Where:

- \(X\) is the original data point
- \(\mu\) is the mean of the data
- \(\sigma\) is the standard deviation

## Model Architecture

The ANN model developed for this project consists of four layers, each containing 100 neurons. The architecture details are as follows:

- **Activation Function**: Scaled Exponential Linear Unit (SELU)
- **Regularization**: L2 regularization to prevent overfitting
- **Weight Initialization**: He normal initialization for efficient training
- **Optimizer**: Adam optimizer for adaptive learning rate

### Model Summary

- **Input Layer**: Standardized feature set
- **Hidden Layers**: 4 layers with 100 neurons each, SELU activation
- **Output Layer**: 1 neuron for load prediction

## Training and Optimization

The devloped model was trained using the Adam optimizer, which combines the best properties of the AdaGrad and RMSProp algorithms to provide an adaptive learning rate. The objective was to minimize the Mean Squared Error (MSE) between the predicted and actual load values.

## Performance Evaluation

The model's performance was evaluated using the R² score, which measures the proportion of variance in the dependent variable predictable from the independent variables. The model achieved an R² score of **0.9584** on the testing dataset, indicating high predictive accuracy.

## Conclusion

The developed ANN model successfully automates power system load forecasting with high accuracy. The preprocessing steps, including outlier removal and data scaling, significantly contributed to the model's performance. This project demonstrates the potential of machine learning techniques in enhancing power system operations.

## References

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Understanding the Box Plot](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51)
- [SELU Activation Function](https://arxiv.org/abs/1706.02515)

---
