
# Stock Price Prediction using Convolutional Neural Networks (CNN) and Feature Extraction

![License](https://img.shields.io/badge/License-MIT-blue.svg)

## Overview

This project focuses on predicting stock prices using Convolutional Neural Networks (CNN) and feature extraction. The dataset used is a significant stock market dataset. This README provides a summary of the project and its key components.

## Table of Contents

- [Dataset Preprocessing](#dataset-preprocessing)
- [Data Visualization](#data-visualization)
- [Feature Extraction](#feature-extraction)
- [Splitting the Dataset](#splitting-the-dataset)
- [Building the Model](#building-the-model)
- [Model Compilation](#model-compilation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Performance Plots](#model-performance-plots)
- [Correlation Matrix](#correlation-matrix)
- [Additional Visualizations](#additional-visualizations)
- [Rolling Average Plot](#rolling-average-plot)
- [Benchmark Models for Comparison](#benchmark-models-for-comparison)
- [Models Evaluation Metrics Against Benchmark Models](#models-evaluation-metrics-against-benchmark-models)
- [Project Summary](#project-summary)

## Overview

This project focuses on predicting stock prices using Convolutional Neural Networks (CNN) and feature extraction. The dataset used is a significant stock market dataset. This README provides a summary of the project and its key components.

## Dataset Preprocessing

The dataset preprocessing involves several steps:

- Loading the data from CSV.
- Checking for missing values and removing rows with missing data.
- Dropping irrelevant columns.
- Sorting the dataset by date in ascending order.

## Data Visualization

A scatter plot is used to illustrate the relationship between the volume of stock trades and the mid price.

## Feature Extraction

In the feature extraction step, relevant information is extracted from the stock dataset. Features are normalized for input to the CNN model.

## Splitting the Dataset

The dataset is split into input (X) and output (y) variables. Data is also reshaped for CNN.

## Building the Model

The CNN model is built using Keras.

## Model Compilation

The model is compiled using the Adam optimizer and mean squared error loss.

## Model Training

The model is trained on the training data.

## Model Evaluation

The model is evaluated using test data, and evaluation metrics are calculated.

## Model Performance Plots

Loss and accuracy plots for model performance are generated.

## Correlation Matrix

A correlation matrix is plotted to visualize the correlations between different features.

## Additional Visualizations

Additional visualizations include volume analysis and a histogram to visualize the distribution of stock prices.

## Rolling Average Plot

A rolling average plot shows the historical stock prices and their corresponding rolling average.

## Benchmark Models for Comparison

This section discusses benchmark models, including Support Vector Regression (SVR), Recurrent Neural Networks (RNN) with LSTM, and ARIMA (Autoregressive Integrated Moving Average).

## Models Evaluation Metrics Against Benchmark Models

This section provides evaluation metrics for the CNN model and benchmark models to compare their performance.

## Project Summary

- Dataset: Huge Stock Market DataSet
- Goal: Predict stock prices using deep learning
- Methodology: Used a deep learning model and trained it on the dataset
- Key Findings: Achieved a high test accuracy of 96.63% and a low test loss of 0.00023

## License

This project is licensed under the MIT License.
