import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from my_logistic_regression import MyLogisticRegression
from other_metrics import f1_score_
from polynomial_model_extended import add_polynomial_features
from data_spliter import data_spliter
import pickle
from benchmark_train import benchmark_train


if __name__ == "__main__":

    # Load the datasets
    census_data = pd.read_csv('solar_system_census.csv')
    planet_data = pd.read_csv('solar_system_census_planets.csv')

    # Drop the "Unnamed: 0" column from census_data and planet_data
    census_data.drop("Unnamed: 0", axis=1, inplace=True)
    planet_data.drop("Unnamed: 0", axis=1, inplace=True)

    # Merge the datasets based on the index column
    merged_data = pd.merge(census_data, planet_data,
                           left_index=True, right_index=True)

    # Define the feature matrix X and target variable y
    X = merged_data.drop('Origin', axis=1).values
    y = merged_data['Origin'].values.astype(int)

    # Train the models using benchmark_train
    models = benchmark_train(X, y)

    # Find the model with the best f1 score
    best_lambda_val, (best_model, _) = max(
        models.items(), key=lambda x: x[1][1])

    # Visualize the performance of the different models
    lambda_values = np.array(list(models.keys()))
    f1_scores = np.array([score for _, score in models.values()])

    plt.bar(lambda_values, f1_scores)
    plt.xlabel('Lambda Values')
    plt.ylabel('F1 Score')
    plt.title('Performance of Different Models')
    plt.show()

    # Plot scatter plots
    feature_names = merged_data.drop('Origin', axis=1).columns

    # Your data and feature_names setup
    # ...
    num_features = len(feature_names)
    total_plots = num_features * (num_features - 1) // 2

    # Create a figure with custom size.
    plt.figure(figsize=(5 * total_plots, 5))

    subplot_idx = 1
    for i in range(num_features - 1):
        for j in range(i + 1, num_features):
            ax = plt.subplot(1, total_plots, subplot_idx)

            for class_label in np.unique(y):
                class_data = merged_data[merged_data['Origin'] == class_label]
                ax.scatter(class_data[feature_names[i]],
                           class_data[feature_names[j]], label=f"Class {class_label}")

            ax.set_xlabel(feature_names[i])
            ax.set_ylabel(feature_names[j])
            ax.legend()
            subplot_idx += 1

    # Display all subplots.
    plt.tight_layout()
    plt.show()
