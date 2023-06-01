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

    merged_data['Origin'] = merged_data['Origin'].astype(int)

    # Define the feature matrix X and target variable y
    X = merged_data.drop('Origin', axis=1).values
    y = merged_data['Origin'].values.astype(int)
    degrees = [0, 1, 2, 3]

    # Train the models using benchmark_train
    models = benchmark_train(X, y, degrees)

    # Load the trained models
    with open('models.pickle', 'rb') as f:
        models = pickle.load(f)

    # Extract the lambda values and corresponding F1 scores for each degree
    lambda_values = {}
    f1_scores = {}


    for degree in degrees:
        lambda_values[degree] = []
        f1_scores[degree] = []
        for key, value in models.items():
            if key[1] == degree:
                lambda_values[degree].append(key[2])
                f1_scores[degree].append(value)

    # Create subplots for each degree
    fig, axes = plt.subplots(1, len(degrees), figsize=(15, 6))

    # Iterate over degrees
    for i, degree in enumerate(degrees):
        axes[i].bar(lambda_values[degree], f1_scores[degree])
        axes[i].set_xlabel('Lambda')
        axes[i].set_ylabel('F1 Score')
        axes[i].set_title(f'Performance of Degree {degree} Models')

    plt.tight_layout()
    plt.show()

    # Get the best degree and lambda value
    best_model_key = max(models.keys(), key=lambda x: models[x])

    # Get the best model and its F1 score
    best_model = best_model_key[0]
    best_degree = best_model_key[1]
    best_lambda = best_model_key[2]
    best_f1_score = models[best_model_key]

    # Get the best model and its coefficients
    coefficients = best_model.theta[1:]

    X_poly = add_polynomial_features(X, power=best_degree)

    X_train, X_test, y_train, y_test = data_spliter(
        X_poly, y, train_proportion=0.2)

    # Create a binary target variable for the best class
    binary_target = (y_train == best_degree).astype(int)

    # Create an instance of your logistic regression model with the obtained coefficients
    model = MyLogisticRegression(
        theta=np.insert(
            coefficients, 0, best_model.theta[0]).reshape(-1, 1)[:X_test.shape[1]+1],
        alpha=0.001,
        max_iter=1000,
        lambda_=best_lambda
    )

    # Train the model on the polynomial feature matrix
    model.fit_(X_train, binary_target)

    # Predict the target values using the best model
    y_pred = model.predict_(X_test)

    # Plot scatter plots
    feature_names = merged_data.drop('Origin', axis=1).columns

    # Calculate the number of features and total number of plots
    num_features = len(feature_names)
    total_plots = num_features * (num_features - 1) // 2

    # Create a figure with custom size
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

    # Display all subplots
    plt.tight_layout()
    plt.show()
