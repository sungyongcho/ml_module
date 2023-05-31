import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from my_logistic_regression import MyLogisticRegression
from other_metrics import f1_score_
from polynomial_model_extended import add_polynomial_features
from data_spliter import data_spliter
import pickle


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

# Transform the features into a polynomial of degree 3
X_poly = add_polynomial_features(X, power=3)

# Split the dataset into a training, cross-validation, and test set
X_train, X_cv, X_test, y_train, y_cv, y_test = data_spliter(
    X, y, 0.6, 0.2, fix='y')
# y_train, y_cv, y_test = np.split(y, [int(0.6 * len(y)), int(0.8 * len(y))])

# Train different regularized logistic regression models with a polynomial hypothesis of degree 3
lambda_values = np.linspace(0, 1, 10)  # Range of lambda values
models = {}  # Dictionary to store the models and their f1 scores

for lambda_val in lambda_values:
    X_train_poly = add_polynomial_features(X_train, power=3)
    X_cv_poly = add_polynomial_features(X_cv, power=3)

    # Create an instance of your logistic regression model with regularization parameter lambda_val
    model = MyLogisticRegression(theta=np.random.rand(X_train_poly.shape[1] + 1, 1),
                                 alpha=0.001, max_iter=1000, lambda_=lambda_val)

    # Train the model on the training set
    model.fit_(X_train_poly, y_train)

    # Evaluate the model on the cross-validation set
    y_cv_pred = model.predict_(X_cv_poly)
    f1 = f1_score_(y_cv, y_cv_pred)

    # Save the model with its f1 score
    models[lambda_val] = (model, f1)

    # Calculate the F1 score on the test set for the current model
    X_test_poly = add_polynomial_features(X_test, power=3)
    y_test_pred = model.predict_(X_test_poly)
    f1_test = f1_score_(y_test, y_test_pred)
    print(f"Lambda: {lambda_val}, F1 Score: {f1}, Test F1 Score: {f1_test}")

# Find the model with the best f1 score
best_lambda_val, (best_model, _) = max(models.items(), key=lambda x: x[1][1])

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
