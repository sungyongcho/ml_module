import matplotlib.pyplot as plt
import pandas as pd
import argparse
from data_spliter import data_spliter
import numpy as np
from my_logistic_regression import MyLogisticRegression

# Create an argument parser
parser = argparse.ArgumentParser(description='Mono Logistic Regression')

# Add the --zipcode argument
parser.add_argument('--zipcode', type=int,
                    choices=[0, 1, 2, 3], help='Favorite planet\'s zipcode (0, 1, 2, or 3)')

# Parse the command-line arguments
args = parser.parse_args()

# Check if --zipcode argument is provided
if args.zipcode is None:
    parser.print_usage()
    exit(1)

# Load the datasets
census_data = pd.read_csv('solar_system_census.csv')
planet_data = pd.read_csv('solar_system_census_planets.csv')

# Drop the "Unnamed: 0" column from census_data and planet_data
census_data.drop("Unnamed: 0", axis=1, inplace=True)
planet_data.drop("Unnamed: 0", axis=1, inplace=True)

# Merge the datasets based on the index column
merged_data = pd.merge(census_data, planet_data,
                       left_index=True, right_index=True)

print(merged_data.head())

X = merged_data.drop('Origin', axis=1).values
y = merged_data['Origin'].values

y = (y == args.zipcode).astype(int)

y = np.reshape(y, (-1, 1))

# print(y)

# X_train, X_test, y_train, y_test = data_spliter(X, y, proportion=0.8)
X_train, X_test, y_train, y_test = data_spliter(X, y, proportion=0.8, fix='y')

# Create an instance of MyLogisticRegression
logistic_regression = MyLogisticRegression(
    theta=np.random.rand(X_train.shape[1] + 1, 1), alpha=1e-3, max_iter=10000)

# Train the logistic regression model
theta = logistic_regression.fit_(X_train, y_train)

print("Theta:", theta.flatten())

# # Make predictions on the test set
y_pred = logistic_regression.predict_(X_test)
y_pred_binary = (y_pred >= 0.5).astype(int).flatten()
y_pred_binary = np.reshape(y_pred_binary, (-1, 1))

print(y_pred_binary)

print("=======", y_pred_binary, "========")

# # Calculate the fraction of correct predictions
accuracy = np.mean(y_pred_binary == y_test)

# Display the fraction of correct predictions
print(f"Fraction of correct predictions: {accuracy}")


# Create a figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Find the column indices of 'Height', 'Weight', and 'Bone Density'
height_column_index = merged_data.columns.get_loc('height')
weight_column_index = merged_data.columns.get_loc('weight')
bone_density_column_index = merged_data.columns.get_loc('bone_density')

# Scatter plot 1: Height vs. Origin
axes[0].scatter(X_test[:, height_column_index], y_test,
                color='blue', marker='o', label='Actual')
axes[0].scatter(X_test[:, height_column_index], y_pred_binary, color='orange',
                marker='x', label='Prediction')
axes[0].set_xlabel('Height')
axes[0].set_ylabel('Origin')
axes[0].set_title('Height vs. Origin - Actual vs. Prediction')
axes[0].legend()

# Scatter plot 2: Weight vs. Origin
axes[1].scatter(X_test[:, weight_column_index], y_test,
                color='blue', marker='o', label='Actual')
axes[1].scatter(X_test[:, weight_column_index], y_pred_binary, color='orange',
                marker='x', label='Prediction')
axes[1].set_xlabel('Weight')
axes[1].set_ylabel('Origin')
axes[1].set_title('Weight vs. Origin - Actual vs. Prediction')
axes[1].legend()

# Scatter plot 3: Bone Density vs. Origin
axes[2].scatter(X_test[:, bone_density_column_index], y_test,
                color='blue', marker='o', label='Actual')
axes[2].scatter(X_test[:, bone_density_column_index], y_pred_binary, color='orange',
                marker='x', label='Prediction')
axes[2].set_xlabel('Bone Density')
axes[2].set_ylabel('Origin')
axes[2].set_title('Bone Density vs. Origin - Actual vs. Prediction')
axes[2].legend()

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()
