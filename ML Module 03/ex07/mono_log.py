from my_logistic_regression import MyLogisticRegression as MyLR
from data_spliter import data_spliter
import numpy as np
import argparse
import pandas as pd


# Read the solar_system_census.csv file
census_data = np.genfromtxt('solar_system_census.csv', delimiter=',')

# Read the solar_system_census_planets.csv file
planets_data = np.genfromtxt('solar_system_census_planets.csv', delimiter=',')

print(census_data.shape)
print(planets_data.shape)

# Merge the two datasets based on a common index or column
merged_data = np.hstack((census_data, planets_data))

# Display the merged dataset
print(merged_data)
# Display the dataset (optional)


# Create an argument parser
parser = argparse.ArgumentParser(
    description='Logistic Regression Classifier for Solar System Census')

# Add the --zipcode argument
parser.add_argument('--zipcode', type=int,
                    choices=[0, 1, 2, 3], help='Specify the favorite planet (0, 1, 2, or 3)')

# Parse the command-line arguments
args = parser.parse_args()

# Check if the --zipcode argument is provided
if args.zipcode is None:
    parser.print_usage()
    exit(1)

# Access the chosen zipcode
chosen_zipcode = float(args.zipcode)

# Print the chosen zipcode (optional)
print('Chosen Zipcode:', chosen_zipcode)


# Extract features and labels
# Extract features and labels
features = merged_data[:, :-1]
labels = merged_data[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = data_spliter(
    features, labels, proportion=0.8)

# Display the shapes of the training and test sets (optional)
print('Training set shape:', X_train.shape, y_train.shape)
print('Test set shape:', X_test.shape, y_test.shape)


# Replace <number_of_features> with the actual number of features in your dataset
n = len(features)

# Initialize the theta values with zeros
thetas = np.zeros((n + 1, 1))  # Add 1 for the bias term

# Create an instance of MyLogisticRegression with the initial theta values

lr_model = MyLR(theta=thetas)

print(X_train, y_train)

print(lr_model.fit_(X_train, y_train))
