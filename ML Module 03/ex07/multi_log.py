import matplotlib.pyplot as plt
import pandas as pd
from data_spliter import data_spliter
import numpy as np
from my_logistic_regression import MyLogisticRegression

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

# Define the feature matrix X and target variable y
X = merged_data.drop('Origin', axis=1).values
y = merged_data['Origin'].values.astype(int)

# Split the dataset into a training and a test set
X_train, X_test, y_train, y_test = data_spliter(X, y, proportion=0.8, fix='y')

# Train logistic regression classifiers for each class
class_predictions = []
class_accuracy = []

unique_classes = np.unique(y_train).astype(int)
print(unique_classes)

for i in range(4):
    # Create a binary target variable for the current class
    binary_target = (y_train == i).astype(int)
    binary_target = np.reshape(binary_target, (-1, 1))

    # Create an instance of MyLogisticRegression
    logistic_regression = MyLogisticRegression(
        theta=np.random.rand(X_train.shape[1] + 1, 1), alpha=1e-2, max_iter=100000)

    # Train the logistic regression model
    theta = logistic_regression.fit_(X_train, binary_target)

    print("Theta:", theta.flatten())

    # Make predictions on the test set
    y_pred = logistic_regression.predict_(X_test)
    y_pred_binary = (y_pred >= 0.5).astype(int).flatten()
    y_pred_binary = np.reshape(y_pred_binary, (-1, 1))
    # print(y_pred)

    actual_binary = (y_test == i).astype(int)
    actual_binary = np.reshape(actual_binary, (-1, 1))
    # actual_binary = (y_test == i).astype(int)
    # print(y_pred_binary)
    # print(actual_binary)

    # Append the binary class labels to the class_predictions array
    class_predictions.append(y_pred_binary)

    # Calculate the accuracy for the current class
    accuracy = np.mean(y_pred_binary == actual_binary)

    # Append the accuracy to the class_accuracy array
    class_accuracy.append(accuracy)

# print(class_predictions)
# print(class_accuracy)

# # Combine the predictions for each class into a single array
# class_predictions = np.array(class_predictions)

# Select the class with the highest count for each example
predicted_classes = np.argmax(class_predictions, axis=0)
predicted_classes = predicted_classes.flatten()

# Calculate the overall accuracy by comparing the predicted classes with the actual labels
overall_accuracy = np.mean(predicted_classes == y_test)
print("Overall Accuracy:", overall_accuracy)

# Display the accuracy for each class
for class_label, accuracy in enumerate(class_accuracy):
    print(f"Accuracy for class {class_label}: {accuracy}")

# Plot scatter plots
feature_names = merged_data.drop('Origin', axis=1).columns

for i in range(len(feature_names) - 1):
    for j in range(i + 1, len(feature_names)):
        plt.figure()
        for class_label in np.unique(y):
            class_data = merged_data[merged_data['Origin'] == class_label]
            plt.scatter(class_data[feature_names[i]],
                        class_data[feature_names[j]], label=f"Class {class_label}")
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        plt.legend()
        plt.show()
