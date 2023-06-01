import numpy as np
import pandas as pd
import pickle
from my_logistic_regression import MyLogisticRegression
from other_metrics import f1_score_
from polynomial_model_extended import add_polynomial_features
from data_spliter import data_spliter


def benchmark_train(input_data, input_labels, output_file='models.pickle'):
    degrees = [0, 1, 2, 3]  # Degrees of polynomial features
    lambda_values = np.linspace(0, 1, 10)  # Range of lambda values
    models = {}  # Dictionary to store the models

    # print(input_data.shape)  # Print the shape of X
    # print(input_data)  # Print the content of X
    for degree in degrees:
        # Transform the features into a polynomial of the current degree
        X_poly = add_polynomial_features(input_data, power=3)

        # Split the dataset into a training, cross-validation, and test set
        X_train, X_cv, X_test, y_train, y_cv, y_test = data_spliter(
            X_poly, input_labels, 0.6, 0.2, fix='y')

        for lambda_val in lambda_values:
            # Create a binary target variable for the current class
            binary_target = (y_train == degree).astype(int)
            binary_target = np.reshape(binary_target, (-1, 1))

            # Create an instance of your logistic regression model with regularization parameter lambda_val
            model = MyLogisticRegression(theta=np.random.rand(X_train.shape[1] + 1, 1),
                                         alpha=0.001, max_iter=1000, lambda_=lambda_val)

            # Train the model on the training set
            model.fit_(X_train, binary_target)

            # Evaluate the model on the cross-validation set
            y_cv_binary = (y_cv == degree).astype(int)
            y_cv_pred = model.predict_(X_cv)
            y_cv_pred = y_cv_pred.flatten().astype(int)
            # print(y_cv)
            # print(y_cv_pred)
            f1 = f1_score_(y_cv_binary, y_cv_pred)

            # Save the model with its f1 score
            models[(degree, lambda_val)] = (model, f1)

            # Calculate the F1 score on the test set for the current model
            y_test_binary = (y_test == degree).astype(int)
            y_test_pred = model.predict_(X_test)
            y_test_pred = y_test_pred.flatten().astype(int)
            f1_test = f1_score_(y_test_binary, y_test_pred)
            print(
                f"Degree: {degree}, Lambda: {lambda_val}, F1 Score: {f1}, Test F1 Score: {f1_test}")

    # Save the models to the output file
    with open(output_file, 'wb') as f:
        pickle.dump(models, f)

    return models
