from data_spliter import data_spliter
from polynomial_model_extended import add_polynomial_features
from ridge import MyRidge
import numpy as np
import pandas as pd


def evaluate_model(model, X_test_poly, y_test):
    # Make predictions on the test data
    y_pred = model.predict_(X_test_poly)

    # Calculate the mean squared error
    mse = np.mean((y_test - y_pred) ** 2)
    return mse


def benchmark_train(x, y, degrees):
    X_train, X_cv, X_test, y_train, y_cv, y_test = data_spliter(
        x.to_numpy(), y.to_numpy(), 0.6, 0.2)

    print(y_train.shape, y_cv.shape, y_test.shape)

    mse_values = []

    # Apply feature scaling to training, cross-validation, and test sets
    X_train_scaled = (X_train - X_train.min()) / \
        (X_train.max() - X_train.min())
    X_cv_scaled = (X_cv - X_train.min()) / (X_train.max() - X_train.min())
    X_test_scaled = (X_test - X_train.min()) / (X_train.max() - X_train.min())

    print(X_train_scaled.shape)
    print(X_cv_scaled.shape)
    print(X_train_scaled.shape[1] == X_cv_scaled.shape[1])

    variance = np.var(y_train)

    for degree in degrees:
        # Apply polynomial feature transformation to training set
        X_train_poly = add_polynomial_features(X_train_scaled, degree)

        X_test_poly = add_polynomial_features(X_test_scaled, degree)

        # Apply polynomial feature transformation to cross-validation set
        X_cv_poly = add_polynomial_features(X_cv_scaled, degree)

        # Create an instance of MyRidge
        # Add 1 for the intercept term
        num_features = X_train_poly.shape[1] + 1
        thetas = np.random.rand(num_features, 1)
        lr = MyRidge(thetas, alpha=0.01, max_iter=1000)

        # Fit the model to the training data
        lr.fit_(X_train_poly, y_train)

        # Make predictions on the cross-validation data
        y_cv_pred = lr.predict_(X_cv_poly)

        # Evaluate the model on the cross-validation data
        mse = lr.mse_(y_cv, y_cv_pred)
        normalized_mse = mse / variance

        mse_values.append(normalized_mse)

    # Find the best degree with the minimum mean squared error
    print(degrees, mse_values)
    best_degree = degrees[np.argmin(mse_values)]
    print("Best Degree:", best_degree)

    # Train the best model based on the best degree using the combined training and cross-validation sets
    X_train_cv_poly = add_polynomial_features(
        np.vstack((X_train_scaled, X_cv_scaled)), best_degree)
    # Concatenate y_train and y_cv
    y_train_cv = np.concatenate((y_train, y_cv))

    X_test_poly = add_polynomial_features(X_test_scaled, best_degree)
    num_features = X_train_cv_poly.shape[1] + 1
    thetas = np.random.rand(num_features, 1)
    best_model = MyRidge(thetas, alpha=0.01, max_iter=1000)
    # Use y_train_cv instead of np.vstack((y_train, y_cv))
    best_model.fit_(X_train_cv_poly, y_train_cv)

    # Save the parameters of all the models into a file (models.csv)
    models = {'degrees': degrees, 'mse_values': mse_values}
    models_df = pd.DataFrame(models)
    models_df.to_csv('models.csv', index=False)
    print("Data saved to models.csv file.")

    return mse_values, best_model, X_test_poly, y_test
