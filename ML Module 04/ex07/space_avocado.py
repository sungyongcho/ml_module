import pandas as pd
from benchmark_train import benchmark_train
import matplotlib.pyplot as plt


def plot_evaluation_curve(degrees, mse_values):
    plt.plot(degrees, mse_values, 'bo-')
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Mean Squared Error')
    plt.title('Evaluation Curve')
    plt.show()


def plot_predictions(model, X_test_poly, y_test):
    print(X_test_poly.shape, y_test.shape)
    y_pred = model.predict_(X_test_poly)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_test_poly[:, 1], X_test_poly[:, 2],
               y_test, c='b', label='True Price')
    ax.scatter(X_test_poly[:, 1], X_test_poly[:, 2],
               y_pred, c='r', label='Predicted Price')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Production Distance')
    ax.set_zlabel('Price')
    ax.legend()
    plt.title('True Price vs Predicted Price')
    plt.show()

    print(X_test_poly.shape, )
    print(X_test_poly, )
# Step 1: Load the dataset
df = pd.read_csv("space_avocado.csv")

# Step 2: Split the dataset into training, cross-validation, and test sets
X = df[['weight', 'prod_distance', 'time_delivery']]
y = df['target']


degrees = [1, 2, 3, 4]

mse_values, best_model, X_test_poly, y_test = benchmark_train(X, y, degrees)

# Plot the evaluation curve
plot_evaluation_curve(degrees, mse_values)


# load the saved

# Plot the true price and predicted price using the best model
plot_predictions(best_model, X_test_poly, y_test)
