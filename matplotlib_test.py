import matplotlib.pyplot as plt
import numpy as np

# Training Data for the x-axis and y-axis
learn_x = np.random.uniform(0, 10, size=100)
learn_y = 2*learn_x +np.random.normal(0, 1, size=100)
# we can multiply x_learn by anything, this coefficient will be the weight with the lowest mse, the np.random adds variation
# Create a line plot

plt.scatter(learn_x, learn_y, label='Training Set')

#mean squared function
def calc_mse(weight, learn_x, learn_y):
    y_pred = weight * learn_x
    mse = np.mean((y_pred - learn_y) ** 2)
    return mse

weights = np.linspace(-2, 4, num=100)
mse_values = [calc_mse(weight, learn_x, learn_y) for weight in weights]
# Add labels and a title
plt.plot(weights, mse_values, label='MSE')
plt.xlabel('Weight')
plt.ylabel('Mean Square Error')
plt.title('Mean Square Error as a Function of Weight')
plt.legend()
plt.show()
