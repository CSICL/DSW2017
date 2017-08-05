import numpy as np
import matplotlib.pyplot as plt

def estimate_coefficients(x, y):
    # number of observations/points
    n = np.size(x)

    # Calculating mean of X and Y vector
    mean_x, mean_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    cross_deviation_xy = np.sum(y*x - n*mean_y*mean_x)
    deviation_xx = np.sum(x*x - n*mean_x*mean_x)

    # calculating regression coefficients
    b = cross_deviation_xy / deviation_xx
    a = mean_y - b*mean_x

    return(a, b)

def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "r",
               marker = "o", s = 30)

    # predicted response vector
    y_prediction = b[0] + b[1]*x
    print "y_prediction :", y_prediction

    # plotting the regression line
    plt.plot(x, y_prediction, color = "b")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()

def main():
    # observations
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    # estimating coefficients
    b = estimate_coefficients(x, y)
    print("The estimated coefficients is:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plot_regression_line(x, y, b)

if __name__ == "__main__":
    main()