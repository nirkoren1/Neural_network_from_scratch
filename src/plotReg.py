# plot points on a graph from a file
import matplotlib.pyplot as plt
import numpy as np
radius = 11
circle_x = 0
circle_y = 0
all_true_x = []
all_true_y = []
all_false_x = []
all_false_y = []
test_size = 30
for i in range(test_size):
    with open("Points2/points_" + str(i) + ".txt", "r") as file:
        x = []
        y = []

        for line in file:
            if line[0] == 'n':
                continue
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
    all_false_x.append(x)
    all_false_y.append(y)
    all_true_x.append(x)
    y_ = []
    for y__ in x:
        y_.append(2 * y__**2 + 1)
    all_true_y.append(y_)


for i in range(test_size):
    print(i, '\r')
    x = all_false_x[i]
    y = all_false_y[i]
    # sctatter plot
    plt.scatter(x, y, marker='o', color='red', label='model predictions')

    x = all_true_x[i]
    y = all_true_y[i]
    plt.scatter(x, y, marker='o', color='blue', label='y = 2x^2+1', alpha=0.005)

    plt.title('Predictions')
    # set the x and y limits
    plt.xlim(-15, 15)
    plt.ylim(-20, 250)
    # position the legend on the upper left
    plt.legend(loc='upper left')
    plt.pause(0.01)
    plt.clf()

x = all_false_x[i]
y = all_false_y[i]
# sctatter plot
plt.scatter(x, y, marker='o', color='red', label='model predictions')

x = all_true_x[i]
y = all_true_y[i]
plt.scatter(x, y, marker='o', color='blue', label='y = 2x^2+1', alpha=0.005)

plt.title('Predictions')
# set the x and y limits
plt.xlim(-15, 15)
plt.ylim(-20, 250)
# position the legend on the upper left
plt.legend(loc='upper left')
plt.show()
