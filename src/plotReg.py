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
test_size = 200
for i in range(test_size):
    file = open("Points2/points_" + str(i) + ".txt", "r")
    x = []
    y = []

    for line in file:
        if line[0] == 'n':
            continue
        x.append(float(line.split()[0]))
        y.append(float(line.split()[1]))
    file.close()
    all_false_x.append(x)
    all_false_y.append(y)
    all_true_x.append(x)
    y_ = []
    for y__ in x:
        y_.append(2 * y__ + 1)
    all_true_y.append(y_)


for i in range(test_size):
    print(i, '\r')
    x = all_false_x[i]
    y = all_false_y[i]
    # sctatter plot
    plt.scatter(x, y, marker='o', color='red', label='model predictions')

    x = all_true_x[i]
    y = all_true_y[i]
    plt.scatter(x, y, marker='o', color='blue', label='y = 2x+1')

    plt.title('Predictions')
    # set the x and y limits
    plt.xlim(0, 10)
    plt.ylim(1, 21)
    # position the legend on the upper left
    plt.legend(loc='upper left')
    plt.pause(0.01)
    plt.clf()
plt.show()
