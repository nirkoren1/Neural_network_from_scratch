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
test_size = 300
for i in range(test_size):
    file = open("Points/predicted-false-" + str(i) + ".txt", "r")
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

    file = open("Points/predicted-true-" + str(i) + ".txt", "r")
    x = []
    y = []

    for line in file:
        if line[0] == 'n':
            continue
        x.append(float(line.split()[0]))
        y.append(float(line.split()[1]))
    file.close()
    all_true_x.append(x)
    all_true_y.append(y)


for i in range(test_size):
    print(i, '\r')
    x = all_false_x[i]
    y = all_false_y[i]
    # sctatter plot
    plt.scatter(x, y, marker='o', color='red', label='Predicted False')

    x = all_true_x[i]
    y = all_true_y[i]
    plt.scatter(x, y, marker='o', color='blue', label='Predicted True')

    # draw a circle at the origin of the coordinate system with radius
    for i in range(0, 360, 1):
        plt.scatter([circle_x + radius * np.cos(i)], [circle_y + radius * np.sin(i)], color='black')

    plt.title('Predictions')
    # position the legend on the upper left
    plt.legend(loc='upper left')
    plt.pause(0.01)
    plt.clf()
plt.show()
