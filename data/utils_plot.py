import matplotlib.pyplot as plt

def plot_that(input_array):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(input_array[:, 0], input_array[:, 1], input_array[:, 2], c='r', marker='o')
    plt.show()