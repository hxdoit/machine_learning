import numpy as np

def gradient(x, y, alpha):
    shape = x.shape
    theta = np.ones(shape[1]).reshape((shape[1], 1))
    gradient = np.dot(x.transpose(), np.dot(x, theta) - y) / shape[0]
    while not np.all(np.absolute(gradient) < 1e-5):
        gradient = np.dot(x.transpose(), np.dot(x, theta) - y) / shape[0]
        theta = theta - alpha * gradient 
    return theta

if __name__ == '__main__':
    m = 20
    x0 = np.ones((m, 1))
    x1 = np.arange(1, m + 1).reshape(m, 1)
    x = np.hstack((x0, x1))
    print x
    y = np.array([
        3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
        11, 13, 13, 16, 17, 18, 17, 19, 21
    ]).reshape(m, 1)
    print y
    for i in range(m):
        print x1[i][0], y[i][0]
    print gradient(x, y, 0.01)
