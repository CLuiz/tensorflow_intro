import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def plot_act(i=1.0, actfunc=lambda x: x):
    ws = np.arange(-0.5, 0.5, 0.5)
    bs = np.arange(-0.5, 0.5, 0.5)

    X, Y = np.meshgrid(ws, bs)

    os = np.array([actfunc(tf.constant(w * i + b)).eval(session=sess)
                   for w, b in zip(np.ravel(X), np.ravel(Y))])

    Z = os.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
