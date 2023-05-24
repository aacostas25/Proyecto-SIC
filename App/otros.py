import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

def cuadrado(n):
    x = np.linspace(-10,10,n)
    y = x**2
    return x,y
