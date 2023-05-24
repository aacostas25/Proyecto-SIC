from App import app
from flask import render_template,send_file, request, url_for,jsonify
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
import plotly.express as px
import csv
import io
import sys
sys.path.append('/Users/andreacamilaacosta/ProyectoSIC/App')
from otros import *
matplotlib.use('agg')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/layout")
def otro():
    return render_template('layout.html')

@app.route('/Implementation',  methods=['GET', 'POST'])
def implementation():
    global values
    values = [x for x in request.form.values()]
    print(values)
    # Primero se deben capturar los parametros (dataset, algoritmo y valor de k)
    return render_template('implementation.html')
def save_to_csv(data,perdida):
    data.insert(0, perdida)  # ingresar el valor de funcion de perdida
    with open('registros.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

@app.route('/Resultados',  methods=['GET'])
def results():
    try:
        print('lista: ', values)
        l = int(values[2])
        k = int(values[1])
        d = int(values[4])
        n = int(values[3])
        data_type = int(values[0])
        print(data_type)
        x,y = cuadrado(n) #Funcion del archivo otros.py
        save_to_csv(values,x[0])
        plt.figure()
        plt.plot(x,y)
        plt.savefig(f"./App/static/images/result_random.png", transparent = True)
        plt.clf()
        return render_template('results.html')
    except:
        return render_template('errores.html')

@app.route('/Registro',methods=['GET'])
def registro():
    df = pd.read_csv('registros.csv')
    df_html = df.to_html()
    return render_template('registro.html',tabla = df_html)
