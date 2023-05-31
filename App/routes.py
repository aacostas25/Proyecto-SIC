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
sys.path.append('/Users/andreacamilaacosta/Proyecto-SIC/App/')
from redes import *
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
    data.insert(len(data), perdida)  # ingresar el valor de funcion de perdida
    with open('registros.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

@app.route('/Sinteticos',  methods=['GET', 'POST'])
def sinteticos():
    global valuesS
    valuesS = [x for x in request.form.values()]
    print(valuesS)
    return render_template('sinteticos.html')

@app.route('/results',  methods=['GET'])
def results():
    try:
        print('lista: ', values)
        data_type = int(values[0])
        iteraciones = int(values[1])
        eta = float(values[2])
        funciones = int(values[3])
        neuronas1 = int(values[4])
        neuronas2 = int(values[5])
        print(data_type)
        x,X,Y,X_test,Y_test = seleccionarDB(data_type)
        funcion = seleccionarFA(funciones)
        capa0_1,capa1_1,capa2_1,errores_1,iteraciones,l3_error_1 = RN2(X,Y,x,funcion,eta = eta)
        capa0_2,capa1_2,capa2_2,errores_2,iteraciones,l3_error_2 = RN2(X,Y,x,funcion,eta = eta+0.005)
        capa0_3,capa1_3,capa2_3,errores_3,iteraciones,l3_error_3 = RN2(X,Y,x,funcion,eta = eta-0.005)
        loss, = np.mean(np.abs(l3_error_1),axis=0)
        save_to_csv(values[:-1],loss)
        plt.figure()
        plt.title('Error Red Neuronal')
        plt.plot(iteraciones,errores_1)
        plt.plot(iteraciones,errores_2)
        plt.plot(iteraciones,errores_3)
        plt.xlabel('Iteraciones')
        plt.ylabel('Error medio')
        plt.legend(['eta = '+str(0.01),'eta = '+str(0.015),'eta = '+str(0.005)])
        #plt.show()
        plt.savefig(f"./App/static/images/result_random.png", transparent = True)
        plt.clf()
        return render_template('results.html',loss=np.round(loss,3))
    except:
        return render_template('errores.html')

@app.route('/Registro',methods=['GET'])
def registro():
    df = pd.read_csv('registros.csv')
    df_html = df.to_html(classes='table table-striped')
    return render_template('registro.html',tabla = df_html)
