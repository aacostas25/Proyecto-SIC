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

@app.route('/sinteticos', methods=['GET', 'POST'])
def sinteticos():
    global valuesS
    valuesS = [x for x in request.form.values()]
    print(valuesS)
    return render_template('sinteticos.html')

@app.route('/resultados_sinteticos', methods=['GET', 'POST'])
def resultados_sinteticos():
    global valuesS
    valuesS = [x for x in request.form.values()]
    muestras = int(valuesS[0])
    std = float(valuesS[1])
    centros = int(valuesS[2])
    iteracionesS = int(valuesS[3])
    etaS = float(valuesS[4])
    funcionesS = int(valuesS[5])
    neuronas1S = int(valuesS[6])
    neuronas2S = int(valuesS[7])
    x1,X1,y1,X_test1,Y_test2= blob(muestras,centros,std)
    plt.clf()
    plt.figure()
    print(np.unique(y1))
    unique_labels = np.unique(y1)
    for label in unique_labels:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][label]
        indices = np.where(y1 == label)[0]
        plt.scatter(X1[indices, 0], X1[indices, 1], c=color, edgecolor='black', label=f'Clase {int(label)}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Valores de X1 respecto a y1')
    plt.legend(loc='lower left')
    plot_filename = "./App/static/images/bloc.png"
    plt.savefig(plot_filename)
    funcion2 = seleccionarFA(funcionesS)
    capa0_1,capa1_1,capa2_1,errores_1S,iteraciones1S,l3_error_1 = RN2(X_test1,Y_test2,x1,funcion2,eta = etaS,ite=iteracionesS,neuronas_1=neuronas1S,neuronas_2=neuronas2S)
    capa0_2,capa1_2,capa2_2,errores_2S,iteraciones2S,l3_error_2 = RN2(X_test1,Y_test2,x1,funcion2,eta=etaS+0.002,neuronas_1=neuronas1S,neuronas_2=neuronas2S)
    capa0_3,capa1_3,capa2_3,errores_3S,iteraciones3S,l3_error_3 = RN2(X_test1,Y_test2,x1,funcion2,eta=etaS-0.002,neuronas_1=neuronas1S,neuronas_2=neuronas2S)
    plt.clf()
    plt.figure()
    plt.title('Error Red Neuronal')
    plt.plot(iteraciones1S,errores_1S)
    plt.plot(iteraciones2S,errores_2S)
    plt.plot(iteraciones3S,errores_3S)
    plt.ylabel('Error medio')
    plt.legend([f'eta ={etaS} ',f'eta = {etaS+0.002}',f'eta = {etaS-0.002}'])
    plot_filename2 = "./App/static/images/bloc2.png"
    plt.savefig(plot_filename2)
    return render_template('sinteticos.html')

@app.route('/results',  methods=['GET','POST'])
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
        capa0_1,capa1_1,capa2_1,errores_1,iteraciones,l3_error_1 = RN2(X,Y,x,funcion,eta = eta,neuronas_1=neuronas1,neuronas_2=neuronas2)
        capa0_2,capa1_2,capa2_2,errores_2,iteraciones,l3_error_2 = RN2(X,Y,x,funcion,eta = eta+0.002,neuronas_1=neuronas1,neuronas_2=neuronas2)
        capa0_3,capa1_3,capa2_3,errores_3,iteraciones,l3_error_3 = RN2(X,Y,x,funcion,eta = eta-0.002,neuronas_1=neuronas1,neuronas_2=neuronas2)
        loss, = np.mean(np.abs(l3_error_1),axis=0)
        save_to_csv(values[:-1],loss)
        plt.figure()
        plt.title('Error Red Neuronal')
        plt.plot(iteraciones,errores_1)
        plt.plot(iteraciones,errores_2)
        plt.plot(iteraciones,errores_3)
        plt.xlabel('Iteraciones')
        plt.ylabel('Error medio')
        plt.legend([f'eta ={eta} ',f'eta = {eta+0.002}',f'eta = {eta-0.002}'])
        #plt.show()
        plt.savefig(f"./App/static/images/result_random.png", transparent = True)
        plt.clf()
        return render_template('results.html',loss=np.round(loss,3))
    except Exception as e:
        print("Exception:", e)
        return render_template('errores.html')

@app.route('/Registro',methods=['GET'])
def registro():
    df = pd.read_csv('registros.csv')
    df_html = df.to_html(classes='table table-striped')
    return render_template('registro.html',tabla = df_html)
