from datetime import date
import time
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Se lee la data
uploaded_file = st.file_uploader('Choose a file')

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)

    # Se crean las columnas necesarias
    raw_data = raw_data.drop(['Fecha'], axis=1)
    raw_data.dropna(inplace=True)
    raw_data.insert(loc=0, column='date', value=pd.to_datetime(
        raw_data.index, dayfirst=True))
    raw_data['index'] = range(1, len(raw_data) + 1)
    raw_data = raw_data.set_index('index')

    module = st.sidebar.selectbox(
        'Módulo',
        ('Desarrollo de contenidos', 'Resultados')
    )

    if(module == 'Desarrollo de contenidos'):

        with st.expander('Datos'):

            st.header('Datos')

            st.write(raw_data)

            st.subheader('Gráficas:')

            c1 = px.line(raw_data, x='date', y='TCR1')

            st.plotly_chart(c1)

            graphType = st.selectbox('Agrupar por:', ('Mes', 'Año', 'Día'))

            if(graphType == 'Mes'):
                c2 = px.bar(raw_data.groupby(
                    raw_data['date'].astype('datetime64').dt.month).mean(),
                    labels={
                        'value': 'promedio',
                    'date': 'mes'
                })

                st.plotly_chart(c2)
            if(graphType == 'Año'):
                c3 = px.bar(raw_data.groupby(
                    raw_data['date'].astype('datetime64').dt.year).mean(),
                    labels={
                        'value': 'promedio',
                    'date': 'año'
                })

                st.plotly_chart(c3)
            if(graphType == 'Día'):
                c4 = px.bar(raw_data.groupby(
                    raw_data['date'].astype('datetime64').dt.day).mean(),
                    labels={
                        'value': 'promedio',
                    'date': 'día'
                })

                st.plotly_chart(c4)

        with st.expander('Tipo de cambio'):

            st.header('Tipo de cambio')

            st.write(
                'El tipo de cambio es el precio de la moneda de un país en términos de otra moneda.')

            st.subheader('Tipos:')

            st.write(
                'Nominal: Consiste en la relación a la que se puede intercambiar una moneda de un país con otra de otro país.')

            st.write(
                'Real: Es un indicador de los precios de una cesta de bienes y servicios de un país con respecto a los de otro país.')

            st.subheader('Tipo de cambio 0:')

            st.write(
                'Si el tipo de cambio quetzal/dólar es de 0, significa que el país dolarizó su moneda y por lo tanto el quetzal perdería su valor (valor 0) y la moneda en curso sería el dolar (valor 1).')

        with st.expander('Modelos'):

            st.header('Modelos')

            graph = st.selectbox('Tipo de gráfica:', ('Modelo exponencial',
                                                      'Modelo logarítmico', 'Modelo polinómico', 'Modelo de promedios móviles'))

            # ---------------------------------------------------------------------------- #
            #                              Modelo exponencial                              #
            # ---------------------------------------------------------------------------- #

            if(graph == 'Modelo exponencial'):

                def expModel(x, m, t, b):
                    return m * np.exp(-t * x) + b

                paramsExpModel, cv = curve_fit(
                    expModel, raw_data.index, raw_data['TCR1'], (2000, .01, 1))

                raw_data['expModel'] = expModel(
                    raw_data.index, *paramsExpModel)

                chart1 = px.line(raw_data, x='date', y=['TCR1', 'expModel'])

                st.subheader('Modelo exponencial')

                st.text('Score r2: ' +
                        str(r2_score(raw_data['TCR1'], raw_data['expModel'])))

                st.plotly_chart(chart1)

            # ---------------------------------------------------------------------------- #
            #                              Modelo logarítmico                              #
            # ---------------------------------------------------------------------------- #

            elif(graph == 'Modelo logarítmico'):

                def logModel(x, a, b):
                    return a*np.log(x) + b

                paramsLogModel, cv = curve_fit(
                    logModel, raw_data.index, raw_data['TCR1'])

                raw_data['logModel'] = logModel(
                    raw_data.index, *paramsLogModel)

                chart2 = px.line(raw_data, x='date', y=['TCR1', 'logModel'])

                st.subheader('Modelo logarítmico')

                st.text('Score r2: ' +
                        str(r2_score(raw_data['TCR1'], raw_data['logModel'])))

                st.plotly_chart(chart2)

            # ---------------------------------------------------------------------------- #
            #                               Modelo polinómico                              #
            # ---------------------------------------------------------------------------- #

            elif(graph == 'Modelo polinómico'):

                st.subheader('Modelo polinómico')

                degree = st.slider('Grado:', 0, 30, 15)

                parts = st.slider('Dividir el modelo en:', 1, 5, 1)

                initial = 0
                length = int(len(raw_data) / parts)

                raw_data['polModel'] = np.zeros(len(raw_data))

                for i in range(parts):

                    if parts == i + 1:
                        final = len(raw_data)
                    else:
                        final = initial + length

                    model3 = np.polyfit(
                        raw_data.index[initial:final], raw_data['TCR1'][initial:final], degree)

                    trendpoly3 = np.poly1d(model3)

                    raw_data['polModel'][initial:final] = trendpoly3(
                        raw_data.index[initial:final])

                    initial = final

                chart3 = px.line(raw_data, x='date', y=['TCR1', 'polModel'])

                st.text('Score r2: ' +
                        str(r2_score(raw_data['TCR1'], raw_data['polModel'])))

                st.write(chart3)

            # ---------------------------------------------------------------------------- #
            #                          Modelo de promedios móviles                         #
            # ---------------------------------------------------------------------------- #

            elif(graph == 'Modelo de promedios móviles'):

                st.subheader('Modelo de promedios móviles')

                degree = st.slider('Grado:', 0, 30, 15)

                window = st.slider('Ventana:', 1, 30, 10)

                raw_data['rolling'] = 0
                raw_data['rolling'] = raw_data.rolling(
                    window=window).mean()['TCR1']

                model4 = np.polyfit(raw_data.index[window:],
                                    raw_data['rolling'][window:], degree)

                trendpoly4 = np.poly1d(model4)

                raw_data['rollModel'] = trendpoly4(raw_data.index)

                chart4 = px.line(raw_data, x='date', y=['TCR1', 'rollModel'])

                st.text('Score r2: ' +
                        str(r2_score(raw_data['TCR1'], raw_data['rollModel'])))

                st.plotly_chart(chart4)

        with st.expander('Suavizamiento Exponencial'):

            st.header('Suavizamiento Exponencial')

            X = raw_data['date']
            y = raw_data['TCR1']

            train_size = st.slider(
                'Tamaño de los datos para entrenar:', 0.85, 0.98, 0.93)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-train_size, shuffle=False)

            model = ExponentialSmoothing(
                y_train, seasonal=None, damped=True, trend='add')

            model.index = X_train

            fit = model.fit()

            simulations = 100

            for i in range(simulations):
                p = fit.simulate(len(y_test), error='add')
                if i == 0:
                    preds = p
                else:
                    preds = preds + p
            pred = preds / simulations

            forecast = fit.forecast(len(y_test))

            chart5 = px.line(pd.DataFrame({
                'date': X,
                'TCR1': y,
                'fit': fit.fittedvalues,
                'predictions': pred,
                'forecast': forecast,
            }), x='date', y=['TCR1', 'fit', 'predictions', 'forecast'])

            st.plotly_chart(chart5)

    if(module == 'Resultados'):

        st.header('Predicciones')

        d = pd.to_datetime(st.date_input(
            'Fecha a predecir:',
            date(2021, 7, 6)))

        progress_bar = st.progress(0)

        indexes = raw_data.index[raw_data['date'] == d].tolist()
        if len(indexes) > 0:
            index = raw_data.index[raw_data['date'] == d].tolist()[0]
            real_TCR1 = raw_data[raw_data['date'] == d]['TCR1'][index]

            realValueText = ('Valor real: ' + str(real_TCR1))

            modelPred = ExponentialSmoothing(
                raw_data['TCR1'][:index], seasonal=None, damped=True, trend='add')

            simulations = 1000

            fitPred = modelPred.fit()

            for i in range(simulations):
                progress_bar.progress((i + 1)/simulations)
                p = fitPred.simulate(1, error='add')
                if i == 0:
                    mins = p
                    maxs = p
                else:
                    mins = np.minimum(mins, p)
                    maxs = np.maximum(maxs, p)

            predictionText = ('Predicción: ' +
                              str(round(fitPred.forecast(1)[index], 5)))
            confidenceText = ('Intervalo de confianza: +-' +
                              str(round((maxs[index] - mins[index]) / 2, 5)))
        else:

            realValueText = ('Valor real: N/A')

            diff = (d - raw_data['date'].iloc[-1]).days

            modelPred = ExponentialSmoothing(
                raw_data['TCR1'], seasonal=None, damped=True, trend='add')

            simulations = 1000

            fitPred = modelPred.fit()

            for i in range(simulations):
                progress_bar.progress((i + 1)/simulations)
                p = fitPred.simulate(diff + 1, error='add')
                if i == 0:
                    preds = p
                    mins = p
                    maxs = p
                else:
                    preds = preds + p
                    mins = np.minimum(mins, p)
                    maxs = np.maximum(maxs, p)
            pred = preds / simulations

            predictionText = ('Predicción: ' + str(round(pred
                                                         [raw_data.index[-1] + diff], 5)))

            confidenceText = ('Intervalo de confianza: +-' +
                              str(round((maxs[raw_data.index[-1] + diff] - mins[raw_data.index[-1] + diff]) / 10, 5)))

        progress_bar.empty()

        st.text(realValueText)

        st.text(predictionText)

        st.text(confidenceText)
