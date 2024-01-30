import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets

st.write("""
# Visualización de datos de la base de datos Iris

""")

st.sidebar.header('Autor: Gabriel Barragán')

iris = datasets.load_iris()
iris_df = pd.DataFrame(data = iris['data'], columns=iris['feature_names'])
iris_df['Iris type'] = iris['target']
iris_df['Iris name'] = iris_df['Iris type'].apply(lambda x: 'setosa' if x==0 else ('versicolor' if x==1 else 'virginica'))

st.write('Primeros 5 datos')
primeros = iris_df.head()
st.write(primeros)

st.write('Últimos 5 datos')
ultimos = iris_df.tail()
st.write(ultimos)

st.write('Visualización')
plot_1 = sns.pairplot(data = iris_df[iris_df.columns.difference(['Iris type'])], hue = 'Iris name', palette='Set2')
st.pyplot(plot_1.fig)

plot_2 = sns.heatmap(iris_df[iris_df.columns.difference(['Iris type'])].corr(), cmap='icefire')
st.pyplot(plot_2.fig)
