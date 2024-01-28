import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

st.write("""
# Visualización de datos de la base de datos **Iris**

Flores Iris
""")

iris = datasets.load_iris()
iris_df = pd.DataFrame(data = iris['data'], columns=iris['feature_names'])
iris_df['Iris name'] = iris_df['Iris type'].apply(lambda x: 'setosa' if x==0 else ('versicolor' if x==1 else 'virginica'))

primeros = iris_df.head()
st.write(primeros)