import pickle
import numpy as np
import pandas as pd
import streamlit as st
import sklearn

loaded_model = pickle.load(open('streamlit_insurance_predictcharges.pkl', 'rb'))

from PIL import Image
image = Image.open('hospital2.jpg')
image_hospital = Image.open('hospital.jpg')

st.title("ML App :sunglasses:")
html_temp = """
<div style="background-color:teal ;padding:10px">
<h2 style="color:white;text-align:center;">Prediccion de Gastos por Seguros</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

st.image(image,use_column_width=False)

add_selectbox = st.sidebar.selectbox(
"Como le gustaría predecir?",
("Online", "Lote"))

st.sidebar.info('Esta aplicación está creada para predecir los gastos hospitalarios de los pacientes')
st.sidebar.success('https://github.com/waloZarate')
    
st.sidebar.image(image_hospital)

if add_selectbox == 'Online':
    #st.title("Predict insurance charges")
    st.write("""*:point_right: @ML app desarrollada por Oswaldo L. Zárate*""")

    def load_data():
        df = pd.DataFrame({'sex': ['Male','Female'],
                        'smoker': ['Yes', 'No']}) 
        return df

    df = load_data()

    def load_data():
        df1 = pd.DataFrame({'region' : ['southeast' ,'northwest' ,'southwest' ,'northeast']}) 
        return df1

    df1 = load_data()

    sex = st.selectbox("Seleccione Genero", df['sex'].unique())
    smoker = st.selectbox("Usted fuma?", df['smoker'].unique())
    region = st.selectbox("A que región de Estados Unidos pertenece?", df1['region'].unique())
    age = st.slider("Cual es su edad?", 18, 100)
    bmi = st.slider("Cual es su IMC?", 10, 60)
    children = st.slider("Número de hijos", 0, 10)

    if sex == 'male':
        gender = 1
    else:
        gender = 0
        
    if smoker == 'yes':
        smoke = 1
    else:
        smoke = 0
        
    if region == 'southeast':
        reg = 2
    elif region == 'northwest':
        reg = 3
    elif region == 'southwest':
        reg = 1
    else:
        reg = 0

    features = [gender, smoke, reg, age, bmi, children]

    int_features = [int(x) for x in features]
    final_features = [np.array(int_features)]
    
    st.info('Pulse el boton **Predecir** para ver el resultado:')

    if st.button('Predecir'):           # when the submit button is pressed
        prediction =  loaded_model.predict(final_features)
        st.balloons()
        st.success(f'Your insurance charges would be: ${round(prediction[0],2)}')
        
if add_selectbox == 'Lote':
    
    file_upload = st.file_uploader("Upload csv file para predicciones", type=["csv"])
    
    if file_upload is not None:
        data = pd.read_csv(file_upload)
        st.write(data)

