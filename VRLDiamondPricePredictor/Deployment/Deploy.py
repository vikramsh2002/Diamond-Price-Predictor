import streamlit as st
import numpy as np
import joblib as jb
import warnings
import os
st.set_page_config(layout="centered",page_icon=":gem:")
warnings.filterwarnings("ignore")
path = os.path.dirname(__file__)
st.image(path+"/MyLogo.png",width=500)
st.write(path)
col1,col2= st.columns(2)
col1.markdown('<h1 style="color:blue; font-family:cursive; text-align:center">Diamomd Price <br>Predictor </h1>',unsafe_allow_html=True)
col2.image(path+"/diamond.png",width=300)
path_main = "/".join(path.split("/") [:-1])
st.write(path_main)

features = np.zeros((11,))
maps=jb.load(path_main+"/Maps.pkl")

# Creating form to take

# Shapes section
with st.expander("Shape of Diamonds"):
    shapes= list(maps["Shapes"].keys())
    col1,col2 = st.columns(2)
    col1.markdown("<h3>Select the Shape of your Diamond</h3>",unsafe_allow_html=True)
    col2.image(path+"/Shapes/allshapes.png",caption='Different available shapes of Diamonds',width=150,)
    sh_select = col1.selectbox("",shapes)
    #print("Shape: ",sh_select)
    features[-1]=maps["Shapes"][sh_select]


# Messurement section
st.cache()
with st.expander("Messurements of Diamonds"):
    features[0] = st.number_input("Depth of Diamond(mm)",min_value=3.0,max_value=13.0,step=0.01,value=3.0)
    features[1] = st.number_input("Height of Diamond(mm)",min_value=3.0,max_value=10.0,step=0.01,value=3.0)
    features[2] = st.number_input("Width of Diamond(mm)",min_value=1.0,max_value=7.0,step=0.01,value=1.0)
    features[3] = st.number_input("Weight of Diamond in Carat(ct)",min_value=0.10,max_value=7.0,step=0.01,value=0.5)


# Clarity Scale
with st.expander("Appearance of Diamonds"):
    features[4] =maps["Clarity"][st.selectbox("Select the Clarity Grade of your diamond",maps["Clarity"].keys())]

    #Goodness Scale
    col1,col2,col3,col4=st.columns(4)
    cut_sc=col2.selectbox("Select the Goodness of Cutting",maps["Goodness_short"].keys())
    pol_sc=col3.selectbox("Select the Goodness of Polishing",maps["Goodness_short"].keys())
    sym_sc=col4.selectbox("Select the Goodness of Symmetry",maps["Goodness_short"].keys())

    features[5]= maps["Goodness"][ maps["Goodness_short"][cut_sc]  ]
    features[6]= maps["Goodness"][ maps["Goodness_short"][pol_sc]  ]
    features[7]= maps["Goodness"][ maps["Goodness_short"][sym_sc]  ]

    # Fluorescence
    features[8] =maps["Fluorescence"][st.selectbox("Select the Fluorescence Grade of your diamond",maps["Fluorescence"].keys())]

    # Color 
    features[9] =maps["Color"][st.selectbox("Select the Color Grade of your diamond",maps["Color"].keys())]

# Prediction Part Begins
def Prediction(features):
    model = jb.load(path_main+"/Model/GBRModel.pkl")
    scale_x=jb.load(path_main+"/Scalers/ScalerX.pkl")
    feat=scale_x.transform(features.reshape(1,-1))
    y_pd= model.predict(feat)
    scale_y = jb.load(path_main+"/Scalers/ScalerY.pkl")
    price = scale_y.inverse_transform(y_pd.reshape(-1,1)).reshape(-1,)
    return abs(price.round(2))

_,col2,col3 = st.columns(3)


# Color button
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: black;
    font-size:20px;
    padding:20px;
    border:90%;
    color:yellow;
}
div.stButton > button:first-child:hover {
    background-color: aqua;
    color:red;
}

.out{
    font-size:50px;
    color: white;
    margin-left:50%;
    border:2px solid black;
    text-align:center;
    border-radius:30%;
    padding:10px;
    font-family:;
    background-color: blue;
    font-weight:normal;
}
.out:hover{
    background-color: black;
    color: orange;
    border-radius:60%;
    padding:20px;
}

</style>""", unsafe_allow_html=True)


if col2.button("♦ Predict My Diamond's Price",):
    price=Prediction(features)[0]
    st.markdown("<div class='out'>Price ${}</div>".format(price),unsafe_allow_html=True)









#print(features)
