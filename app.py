import streamlit as st
import pandas as pd
# from sklearn import datasets
# from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from streamlit_option_menu import option_menu


print(pickle.format_version)
print("hello")


pickle_in = open('model_india.pkl','rb')
classifier = pickle.load(pickle_in)
def Welcome():
    return 'WELCOME ALL!'

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Select City',
                          
                          ['Dhaka',
                           'Melbourn',
                           'Bangaluru'],
                          icons=['Bangladesh','Melbourn','Bangaluru'],
                          default_index=0)
    



def predict_price(location,sqft,bath,bhk):    

    # loc_index = np.where(X.columns==location)[0][0]
# 243 colums 
    x = np.zeros(243)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    # if loc_index >= 0:
    #       x[loc_index] = 1

    return np.round(classifier.predict([x])[0],2)



def main(): 
    home = pd.read_csv("bengaluru_house_prices.csv")
    loc = home["location"].unique()
    # st.title("Bangalore House Rate Prediction")
    html_temp = """
    <h2 style="color:white;text-align:left;"> Rentology Price Prediction Model </h2>
    """

    st.markdown(html_temp,unsafe_allow_html=True)
    st.subheader('Please enter the required details:')
    st.write(f"Selected city: {selected}")  

    location = st.selectbox("Location",loc)
    sqft = st.text_input("Sq-ft area","")
    bath = st.text_input("Number of Bathroom","")
    bhk = st.text_input("Number of BHK","")

    result=""


    # data = {

    #     'lat': [23.8103, 23.7949, 23.7806],
    #     'lon': [90.4125, 90.4043, 90.4167]
    # }
    # df = pd.DataFrame(data)

    # # Display map
    # st.map(df)



    if st.button("House Price in Lakhs"):
        result=predict_price(location,sqft,bath,bhk)
    st.success('The output is {}'.format(result))
    # if st.button("About"):
    #     st.text("Please find the code at")
    #     st.text("https://github.com/Lokeshrathi/Bangalore-house-s-rate")

if __name__=='__main__':
    main()
    