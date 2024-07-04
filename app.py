import streamlit as st
import pandas as pd
# from sklearn import datasets
# from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from streamlit_option_menu import option_menu
# joblib melbourn er tar jonno lagche 
import joblib


# just to check the version || naile jamela hoi requirment file create e || faltu shob error
# print(pickle.format_version)
# print("hello")

#loading the saved models || akbere kora jaito but good practise 
pickle_in = open('model_india.pkl','rb')
classifier = pickle.load(pickle_in)

wasington_model = open('trained_model_wasington.pkl','rb')
wasington_model_classifier =  pickle.load(wasington_model)

dhaka_model= open('model_dhaka.pkl', 'rb')
dhaka_model_classifier = pickle.load(dhaka_model)


def Welcome():
    return 'WELCOME ALL!'
st.title('Rentology Price Prediction Model')


# sidebar for navigation
with st.sidebar:
    selected = option_menu('Select City',                       
                          ['Dhaka',
                           'Entire-Dhaka',
                           'Melbourne',
                           'Bangaluru',
                           'washington'],
                          icons=['1-circle-fill','2-circle-fill','3-circle-fill','4-circle-fill','5-circle-fill'],
                          default_index=0) #jathe default dhaka nei
    

# Load the trained model

if selected == 'Dhaka':
    # Load the model from the pickle file
    loaded_model = pickle.load(open('model_dhaka.pkl', 'rb'))

    # Function to predict rent price
    def predict_price(location, area, bed, bath):
        x = np.zeros(418)  # Adjust this based on your model's feature set ??? eita mathai raikho as feature expected error dei ||
        x[0] = area
        x[1] = bed
        x[2] = bath

        return np.round(loaded_model.predict([x])[0], 2)

    def main():
        # List of hardcoded locations for Dhaka map
        # locations = {
        #     'Mohammadpur Dhaka': (23.759, 90.361),
        #     'Mirpur Dhaka': (23.822, 90.365),
        #     'Block D Section 12 Mirpur Dhaka': (23.815, 90.358),
        #     'Dhanmondi Dhaka': (23.746, 90.372),
        #     'Block E Section 12 Mirpur Dhaka': (23.815, 90.36),
        #     'Sector 10 Uttara Dhaka': (23.879, 90.391),
        #     'Paikpara Ahmed Nagar Mirpur Dhaka': (23.812, 90.364),
        #     'Kallyanpur Mirpur Dhaka': (23.764, 90.365),
        #     'Section 12 Mirpur Dhaka': (23.816, 90.36),
        #     'Block B Section 12 Mirpur Dhaka': (23.817, 90.36),
        #     'Joar Sahara Dhaka': (23.848, 90.42),
        #     'Block C Section 12 Mirpur Dhaka': (23.818, 90.36),
        #     'West Shewrapara Mirpur Dhaka': (23.797, 90.366),
        #     'Shyamoli Dhaka': (23.762, 90.366),
        #     'PC Culture Housing Mohammadpur Dhaka': (23.762, 90.362),
        #     'Hazaribag Dhaka': (23.725, 90.408),
        #     'South Baridhara Residential AreaD. I. T. Project Badda Dhaka': (23.795, 90.423),
        #     'Block G Bashundhara R-A Dhaka': (23.811, 90.411),
        #     'Sector 13 Uttara Dhaka': (23.871, 90.394),
        #     'Uttar Badda Badda Dhaka': (23.796, 90.431),
        #     'Baitul Aman Housing Society Adabor Dhaka': (23.769, 90.353),
        #     'Section 1 Mirpur Dhaka': (23.814, 90.355),
        #     'Mohammadi Housing LTD. Mohammadpur Dhaka': (23.757, 90.361),
        #     'Badda Dhaka': (23.793, 90.419),
        #     'Sector 14 Uttara Dhaka': (23.873, 90.398),
        #     'Rupnagar R/A Mirpur Dhaka': (23.828, 90.362),
        #     'Shantinagar Dhaka': (23.738, 90.41)
        #     
        # }
        locations = {
        'Mohammadpur Dhaka': (23.759, 90.361),
        'Mirpur Dhaka': (23.822, 90.365),
        'Block D Section 12 Mirpur Dhaka': (23.815, 90.358),
        'Dhanmondi Dhaka': (23.746, 90.372),
        'Block E Section 12 Mirpur Dhaka': (23.815, 90.36),
        'Sector 10 Uttara Dhaka': (23.879, 90.391),
        'Paikpara Ahmed Nagar Mirpur Dhaka': (23.812, 90.364),
        'Kallyanpur Mirpur Dhaka': (23.764, 90.365),
        'Section 12 Mirpur Dhaka': (23.816, 90.36),
        'Block B Section 12 Mirpur Dhaka': (23.817, 90.36),
        'Joar Sahara Dhaka': (23.848, 90.42),
        'Block C Section 12 Mirpur Dhaka': (23.818, 90.36),
        'West Shewrapara Mirpur Dhaka': (23.797, 90.366),
        'Shyamoli Dhaka': (23.762, 90.366),
        'PC Culture Housing Mohammadpur Dhaka': (23.762, 90.362),
        'Hazaribag Dhaka': (23.725, 90.408),
        'South Baridhara Residential AreaD. I. T. Project Badda Dhaka': (23.795, 90.423),
        'Block G Bashundhara R-A Dhaka': (23.811, 90.411),
        'Sector 13 Uttara Dhaka': (23.871, 90.394),
        'Uttar Badda Badda Dhaka': (23.796, 90.431),
        'Baitul Aman Housing Society Adabor Dhaka': (23.769, 90.353),
        'Section 1 Mirpur Dhaka': (23.814, 90.355),
        'Mohammadi Housing LTD. Mohammadpur Dhaka': (23.757, 90.361),
        'Badda Dhaka': (23.793, 90.419),
        'Sector 14 Uttara Dhaka': (23.873, 90.398),
        'Rupnagar R/A Mirpur Dhaka': (23.828, 90.362),
        'Shantinagar Dhaka': (23.738, 90.41),
        'Banani Dhaka': (23.780, 90.404),
        'Kakrail Dhaka': (23.742, 90.409),
        'Gulshan 2 Dhaka': (23.792, 90.407),
        'Tejgaon Dhaka': (23.760, 90.391),
        'Kalachandpur Dhaka': (23.812, 90.416),
        'Khilgaon Dhaka': (23.754, 90.427),
        'Jatra Bari Dhaka': (23.698, 90.450),
        'Shangkar Dhaka': (23.743, 90.367),
        'Nikunja Dhaka': (23.828, 90.409),
        'Jigatola Dhaka': (23.738, 90.369),
        'Rajabazar Dhaka': (23.756, 90.384),
        'West Malibag Dhaka': (23.746, 90.412),
        'Farmgate Tejgaon Dhaka': (23.753, 90.384) 
    }


        # Streamlit UI elements
        # st.title("Dhaka Rent Price Prediction")
        st.write(f"Selected city: {selected}")  
        st.sidebar.title("Input Features")
        location = st.sidebar.selectbox("Location", list(locations.keys()))
        area = st.sidebar.text_input("Area (sqft)", "")
        bed = st.sidebar.text_input("Number of Bedrooms", "")
        bath = st.sidebar.text_input("Number of Bathrooms", "")

        # Button to trigger prediction
        if st.sidebar.button("Predict Rent"):
            result = predict_price(location, float(area), int(bed), int(bath))
            st.sidebar.success(f'Estimated Rent(monthly): {result} Taka')

        # Display the map based on selected location
        st.subheader("Map of Selected Area")
        if location in locations:
            data = {
                'latitude': [locations[location][0]],
                'longitude': [locations[location][1]]
            }
            df = pd.DataFrame(data)
            st.map(df)

    if __name__ == '__main__':
        main()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
if selected == 'Entire-Dhaka':
    loaded_model = pickle.load(open('model_dhaka.pkl', 'rb'))
    def predict_price(location, area, bed, bath):
        # Assuming the locations are stored in a list or retrieved dynamically
        # loc_index = np.where(X.columns==location)[0][0]
        # Adjust the number of columns (features) based on your dataset
        x = np.zeros(418)  # Adjust this based on your model's feature set ||***********too important otherwise feature er akta error dei
        x[0] = area
        x[1] = bed
        x[2] = bath

        return np.round(loaded_model.predict([x])[0], 2)

    def main():
        # Load your dataset for Dhaka
        df_dhaka = pd.read_csv('dataset_rentology.csv')
        loc = df_dhaka["Location"].unique()


        # HTML template for heading
        # html_temp = """
        # <h4 style="color:white;text-align:left;"> Price Prediction in Dhaka City </h4>
        # """
        # st.markdown(html_temp, unsafe_allow_html=True)
        st.subheader('Please enter the required details:')
        st.write(f"Selected city: Dhaka")  

        # Dropdown for location selection
        location = st.selectbox("Location", loc)

        # Input fields for area, number of bedrooms, number of bathrooms
        area = st.text_input("Area (sqft)", "")
        bed = st.text_input("Number of Bedrooms", "")
        bath = st.text_input("Number of Bathrooms", "")

        result = ""

        # Button to trigger prediction
        if st.button("Predict Rent"):
            result = predict_price(location, float(area), int(bed), int(bath))

        # Display the predicted price
        st.success('Estimated Rent(monthly): {} Taka'.format(result))

    if __name__=='__main__':
        main()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if selected == 'Melbourne':
    from utils import * 

    # Load the pipeline from the joblib file
    loaded_pipeline = joblib.load('housing_pipeline.joblib')

    # # Streamlit app code
    # st.title("Melbourne Housing Price Prediction")
    st.write(f"Selected city: {selected}")  

    # Input form
    st.sidebar.title("Input Features")
    rooms = st.sidebar.number_input("Number of Rooms", value=3, min_value=1, max_value=10)
    distance = st.sidebar.number_input("Distance", value=10)
    bathroom = st.sidebar.number_input("Number of Bathrooms", value=2, min_value=0, max_value=10)
    landsize = st.sidebar.number_input("Landsize", value=600, min_value=1, max_value=50000)
    building_area = st.sidebar.number_input("Building Area", value=150, min_value=0, max_value=50000)
    year_built = st.sidebar.number_input("Year Built", value=2000, min_value=1800, max_value=2020)
    car = st.sidebar.number_input("Number of Car Spaces", value=2, min_value=0, max_value=10)

    suburb = st.sidebar.selectbox("Suburb", suburb_options)
    type = st.sidebar.selectbox("Type", ['H: House, Cottage,Villa,Semi-terrace', 'T: Townhouse, Dev-site, or other residential.', 'U: Unit, Duplex'])

    regionname = st.sidebar.selectbox("Region Name", region_options)

    # Create a dictionary with input data
    input_data = {
        'Rooms': rooms,
        'Distance': distance,
        'Bathroom': bathroom,
        'Landsize': landsize,
        'BuildingArea': building_area,
        'YearBuilt': year_built,
        'Car': car,
        'Suburb': suburb,
        'Type': type,
        'Regionname': regionname
    }

    # Convert the dictionary into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Make predictions using the loaded pipeline
    predictions = loaded_pipeline.predict(input_df)
    formatted_prediction = "${:,.2f}".format(predictions[0])

    # Display the prediction
    # st.write(f"Predicted Price: ${predictions[0]:,.2f}")
    st.write(f"House Price:", f"<span style='color:green; font-size:24px'>{formatted_prediction}</span>", unsafe_allow_html=True)


    # Display a map based on the selected suburb and region
    # st.header("Map")
    st.markdown("<h2 style='font-size: 24px;'>Map of Selected Area</h2>", unsafe_allow_html=True)

    latitude, longitude = get_coordinates_for_suburb(suburb)

    # Create a DataFrame with the latitude and longitude data
    data = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude]
    })

    st.map(data)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if selected == 'Bangaluru':

    # page title
    # st.title('Pirce prediction in Bangaluru City ')
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
        # html_temp = """
        # <h2 style="color:white;text-align:left;"> Price Prediction in Bangaluru City  </h2>
        # """

        # st.markdown(html_temp,unsafe_allow_html=True)
        st.subheader('Please enter the required details:')
        st.write(f"Selected city: {selected}")  

        location = st.selectbox("Location",loc)
        sqft = st.text_input("Sq-ft area","")
        bath = st.text_input("Number of Bathrooms","")
        bhk = st.text_input("Number of Bedrooms","")

        result=""


        # data = {

        #     'lat': [23.8103, 23.7949, 23.7806],
        #     'lon': [90.4125, 90.4043, 90.4167]
        # }
        # df = pd.DataFrame(data)

        # # Display map
        # st.map(df)



        if st.button("House Price in Rupee"):
            result=predict_price(location,sqft,bath,bhk)
        st.success('The output is {} in Lakhs'.format(result))


    if __name__=='__main__':
        main()
        





# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if selected == 'washington':

    # page title
    # st.title('Price prediction in Washington City')   
    data = pd.read_csv("final_washington_dataset.csv")
    def predict_price(zip_code, size, baths, beds):    
        # Assuming that the Washington model uses exactly 14 features
        # Replace these dummy values with the actual features your model uses
        x = np.zeros(14)  # Update this to reflect the actual number of features
        x[0] = zip_code
        x[1] = size
        x[2] = baths
        x[3] = beds
        # Add any other required feature assignments here
        
        return np.round(wasington_model_classifier.predict([x])[0], 2)

    def main(): 
        home = data
        zip_codes = home["zip_code"].unique()
        # st.title("Washington House Rate Prediction")
        # html_temp = """
        # <h2 style="color:white;text-align:left;">Price Prediction in Washington City </h2>
        # """

        # st.markdown(html_temp, unsafe_allow_html=True)
        # st.subheader('Please enter the required details:')
        st.write(f"Selected city: {selected}")  

        zip_code = st.selectbox("Zip Code", zip_codes)
        size = st.text_input("Sq-ft area", "")
        baths = st.text_input("Number of Bathrooms", "")
        beds = st.text_input("Number of Bedrooms", "")

        result = ""

        # if st.button("House Price in Dollars"):
        #     result = predict_price(zip_code, size, baths, beds)
        # st.success(f'The output is ${result}')

        if st.button("House Price in Dollars"):
            result = predict_price(zip_code, size, baths, beds)
            result_in_thousands = result / 10000
            st.success(f'The output is ${result_in_thousands}')        


    if __name__ == '__main__':
        main()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# if st.button("About"):
#     st.text("Please find the code at")
#     st.text("https://github.com/Habib-Un-Hemel/Rentology-s-Prediction-Feature")

st.markdown("""
    <style>
    .button {
        display: inline-block;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        outline: none;
        color: #FFFFFF;  /* Text color white */
        background-color: #FFFFFF;
        border: none;
        border-radius: 15px;
        box-shadow: 0 2px #999;
        position: absolute;
        top: 10px;
        right: 10px;
    }
    .button:hover {background-color: #FF4B4B}
    .button:active {
        background-color: #FF4B4B;
        box-shadow: 0 1px #666;
        transform: translateY(1px);
    }
    </style>
    <a href="https://github.com/Habib-Un-Hemel/Rentology-s-Prediction-Feature" class="button">GitHub</a>
    """, unsafe_allow_html=True)
