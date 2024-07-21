import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from src.pipeline import pipeline_run  # Assuming your `main` function is in `src`

# Function to load the model
def load_model():
    try:
        with open("model.pkl", "rb") as pickle_in:
            return pickle.load(pickle_in)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to train the model
def train_model():
    try:
        pipeline_run()  # Calls your training function from `src`
        st.success("Model trained and saved successfully!")
    except Exception as e:
        st.error(f"Error during training: {e}")

# Streamlit app
def run_app():
    st.title("Electricity Bill Prediction")

    mode = st.sidebar.selectbox("Select Mode", ["Prediction", "Training"])

    if mode == "Prediction":
        st.markdown("""
        ### Created By : Bala Murugan
        #### LinkedIn : https://www.linkedin.com/in/balamurugan14/
        """)
        
        # Load model for prediction
        regressor = load_model()
        
        if regressor is None:
            st.stop()  # Stops the execution if model loading fails

        html_temp = """
        <div style="background-color:slateblue;padding:10px">
        <h2 style="color:black;text-align:center;">Electricity Bill Prediction</h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        years_list = list(range(2001, 2024 + 1))
        months_list = list(range(1, 12 + 1))
        state_description_list = ['Wyoming', 'New England', 'South Carolina', 'South Dakota',
                                    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
                                    'West Virginia', 'Wisconsin', 'Alabama', 'Louisiana', 'Maine',
                                    'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
                                    'Mississippi', 'Middle Atlantic', 'Pacific Contiguous',
                                    'Pacific Noncontiguous', 'U.S. Total', 'Missouri', 'Montana',
                                    'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
                                    'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
                                    'Oregon', 'Rhode Island', 'Alaska', 'Arizona', 'Georgia', 'Hawaii',
                                    'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
                                    'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
                                    'District of Columbia', 'Florida', 'East North Central',
                                    'West North Central', 'South Atlantic', 'East South Central',
                                    'West South Central', 'Mountain', 'Pennsylvania'] 
        
        sector_name_list = ['all sectors', 'commercial', 'industrial', 'other', 'residential', 'transportation']

        year = st.selectbox("Year", years_list)
        month = st.selectbox("Month", months_list)
        stateDescription = st.selectbox("State Description", state_description_list)
        sectorName = st.selectbox("Sector Name", sector_name_list)
        customers = st.text_input("Customers", "")
        revenue = st.text_input("Revenue", "")
        sales = st.text_input("Sales", "")

        input_data = {
            "year": [year], 
            "month": [month], 
            "stateDescription": [stateDescription],
            "sectorName": [sectorName], 
            "customers": [customers],
            "revenue": [revenue],
            "sales": [sales]
        }

        dataframe = pd.DataFrame(input_data)

        label_encoder = LabelEncoder()
        dataframe['sectorName'] = label_encoder.fit_transform(dataframe['sectorName'])
        dataframe['stateDescription'] = label_encoder.fit_transform(dataframe['stateDescription'])
        dataframe.drop(['customers', 'revenue', 'sales'], axis=1, inplace=True)

        if st.button("Predict Bill Price"):
            result = regressor.predict(dataframe)
            st.success(f'The Electricity Bill Price: {result[0]}')

    elif mode == "Training":
        st.title("Train the Model")

        if st.button("Start Training"):
            train_model()

# Run the app
if __name__ == '__main__':
    run_app()
