# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
from MyCO2Proj import predict_co2_emission
import joblib
from database import connect_to_db, insert_emission_data

# Replace "model.joblib" with the actual filename of your trained model
model_filename = "model.joblib"
model = joblib.load(model_filename)

# Write the welcome message
st.write("# CO2 Prediction Project")
st.write("This web app allows you to predict CO2 emissions based on various inputs.")
st.write('''

         
        -What is Carbon Dioxide (CO2):

    CO2 is a colorless, odorless gas composed that is essential component of Earth's atmosphere 
    and is produced naturally through various processes, including respiration, volcanic activity,
    and decomposition.

        
        -Why is it Significant :

    It plays a crucial role in the Earth's climate system by trapping heat from the sun in what is
    known as the greenhouse effect, which is necessary to maintain a habitable climate on Earth by
    keeping temperatures within a certain range.

        -Is there correlation between Human Activities and CO2 Emissions:

    Human activities, particularly the burning of fossil fuels such as coal, oil, and gas, have 
    significantly increased CO2 levels in the atmosphere.One of the main sources of this increase 
    is the automobile industry, which is the subject of our study here


        -Consequences of High CO2 Levels:

    Climate Change: because higher temperatures contribute to melting glaciers, rising sea levels, and extreme weather events like hurricanes and droughts.
    Ocean Acidification: Excess CO2 is absorbed by the oceans, leading to ocean acidification, 
    which harms marine life and ecosystems.

        -The Importance of Monitoring CO2 Emissions:

    Understanding CO2 emissions and their sources is critical for developing effective climate 
    change mitigation strategies.If we can predict which car brand, type or which engine 
    contributes most to pollution, we could also predict future legislation, trade deals and 
    economic booms
         
         ''')


# Add a local image to the app
image_path = "1.png"
# Set the desired width for the image (twice as large)
desired_width = 2 * 278
# Display the image and center it using CSS
st.image(image_path, caption="CO2 vs Engine Power", width=desired_width, 
         use_column_width='center')

image_path = "2.png"
# Set the desired width for the image (twice as large)
desired_width = 2 * 278
# Display the image and center it using CSS
st.image(image_path, caption="Engine Power vs CO2", width=desired_width, 
         use_column_width='center')


st.write(
'''

      So as we see in the scatter plots above of the CO2 emission as function of Engine power and 
vise versa, and it shows something that we might have already speculated. Even for the same engine
power, the newer models, although to a very small extent, release less co2 for the same power, 
which is a trend that should continue. But also we can notice that sometimes even for lower power, 
the CO2 release is great, truly indicating the significance of technology on the efficiency of 
these engines
''')
 
image_path = "3.png"
# Set the desired width for the image (twice as large)
desired_width = 2 * 278
# Display the image and center it using CSS
st.image(image_path, caption="CO2 vs Fuel type", width=desired_width, 
         use_column_width='center')

st.write(
'''
On the other, above, we can see the distribution of CO2 emissions against the Fuel type, and 
although we do see difference especially at the lower end, all the fuel types look like they 
average around 200-300, which will help too conclude later in our project and would impact our
 findings 
''')


image_path = "4.png"
# Set the desired width for the image (twice as large)
desired_width = 2 * 278
# Display the image and center it using CSS
st.image(image_path, caption="CO2 vs Brands", width=desired_width, 
         use_column_width='center')

image_path = "7.png"
# Set the desired width for the image (twice as large)
desired_width = 2 * 278
# Display the image and center it using CSS
st.image(image_path, caption="CO2 share of major brands", width=desired_width, 
         use_column_width='center')


st.write(
'''
Here we have some interesting figures related to the brands. While it's common sense to blame 
sports cars to be the most pollutant, our charts says otherwise, and ther are many reasons for 
that
''')

st.write("-Less sports cars")
st.write("-More expensive means more efficient")
st.write("-Better fuel types")
st.write("-Adhering by the laws and speed limits")


image_path = "5.png"
# Set the desired width for the image (twice as large)
desired_width = 2 * 278
# Display the image and center it using CSS
st.image(image_path, caption="CO2 Frequency Distribution", width=desired_width, 
         use_column_width='center')

image_path = "6.png"
# Set the desired width for the image (twice as large)
desired_width = 2 * 278
# Display the image and center it using CSS
st.image(image_path, caption="CO2 Frequency Distribution", width=desired_width, 
         use_column_width='center')

st.write('''
         The above figures show us the distribution of CO2 emission, indicating that the bulk of
         the emissions is around 200 and between 150-250
         ''')

st.write("------------")
st.write("------------")
st.write("------------")
image_path = "9.png"
# Set the desired width for the image (twice as large)
desired_width = 2 * 278
# Display the image and center it using CSS
st.image(image_path, caption="Important Features Influencing our model", width=desired_width, 
         use_column_width='center')
st.write("------------")
st.write("------------")
st.write("------------")
# Ask the user to input a year
user_input_year = st.number_input("Enter a year to predict CO2 emissions:", min_value=2010, max_value=2025)

# Ask the user to input the engine power in kW
user_input_eng_pow = st.number_input("Enter the engine power (Eng_Pow) in kW:")

# Ask the user to input the brand
user_input_brand = st.text_input("Enter the brand:")

# Ask the user to input the fuel type
user_input_fuel_type = st.text_input("Enter the fuel type:")

# Make predictions using the predict_co2_emission function
predicted_co2, predicted_co2_mean, picked_predicted_co2 = predict_co2_emission(user_input_year, user_input_eng_pow, user_input_brand, user_input_fuel_type, model)


db_connection = connect_to_db()
insert_emission_data(db_connection, user_input_year, user_input_eng_pow, user_input_brand, user_input_fuel_type, predicted_co2_mean)
db_connection.close()


# Display the predicted CO2 emissions
st.write("Predicted CO2 Emissions:", predicted_co2)
st.write("Mean Predicted CO2 Emission:", predicted_co2_mean)
st.write("Picked Predicted CO2 Emission (Random):", picked_predicted_co2)
st.write("------------")
st.write("In Conclusion:")
st.write('''
         We, can say that we were successfully able to analyze our data, even though it was 
         limited by availability on one hand or our ability to read the data on the other. We 
         can also add that we were able to predict the CO2 emissions, although as it seems to not
         very accurate and great extend, and the reasons are many, so of which are:
             \n-As we see in the importance plot, the most important factors are the consumption
             \n-Consumption is a bit illogical to predict considering it is related to fuel and 
             engine power but also on miles driven and varies greatly between individuals
             \n-All the other factors have close distribution, meaning they yield similar results
         ''')
         
st.write('''
         But that doesn't mean unsignificance of our findings, first we are able to mildly predict
         some parameters, but also we were able to find which variables are the least important
         and which ones are more, allowing us to focus our efforts on said variable in the future
         Also, considering all the variables influencing CO2 emissions are very close, this 
         indicates to us, that a future approach should rethink the whole automobile industry,
         and perhaps focusing more on hybrid and electric cars which are proven to be more 
         enivronmentaly friendly
         ''')



