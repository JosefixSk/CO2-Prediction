import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib


# Read the datasets
df2010 = pd.read_csv(
    "C://Users//Dell//Downloads//16-fic_etiq_edition_mai_2010.csv", 
                     encoding='latin-1')
df2013 = pd.read_csv(
    "C://Users//Dell//Downloads//cl_JUIN_2013-complet3 2.csv", 
                     encoding='latin-1')
df2014 = pd.read_csv(
    "C://Users//Dell//Desktop//mars-2014-complete.csv", 
                     encoding='latin-1')
df2015 = pd.read_csv(
    "C://Users//Dell//Desktop//fic_etiq_edition_40-mars-2015.csv",
                     encoding='latin-1')



# Select the desired columns and rename them
df2010_selected = df2010[['MARQUE', 'puissance reelle', 'CO2', 'carburant',
                          'urb', 'ex-urb', 'mixte']]
df2010_selected.columns = ['Brand', 'Eng_Pow', 'CO2', 'Fuel Type', 
                           'Consumption_Urban', 'Consumption_Ex_Urban', 
                           'Consumption_Mix']
df2010_selected.loc[:, 'Year'] = 2010

df2013_selected = df2013[['Marque', 'Puissance maximale (kW)', 'CO2 (g/km)',
                          'Carburant', 'Consommation urbaine (l/100km)',
                          'Consommation extra-urbaine (l/100km)',
                          'Consommation mixte (l/100km)']]
df2013_selected.columns = ['Brand', 'Eng_Pow', 'CO2', 'Fuel Type',
                           'Fuel Consumption_Urban', 
                           'Fuel Consumption_Extra_Urban',
                           'Fuel Consumption_Mixed']
df2013_selected.loc[:, 'Year'] = 2013

df2014_selected = df2014[['lib_mrq', 'puiss_max', 'co2', 'cod_cbr', 
                          'conso_urb', 'conso_exurb', 'conso_mixte']]
df2014_selected.columns = ['Brand', 'Eng_Pow', 'CO2', 'Fuel Type', 
                           'Fuel Consumption_Urban', 
                           'Fuel Consumption_Extra_Urban', 
                           'Fuel Consumption_Mixed']
df2014_selected.loc[:, 'Year'] = 2014

df2015_selected = df2015[['lib_mrq_doss', 'puiss_max', 'co2_mixte', 'energ',
                          'conso_urb_93', 'conso_exurb', 'conso_mixte']]
df2015_selected.columns = ['Brand', 'Eng_Pow', 'CO2', 'Fuel Type', 
                           'Fuel Consumption_Urban', 
                           'Fuel Consumption_Extra_Urban', 
                           'Fuel Consumption_Mixed']
df2015_selected.loc[:, 'Year'] = 2015

# Clean up the 'Brand' column in each dataframe
df2010_selected['Brand'] = df2010_selected['Brand'].str.strip().str.upper()
df2013_selected['Brand'] = df2013_selected['Brand'].str.strip().str.upper()
df2014_selected['Brand'] = df2014_selected['Brand'].str.strip().str.upper()
df2015_selected['Brand'] = df2015_selected['Brand'].str.strip().str.upper()


# Concatenate the datasets
concatenated_data = pd.concat([
    df2010_selected, df2013_selected, df2014_selected, df2015_selected],
    ignore_index=True)

# Clean up the 'Brand' column
concatenated_data['Brand'] = concatenated_data['Brand'].str.strip().str.upper()


# Replace inconsistent brand names
concatenated_data['Brand'] = concatenated_data['Brand'].replace({
    'MERCEDES BENZ': 'MERCEDES-BENZ',
    'BMW I': 'BMW'
})

# Remove rows with missing brand names
concatenated_data.dropna(subset=['Brand'], inplace=True)

# Print unique brands and their counts
unique_brands = concatenated_data['Brand'].unique()
print("Brand", unique_brands)

brand_counts = concatenated_data['Brand'].value_counts()
print("Count", brand_counts)


# Convert Eng_Pow column to numeric
concatenated_data['Eng_Pow'] = pd.to_numeric(
    concatenated_data['Eng_Pow'], errors='coerce')

# Fill NaN values in Eng_Pow with the mean value
mean_eng_pow = concatenated_data['Eng_Pow'].mean()
concatenated_data['Eng_Pow'].fillna(mean_eng_pow, inplace=True)


# Clean up the 'carburant' column
concatenated_data['Fuel Type'] = concatenated_data['Fuel Type'].str.strip().str.upper()
# Replace 'nan' values in the 'Fuel Type' column with 'Unknown'
concatenated_data['Fuel Type'].fillna('Unknown', inplace=True)

# Remove commas and convert 'Fuel Consumption_Extra_Urban' column to numeric
concatenated_data['Fuel Consumption_Extra_Urban'] = concatenated_data['Fuel Consumption_Extra_Urban'].str.replace(',', '.').astype(float)
# Similarly, do the same for 'Fuel Consumption_Urban' and 'Fuel Consumption_Mixed' columns if they also have non-numeric values
concatenated_data['Fuel Consumption_Urban'] = concatenated_data['Fuel Consumption_Urban'].str.replace(',', '.').astype(float)
concatenated_data['Fuel Consumption_Mixed'] = concatenated_data['Fuel Consumption_Mixed'].str.replace(',', '.').astype(float)


# Convert CO2 column to numeric
concatenated_data['CO2'] = pd.to_numeric(concatenated_data['CO2'], 
                                         errors='coerce')
# Fill NaN values in CO2 with the mean value
mean_co2 = concatenated_data['CO2'].mean()
concatenated_data['CO2'].fillna(mean_co2, inplace=True)


# Choose the CO2 column as the target variable
target_variable = 'CO2'
# Filter the dataset based on the target variable
target_data = concatenated_data[concatenated_data[target_variable].notnull()]


# Generate a color map for the years
cmap = plt.get_cmap('tab10')

# Plot the data
plt.scatter(target_data['Eng_Pow'], target_data[target_variable],
            c=target_data['Year'], cmap=cmap)
# Customize the plot
plt.xlabel('Engine Power')
plt.ylabel(target_variable)
plt.title('CO2 Emission vs Engine Power')
plt.colorbar(label='Year')
# Show the plot
plt.show()

# Plot the data with flipped axes
plt.scatter(concatenated_data[target_variable], concatenated_data['Eng_Pow'],
            c=concatenated_data['Year'], cmap=cmap)
# Customize the plot
plt.xlabel(target_variable)
plt.ylabel('Engine Power')
plt.title('Engine Power vs CO2 Emission')
plt.colorbar(label='Year')
# Show the plot
plt.show()

# Create a box plot of CO2 emissions grouped by fuel type and year
plt.figure(figsize=(12, 6))
sns.boxplot(data=concatenated_data, x='Fuel Type', y=target_variable)
# Customize the plot
plt.xlabel('Fuel Type')
plt.ylabel(target_variable)
plt.title('CO2 Emission by Fuel Type and Year')
plt.xticks(rotation=90)
# Show the plot
plt.show()


# Convert Brand column to string
concatenated_data['Brand'] = concatenated_data['Brand'].astype(str)
# Create a bar graph
plt.figure(figsize=(12, 6))
sns.barplot(data=concatenated_data, x='Brand', y=target_variable)
# Customize the plot
plt.xlabel('Brand')
plt.ylabel(target_variable)
plt.title('CO2 Emission by Brand')
# Rotate x-axis labels for better readability
plt.xticks(rotation=90)
# Show the plot
plt.show()

# Plot histogram of CO2 emissions
plt.figure(figsize=(8, 6))
sns.histplot(data=concatenated_data, x=target_variable, bins=30, kde=True)
plt.xlabel('CO2 Emission')
plt.ylabel('Frequency')
plt.title('Distribution of CO2 Emissions')
plt.show()

# Plot distribution plot of CO2 emissions
plt.figure(figsize=(8, 6))
sns.kdeplot(data=concatenated_data, x=target_variable)
plt.xlabel('CO2 Emission')
plt.ylabel('Density')
plt.title('Distribution of CO2 Emissions')
plt.show()

# Calculate the total CO2 emissions for each brand
brand_emissions = concatenated_data.groupby('Brand')['CO2'].sum()

# Sort the brands based on total CO2 emissions in descending order
top_emitters = brand_emissions.sort_values(ascending=False).head(10)

# Plot a pie chart of top CO2 emitters
plt.figure(figsize=(8, 8))
plt.pie(top_emitters, labels=top_emitters.index, autopct='%1.1f%%')
plt.title('Top CO2 Emitters by Brand')
plt.show()

# Calculate the correlation matrix
correlation_matrix = concatenated_data.corr()

# Generate a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Split the data into features (X) and target variable (y)
X = concatenated_data.drop(target_variable, axis=1)
y = concatenated_data[target_variable]

# Clean up the 'Brand' column in the concatenated dataset
concatenated_data['Brand'] = concatenated_data['Brand'].str.strip().str.upper()

# Get the complete list of all possible brands
all_brands = concatenated_data['Brand'].unique()

# Apply one-hot encoding to the 'Brand' feature using the complete list of brands in concatenated_data
brand_encoded_data = pd.get_dummies(concatenated_data['Brand'], prefix='Brand', columns=all_brands)

# Apply one-hot encoding to the 'Brand' feature using the complete list of brands in concatenated_data
X_encoded = pd.get_dummies(X, columns=['Brand'], prefix='Brand', dtype=int)

# Split the data into features (X) and target variable (y)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting datasets
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

print("Column names in X_encoded:")
print(X_encoded.columns,"Shape of X_encoded:", X_encoded.shape)

# Apply label encoding to the 'Brand' column using the complete list of brands
#label_encoder = LabelEncoder()
# Select only the numerical columns
X_train_numeric = X_train.select_dtypes(include=[np.number])
X_test_numeric = X_test.select_dtypes(include=[np.number])


# Create a pipeline with an imputer and a scaler
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler())
 

# Fit and transform the training data
X_train_scaled = pipeline.fit_transform(X_train_numeric)
# Transform the testing data using the fitted pipeline
X_test_scaled = pipeline.transform(X_test_numeric)


# Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)
# Model Evaluation
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Linear Regression")
print("Training set:")
print("MSE: ", mse_train)
print("MAE: ", mae_train)
print("R-squared: ", r2_train)

print("Testing set:")
print("MSE: ", mse_test)
print("MAE: ", mae_test)
print("R-squared: ", r2_test,"\n")


# Model Training
model = GradientBoostingRegressor()
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Gradient Boosting:")
print("Training set:")
print("MSE: ", mse_train)
print("MAE: ", mae_train)
print("R-squared: ", r2_train)

print("Testing set:")
print("MSE: ", mse_test)
print("MAE: ", mae_test)
print("R-squared: ", r2_test, "\n")



# Model Training
model = DecisionTreeRegressor()
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Decision Tree:")
print("Training set:")
print("MSE: ", mse_train)
print("MAE: ", mae_train)
print("R-squared: ", r2_train)

print("Testing set:")
print("MSE: ", mse_test)
print("MAE: ", mae_test)
print("R-squared: ", r2_test,"\n")
joblib.dump(model, "model.joblib")

feature_importances = model.feature_importances_
# Get the feature names
feature_names = X_train_numeric.columns

# Sort the features and importances in descending order
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_importances = feature_importances[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

# Create a bar plot for feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, tick_label=sorted_feature_names)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importances of the Decision Tree Model')
plt.show()


# Ask the user to input a year
user_input_year = int(input("Enter a year to predict CO2 emissions: "))

# Ask the user to input the engine power in kW
user_input_eng_pow = float(input("Enter the engine power (Eng_Pow) in kW: "))

# Ask the user to input the brand
user_input_brand = input("Enter the brand: ")

# Ask the user to input the fuel type
user_input_fuel_type = input("Enter the fuel type: ")

# Step 1: Create a copy of the X_encoded dataset
input_row_copy = X_encoded.copy()

# Step 2: Modify the 'Year' column of the copied dataset with the user's input
input_row_copy['Year'] = user_input_year

# Step 3: Modify the 'Eng_Pow' column of the copied dataset with the user's input
input_row_copy['Eng_Pow'] = user_input_eng_pow

# Step 4: Set the 'Brand' and 'Fuel Type' columns of the copied dataset with the user's input
input_row_copy['Brand_' + user_input_brand] = 1
input_row_copy['Fuel Type_' + user_input_fuel_type] = 1

# Step 5: Preprocess the modified dataset using the same pipeline used during model training
# Select only the numeric features (as used during training)
input_row_numeric = input_row_copy[X_train_numeric.columns]

# Scale the input dataset using the fitted pipeline
input_row_scaled = pipeline.transform(input_row_numeric)

# Step 6: Make predictions using the trained model
predicted_co2 = model.predict(input_row_scaled)

print("Predicted CO2 Emissions:", predicted_co2)

predicted_co2_mean = predicted_co2.mean()
print("Mean Predicted CO2 Emission:", predicted_co2_mean)


# Pick a random predicted CO2 emission value from the array
picked_predicted_co2 = np.random.choice(predicted_co2)

print("Predicted CO2 Emissions:", predicted_co2)
print("Picked Predicted CO2 Emission (Random):", picked_predicted_co2)

def predict_co2_emission(year, eng_pow, brand, fuel_type, model):
    # Create a copy of the X_encoded dataset
    input_row_copy = X_encoded.copy()

    # Modify the 'Year' column of the copied dataset with the user's input
    input_row_copy['Year'] = year

    # Modify the 'Eng_Pow' column of the copied dataset with the user's input
    input_row_copy['Eng_Pow'] = eng_pow

    # Set the 'Brand' and 'Fuel Type' columns of the copied dataset with the user's input
    input_row_copy['Brand_' + brand] = 1
    input_row_copy['Fuel Type_' + fuel_type] = 1

    # Preprocess the modified dataset using the same pipeline used during model training
    # Select only the numeric features (as used during training)
    input_row_numeric = input_row_copy[X_train_numeric.columns]

    # Scale the input dataset using the fitted pipeline
    input_row_scaled = pipeline.transform(input_row_numeric)

    # Make predictions using the trained model
    predicted_co2 = model.predict(input_row_scaled)

    # Calculate the mean predicted CO2 emission
    predicted_co2_mean = predicted_co2.mean()

    # Pick a random predicted CO2 emission value from the array
    picked_predicted_co2 = np.random.choice(predicted_co2)

    return predicted_co2, predicted_co2_mean, picked_predicted_co2


#cd C:\Users\Dell\.spyder-py3\CO2
#streamlit run streamlit_app.py