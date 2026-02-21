import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pycaret.classification import setup as class_setup
from pycaret.regression import setup as reg_setup, compare_models, pull, create_model, finalize_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



# Data reading
def load_data(file):
    file_extension = file.name.split('.')[-1]
    if file_extension.lower() == 'csv':
        return pd.read_csv(file)
    elif file_extension.lower() in ['xls', 'xlsx']:
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")



# Data Analysis
def plot_handling(data, plot_type, x_var=None, y_var=None, hue=None):
    if plot_type == 'pairplot':
        st.write("Pairplot:")
        sns.pairplot(data)
    else:
        plt.figure(figsize=(8, 6))
        if plot_type == 'scatter':
            st.write("Scatter Plot:")
            sns.scatterplot(x=data[x_var], y=data[y_var], hue=data[hue])
        elif plot_type == 'histogram':
            st.write("Histogram:")
            sns.histplot(data[x_var])
        elif plot_type == 'barplot':
            st.write("Bar Plot:")
            sns.barplot(x=data[x_var], y=data[y_var], hue=data[hue])
        elif plot_type == 'countplot':
            st.write("Count Plot:")
            sns.countplot(x=data[x_var], hue=data[hue])
        elif plot_type == 'boxplot':
            st.write("Box Plot:")
            sns.boxplot(x=data[x_var], y=data[y_var], hue=data[hue])

    st.pyplot()



def null_handling(data, cat_handling, reg_handling):
    null_replacement = None
    
    for col in data.select_dtypes(include=[np.number]).columns:
        if cat_handling == 'None':
            st.info("_Select a value to replace missing regression values, **leaving it as 'None' is highly not recommended**_")
            return data
        elif cat_handling == 'mode':
            null_replacement = data[col].mode().iloc[0]
        elif cat_handling == 'additional class':
            null_replacement = 'MISSING'
    
        data[col].fillna(null_replacement, inplace=True)
    
    for col in data.select_dtypes(include=[np.number]).columns:
        if reg_handling == 'None':
            st.info("_Select a value to replace missing regression values, **leaving it as 'None' is highly not recommended**_")
            break
        elif reg_handling == 'mean':
            null_replacement = np.mean(data[col])
        elif reg_handling == 'median':
            null_replacement = np.median(data[col])
        elif reg_handling == 'mode':
            null_replacement = data[col].mode().iloc[0]
    
        data[col].fillna(null_replacement, inplace=True)
    
    st.write("The dataframe has been cleansed of null values successfully!")
    st.dataframe(data)
    return data



def encode_handling(data, encode_cat, column=None):
    if encode_cat == 'None':
        st.info("_Select a method to encode categorical data, **leaving it as 'None' is highly not recommended**_")
        return data
    elif encode_cat == 'label':
        le = LabelEncoder()
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = le.fit_transform(data[col])
    elif encode_cat == 'one-hot':
        for col in data.select_dtypes(include=['object']).columns:
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = encoder.fit_transform(data[[col]])
            new_encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))
            
            data = pd.concat([data, new_encoded_df], axis=1)
            data = data.drop(columns=[col])
    
    st.write("The dataframe has been encoded successfully!")
    st.dataframe(data)
    return data



# Model Training
def model_training(data, model_choice, target_column, selected_features):
    # Setup PyCaret with auto target type detection
    st.write("_**conducting traing on the following data frame, please wait...**_")
    st.dataframe(data)
    
    # Automatically detect whether the values are classified or regressed
    if data[target_column].dtype == 'object' or data[target_column].nunique() < 20:
        task = "classification"
    else:
        task = "regression"

    # Setup PyCaret with user-selected features
    setup_data = data[selected_features + [target_column]]
        
    
    if model_choice == "Auto-Detect Best Model":
        # Automatically detect whether the values are classified or regressed
        if task == "classification":
            class_setup(data=setup_data, target=target_column)
        else:
            reg_setup(data=setup_data, target=target_column)
        
        best_model = compare_models()
        compare_df = pull()
        
        st.write("### Best Model selection by AutoML")
        st.dataframe(compare_df)
        
        best_model_name = pull().iloc[0]["Model"]
        
        st.write(f"### The Best Model is '{best_model_name}'")
        st.write(f"_**Task type: '{task}'**_")
    else:
        clf = None
    
        if model_choice == "Logistic/Linear":
            if task == "classification":
                clf = LogisticRegression()
            else:
                clf = LinearRegression()

        elif model_choice == "Decision Tree":
            if task == "classification":
                clf = DecisionTreeClassifier()
            else:
                clf = DecisionTreeRegressor()

        elif model_choice == "Random Forest":
            if task == "classification":
                clf = RandomForestClassifier()
            else:
                clf = RandomForestRegressor()

        st.write(f"### Model: {model_choice}")
        st.write(f"_**Task type: '{task}'**_")
        
        x = data[selected_features]
        y = data[target_column]
        
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        
        cv_scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy' if task == 'classification' 
            else 'neg_mean_squared_error')
    
        clf.fit(x_train, y_train)
        train_preds = clf.predict(x_train)
        test_preds = clf.predict(x_test)

        if task == "classification":
            st.write("\n")
            st.write(f"_Train Accuracy: {accuracy_score(y_train, train_preds):.2f}_")
            st.write(f"**Test Accuracy: {accuracy_score(y_test, test_preds):.2f}**")
            st.write(f"**Cross-Validation Accuracy: {cv_scores.mean():.2f}, Std: +/- {cv_scores.std():.2f}**")
        else:
            train_mse = mean_squared_error(y_train, train_preds)
            test_mse = mean_squared_error(y_test, test_preds)
            st.write("\n")
            st.write(f"_Train Mean Squared Error: {train_mse:.2f}_")
            st.write(f"**Test Mean Squared Error: {test_mse:.2f}**")
            st.write("\n")
            st.write(f"_Train R-squared: {r2_score(y_train, train_preds):.2f}_")
            st.write(f"**Test R-squared: {r2_score(y_test, test_preds):.2f}**")
            st.write(f"**Cross-Validation MSE: {(-cv_scores.mean()):.2f}, Std: +/- {cv_scores.std():.2f}**")
        




# Streamlit App

st.title("Machine Learning Model Trainer")
st.write("_Upload your dataset, select a target column, and choose a model to train._")

# File upload
file = st.file_uploader('Upload your CSV or Excel file', type=['csv', 'xlsx'])

# Variable to store the loaded data
data = None
columns = ['None']


# hide other widgets until a file is uploaded
if file:
    # Handle file upload
    data = load_data(file)
    columns = data.columns.tolist()
    
    # Show the dataframe
    st.divider()
    st.dataframe(data)
    st.divider()


    st.write("## Exploratory Data Analysis")
    
    # Dropdown for plot type selection
    st.write("### Plot drawing")
    plot_type = st.selectbox('Plot Type', 
        options=[None, 'pairplot', 'scatter', 'histogram', 'barplot', 'countplot', 'boxplot'])

    x_var, y_var, hue = columns[0], columns[0], columns[0]
    # Create a row for the three selectboxes
    req_plot_opt = []
    col1, col2, col3 = st.columns(3)
    if plot_type in ['scatter', 'histogram', 'barplot', 'countplot', 'boxplot']:
        with col1:
            x_var = st.selectbox('X value', options=[None] + columns)
            if x_var == None:
                req_plot_opt.append('X value')
    if plot_type in ['scatter', 'barplot', 'boxplot']:
        with col2:
            y_var = st.selectbox('Y value', options=[None] + columns)
            if y_var == None:
                req_plot_opt.append('Y value')
    if plot_type in ['scatter', 'barplot', 'countplot', 'boxplot']:
        with col3:
            hue = st.selectbox('Hue', options=[None] + columns)
            if hue== None:
                req_plot_opt.append('Hue')
    
    
    # Only show the plot if all necessary variables are properly initialized
    if plot_type == None:
        st.info("_Select a plot type to visualize the data, this can help you to understand the data more_")
    elif hue and x_var and y_var:
        st.write(f"## Showing {plot_type} Plot:")
        plot_handling(data, plot_type, x_var, y_var, hue)
    else:
        st.info(f'_Please select values for {req_plot_opt}_')
        


    # Selectbox for encoding categorical data
    st.write("### Data Encoding Method")
    encode_cat = st.selectbox('Encoding method', options=['None', 'label', 'one-hot'])
    data = encode_handling(data, encode_cat)
    columns = data.columns.tolist()

    # SelectSlider for setting null values for numerical data
    st.write("### Handling Messing Values")
    cat_handling = st.selectbox('Categorical Values Replacement', options=['None', 'mode', 'additional class'])
    reg_handling = st.selectbox('Regression Values Replacement', options=['None', 'mean', 'median', 'mode'])
    data = null_handling(data, cat_handling, reg_handling)
    columns = data.columns.tolist()



    st.divider()
    
    st.write("## Model Training")
    # Select model, or let the app decide
    st.info("_Select a model to train_"
    + "\n- **Auto-Detect Best Mode** uses AutoML to find the best model. (Recommended)"
    + "\n- Or you can **manually** select a specific model.")

    model_choice = st.radio("Choose a model",
        ["Auto-Detect Best Model", "Logistic/Linear", "Decision Tree", "Random Forest"])

    if model_choice != 'Auto-Detect Best Model':
        st.warning("Manually selecting a model might lead to problems, "
            + "as not all models will perform well on all data, use Auto-Detect for best results.")

    # Dropdowns for target selection
    selected_features = st.multiselect('Target Features (X)', options=data.columns)
    target_column = st.selectbox('Target Variable (Y)', options=[None] + columns)
    
    # Button to trigger model training
    if st.button('Train Model'):
        if target_column:
            final_model = model_training(data, model_choice, target_column, selected_features)
            
            st.divider()
            st.write("### Thank you for using my App!")
            st.markdown("**Please check out my [Github Page](https://github.com/VenStudio/) for other cool apps and games!**")
            st.markdown("_Special thanks to [ElectoPI](https://electropi.ai/) for giving me the chance to make this awesome app!_")
            st.markdown("\n")
            st.markdown("   - This website was made with [Streamlit](https://streamlit.io/)")
            st.markdown("   - The Best-Model-Selection is powered by [Pycaret](https://pycaret.org/)")
        else:
            st.warning("_Please select a target column to train the model_")
