import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load the trained model and pre-processing pipeline
model = joblib.load("housepricepredictor/house_price_predictor_model.joblib")

# Feature names for column transformer (same as those in training)
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Sample column transformer setup (to mimic what was used in the training pipeline)
column_transformer = ColumnTransformer(
    transformers=[
        ('one_hot', OneHotEncoder(), categorical_features)
    ], remainder='passthrough'
)

# Pre-processing function (transform the input data)
def pre_process(input_data):
    """
    This function takes raw input data and preprocesses it according to the pipeline used in training.
    It handles one-hot encoding for categorical features and ensures the data is in the correct format.
    """
    # Convert input_data (assumed to be a dictionary) to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply the column transformer (one-hot encoding and numerical handling)
    transformed_input = column_transformer.transform(input_df)
    
    # Convert the result to a list for further use in prediction
    return transformed_input.tolist()

# Post-processing function (to handle the output from the model)
def post_process(prediction):
    """
    This function handles the output of the prediction and formats it for end users.
    It can be adjusted to round or format the output.
    """
    # In this case, we'll just return the predicted price.
    return prediction[0]

