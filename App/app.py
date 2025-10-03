import gradio as gr
import joblib
import os

# Load the model
try:
    pipe = joblib.load("drug_pipeline.joblib")
    print("✅ Model loaded successfully from drug_pipeline.joblib")
except FileNotFoundError:
    print("❌ Model file not found. Please ensure drug_pipeline.joblib is in the same directory.")
    raise

def predict_drug(age, gender, blood_pressure, cholesterol, na_to_k_ratio):
    """Predict drugs based on patient features.
    
    Args:
        age (int): Age of patient
        gender (str): Gender of patient
        blood_pressure (str): Blood pressure level
        cholesterol (str): Cholesterol level
        na_to_k_ratio (float): Ratio of sodium to potassium in blood
    
    Returns:
        str: Predicted drug label
    """
    features = [age, gender, blood_pressure, cholesterol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]
    label = f"Predicted Drug: {predicted_drug}"
    return label

# Define the input components
inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Gender"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]

# Define output
outputs = [gr.Label(num_top_classes=5)]

# Example inputs
examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]

# App metadata
title = "Drug Classification"
description = "Enter the patient details to predict the appropriate drug type. This ML model uses Random Forest to classify drugs based on patient characteristics."

# Create and launch the interface
gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    theme=gr.themes.Soft(),
).launch()