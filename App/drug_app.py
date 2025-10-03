import gradio as gr
import joblib
import os

# Try to load model from different possible paths
model_path = None
possible_paths = [
    "../Model/drug_pipeline.joblib",
    "./Model/drug_pipeline.joblib", 
    "Model/drug_pipeline.joblib",
    "drug_pipeline.joblib"
]

for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    raise FileNotFoundError("Could not find the trained model file")

pipe = joblib.load(model_path)

def predict_drug(age, gender, blood_pressure, cholesterol, na_to_k_ratio):
    features = [age, gender, blood_pressure, cholesterol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]
    label = f"Predicted Drug: {predicted_drug}"
    return label

inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Gender"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]
outputs = [gr.Label(num_top_classes=5)]
examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]
title = "Drug Classification"
description = "Enter the details to correctly identify Drug type?"
gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    theme=gr.themes.Soft(),
).launch()