# Drug Classification ML Pipeline

A complete machine learning pipeline for drug classification with CI/CD automation and deployment to Hugging Face Spaces.

## 🏗️ Project Structure

```
├── App/                    # Gradio web application
│   ├── drug_app.py        # Main app file
│   ├── README.md          # HF Space metadata
│   └── requirements.txt   # App dependencies
├── Data/                  # Dataset storage
│   └── drug.csv          # Drug classification dataset
├── Model/                 # Trained models (generated)
│   └── drug_pipeline.joblib
├── Results/               # Model metrics and plots (generated)
│   ├── metrics.txt       
│   └── model_results.png
├── .github/workflows/     # CI/CD automation
│   └── ci.yml            # GitHub Actions workflow
├── Makefile              # Automation commands
├── requirements.txt      # Project dependencies
├── train.py             # Training script
└── notebook.ipynb       # Experimentation notebook
```

## 🚀 Features

- **Reproducible ML Pipeline**: Standardized data processing and model training
- **CI/CD Automation**: GitHub Actions for automated training and deployment
- **Interactive Web App**: Gradio interface for model predictions
- **Model Versioning**: Automated model and results tracking
- **Cloud Deployment**: Automatic deployment to Hugging Face Spaces

## 📊 Dataset

The drug classification dataset contains patient information:
- **Age**: Patient age (15-74)
- **Sex**: Gender (M/F)
- **BP**: Blood pressure (HIGH/LOW/NORMAL)
- **Cholesterol**: Cholesterol level (HIGH/NORMAL)
- **Na_to_K**: Sodium to potassium ratio (6.2-38.2)
- **Drug**: Target drug class (drugA, drugB, drugC, drugX, drugY)

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Git
- GitHub account
- Hugging Face account

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/varshavenkatesan/MachineLearning.git
   cd MachineLearning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train.py
   ```

4. **Run the Gradio app**
   ```bash
   python App/drug_app.py
   ```

### GitHub Repository Setup

1. **Create new repository**
   - Go to GitHub and create a new repository named `MachineLearning`
   - Initialize with README and Python .gitignore

2. **Add Hugging Face token as secret**
   - Go to Settings → Secrets and variables → Actions
   - Add new secret named `HF` with your Hugging Face write token

### Hugging Face Space Setup

1. **Create new Space**
   - Go to Hugging Face Spaces
   - Create new Space named `MachineLearning`
   - Choose Gradio SDK

2. **Get your write token**
   - Profile → Settings → Access Tokens
   - Create new token with write permissions

## 🔄 CI/CD Pipeline

The pipeline automatically:

1. **Continuous Integration**
   - Installs dependencies
   - Trains the model
   - Evaluates performance
   - Generates metrics and plots

2. **Continuous Deployment**
   - Creates update branch with model/results
   - Deploys app, model, and results to HF Space
   - Updates the live application

### Manual Commands

```bash
# Install dependencies
make install

# Train model
make train

# Evaluate and create report
make eval

# Update branch with results
make update-branch

# Deploy to Hugging Face
make deploy
```

## 📈 Model Performance

The Random Forest classifier uses:
- **Preprocessing**: Ordinal encoding for categorical features, median imputation and scaling for numerical features
- **Algorithm**: Random Forest with 100 estimators
- **Evaluation**: Accuracy and F1-score metrics
- **Visualization**: Confusion matrix plot

## 🌐 Deployment

The app is automatically deployed to:
- **Hugging Face Space**: `https://huggingface.co/spaces/varshavenkatesan/MachineLearning`
- **GitHub Repository**: `https://github.com/varshavenkatesan/MachineLearning`

## 🛠️ Technology Stack

- **ML Framework**: scikit-learn
- **Web Framework**: Gradio
- **Model Persistence**: joblib
- **CI/CD**: GitHub Actions
- **Deployment**: Hugging Face Spaces
- **Automation**: Makefile
- **Version Control**: Git

## 📝 Usage Example

```python
# Load the trained pipeline
import joblib
pipe = joblib.load('Model/drug_pipeline.joblib')

# Make prediction
features = [30, "M", "HIGH", "NORMAL", 15.4]  # age, sex, bp, cholesterol, na_to_k
prediction = pipe.predict([features])[0]
print(f"Predicted Drug: {prediction}")
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Push to GitHub (triggers CI/CD)
5. Create pull request

## 📄 License

This project is licensed under the Apache 2.0 License.

## 📞 Contact

- **Author**: Varsha Venkatesan
- **GitHub**: [@varshavenkatesan](https://github.com/varshavenkatesan)
- **Hugging Face**: [@varshavenkatesan](https://huggingface.co/varshavenkatesan)