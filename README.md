This is the main documentation for your project, explaining what it is and how to use it.

# Mobiticket Transport Demand Prediction

This project aims to build a machine learning model to predict the number of seats that Mobiticket can expect to sell for each ride. The model uses historical ticket data to forecast demand based on factors like the route, time of day, and day of the week.

## 📂 Project Structure

mobiticket-demand-prediction/
│
├── data/
│   ├── train_revised.csv       # Raw transactional data
│   ├── processed_data.csv      # Processed data ready for modeling
│   ├── X_test.csv              # Test features for evaluation
│   └── y_test.csv              # Test target for evaluation
│
├── models/
│   └── demand_predictor.pkl    # Saved (trained) model pipeline
│
├── notebooks/
│   └── EDA_and_Model.ipynb     # Jupyter Notebook for exploration and experimentation
│
├── src/
│   ├── data_preprocessing.py   # Script to process raw data
│   ├── model_training.py       # Script to train and save the model
│   └── model_evaluation.py     # Script to evaluate the trained model
│
├── report/
│   └── Data_Science_Capstone_Project.pdf # Original project description
│
├── requirements.txt            # List of Python libraries
├── README.md                   # This file
└── .gitignore                  # Specifies files to be ignored by Git


## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://your-repository-url/mobiticket-demand-prediction.git
    cd mobiticket-demand-prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 How to Run the Project

Execute the scripts from the `src` folder in the following order.

### **Step 1: Process the Raw Data**
This script will clean the raw data, perform feature engineering, and save the result in the `data/` directory.

```bash
python src/data_preprocessing.py
Step 2: Train the Model
This script will load the processed data, train the Random Forest model, and save the trained pipeline to the models/ directory. It also saves the test set in data/ for evaluation.

Bash

python src/model_training.py
Step 3: Evaluate the Model
This script will load the saved model and the test set to evaluate its performance.

Bash

python src/model_evaluation.py
📈 Results
The project uses a Random Forest Regressor model. The evaluation on the unseen test set yielded the following results:

Mean Absolute Error (MAE): 4.54 (The model's predictions are, on average, off by about 4-5 seats).

R-squared (R²): 0.32 (The model explains 32% of the variance in ticket sales).

For a detailed walkthrough of the exploratory data analysis (EDA) and model building process, please see the Jupyter Notebook at notebooks/EDA_and_Model.ipynb.