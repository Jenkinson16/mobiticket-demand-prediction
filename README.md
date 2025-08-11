**Mobiticket Transport Demand Prediction**

This project aims to build a machine learning model to predict the number of seats that Mobiticket can expect to sell for each ride. The model uses historical ticket data to forecast demand based on factors like the route, time of day, and day of the week.

📂 Project Structure
```
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
```

⚙️ Setup and Installation
Clone the repository:
```
git clone https://your-repository-url/mobiticket-demand-prediction.git
cd mobiticket-demand-prediction
```
Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```
Install the required packages:
```
pip install -r requirements.txt
```

🚀 How to Run the Project
Execute the scripts from the src folder in the following order.

Step 1: Process the Raw Data
This script will clean the raw data, perform feature engineering, and save the result in the data/ directory.
```
python src/data_preprocessing.py
```
Step 2: Train the Model
This script will load the processed data, train the Random Forest model, and save the trained pipeline to the models/ directory. It also saves the test set in data/ for evaluation.
```
python src/model_training.py
```
Step 3: Evaluate the Model
This script will load the saved model and the test set to evaluate its performance.
```
python src/model_evaluation.py
```
**📈 Results**

The model's performance was evaluated on the unseen test data with the following results:

**R-squared (R²):** 0.61

**Interpretation:** The model successfully explains 61% of the variability in the number of tickets sold, indicating a strong relationship between the features and the target variable.

**Mean Absolute Error (MAE):** 3.22

**Interpretation:** On average, the model's prediction for the number of seats sold is off by approximately 3 seats.

**Mean Squared Error (MSE):** 20.57

For a detailed walkthrough of the exploratory data analysis (EDA) and model building process, please refer to the Jupyter Notebook at notebooks/EDA_and_Model.ipynb.


