# Soccer Analytics Project ⚽

This project is a comprehensive pipeline for soccer analytics, built using Statsbomb open data. It includes data collection, feature engineering, and machine learning models to predict match outcomes and player performance.

## 🌟 Key Features

* **Model #1: Match Outcome Predictor**
    * Downloads multiple seasons of La Liga event data.
    * Calculates rolling team statistics (e.g., avg. xG, avg. possession).
    * Trains a model (Logistic Regression & Random Forest) to predict match outcomes (Home Win, Draw, Away Win).

* **Model #2: Player Scoring Predictor**
    * *Work in Progress: Aims to predict if a player will score in an upcoming match.*

* **Computer Vision**
    * *Planned: Use YOLOv8 for player detection in match footage.*

* **Dashboard**
    * *Planned: A Streamlit dashboard to interact with the models.*

## 💻 Tech Stack

* Python 3.10+
* pandas & numpy (for data manipulation)
* statsbombpy (for data collection)
* scikit-learn (for feature scaling and modeling)
* joblib (for saving models)
* tqdm (for progress bars)

## 🛠️ Installation & Setup

Follow these steps to set up your local environment.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd soccer-analytics-project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # On Windows
    .\venv\Scripts\activate

    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you haven't created a `requirements.txt` file yet, run `pip freeze > requirements.txt` after installing all the libraries).*

## 🚀 How to Run the Pipeline (Model #1)

This guide will run the complete pipeline to download data and train the Match Outcome Predictor.

**IMPORTANT:** All commands must be run from *within* the `src` folder to ensure the Python imports for `config.py` work correctly.

```bash
# First, move into the source directory
cd src

Now, run the following scripts in order:

1. Download Raw Data
This script fetches all match event data for the specified seasons from the Statsbomb API and saves them as raw JSON files in data/raw/.

This will take a long time (potentially hours) and several gigabytes of disk space.

Bash

python -m data.data_collection

2. Process Data
This script reads all the raw JSON files from data/raw/ and aggregates them into a single, clean data/processed/matches.csv file. This file will contain one row per match with high-level stats (score, xG, possession, etc.).

Bash

python -m data.data_processing

3. Engineer Features
This script takes the matches.csv file and creates rolling-average features for each team (e.g., "average xG over last 5 games"). It saves this final, model-ready dataset to data/processed/match_features.csv.

Bash

python -m features.match_features
4. Train the Model
This final script loads match_features.csv, splits the data into training and test sets, scales the features, and trains the models. It will print a classification report to your terminal and save the best-performing model to models/match_outcome_model.pkl.

Bash

python -m src.models.match_predictor


📁 Project Structure
soccer-analytics-project/
├── config.py           # Stores all file paths
├── data/
│   ├── raw/            # Raw JSON event files (ignored by Git)
│   └── processed/      # Cleaned CSV datasets
├── models/             # Saved .pkl model files
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # All Python source code
│   ├── data/           # Data collection & processing scripts
│   ├── features/       # Feature engineering scripts
│   └── models/         # Model training scripts
├── venv/
├── .gitignore
├── README.md
└── requirements.txt    # All project dependencies



📊 Data Source
All data is sourced from the Statsbomb Open Data repository.