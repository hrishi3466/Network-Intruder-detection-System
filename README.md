# Network Intrusion Detection System (NIDS)

[](https://www.google.com/search?q=https://github.com/hrishi3466/Network-Intruder-detection-System/blob/main/LICENSE) [](https://www.python.org/)
[](https://fastapi.tiangolo.com/)
[](https://streamlit.io/)

-----

### **Table of Contents**

1.  [Introduction](https://www.google.com/search?q=%231-introduction)
2.  [Architecture](https://www.google.com/search?q=%232-architecture)
3.  [Technology Stack](https://www.google.com/search?q=%233-technology-stack)
4.  [Setup & Run](https://www.google.com/search?q=%234-setup--run)
5.  [Usage](https://www.google.com/search?q=%235-usage)


-----

### 1\. Introduction

This project develops an **Intrusion Detection System (IDS)** using Machine Learning to classify network traffic as `normal` or various `attack` types. It leverages the **KDD Cup 99 dataset** and features a **Decision Tree Classifier** for detection, a **FastAPI** backend for predictions, and a **Streamlit** web interface.

### 2\. Architecture

The system consists of a training pipeline, a real-time prediction API, and a user interface.

```
Raw Data -> Train Model -> Save Artifacts -> FastAPI (Loads Artifacts) <- Streamlit UI
```



### 3\. Technology Stack

  * **Model**: **Decision Tree Classifier** (chosen for its interpretability and robust classification on tabular data).
  * **Backend API**: **FastAPI** (for high-performance, easy-to-document RESTful predictions).
  * **Frontend UI**: **Streamlit** (for quick and interactive web application development).
  * **Data Processing**: Pandas, scikit-learn (Label Encoding, Min-Max Scaling, One-Hot Encoding).
  * **Serialization**: Joblib (for saving/loading models and transformers).

### 4\. Setup & Run

Follow these steps to get the project running locally.

#### Prerequisites

  * Python 3.8+
  * pip

#### Project Structure

```
your-repo-name/
├── .git/                 # Git version control files
├── .gitignore            # Files/folders to ignore from Git tracking
├── data/              # Raw datasets (KDDTrain+.txt, KDDTest+.txt)
├── models/            # Trained model & preprocessors (ignored by Git)
├── train/
│   └── train_model.py # Model training script
├── app/
│   └── predict.py     # FastAPI prediction service
├── streamlit_app.py   # Streamlit UI
└── requirements.txt   # Python dependencies
```

#### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hrishi3466/Network-Intruder-detection-System.git
    cd Network-Intruder-detection-System
    ```
2.  **Create & activate virtual environment:**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
3.  **Install dependencies** (from `requirements.txt`):
    ```bash
    pip install -r requirements.txt
    ```

#### Run Commands

1.  **Train the model** (creates `models/` directory):
    ```bash
    python train/train_model.py
    ```
2.  **Start the Prediction API** (keep this running in a separate terminal):
    ```bash
    uvicorn app.predict:app --reload --host 0.0.0.0 --port 8000
    ```
    (API docs at `http://localhost:8000/docs`)
3.  **Launch the Streamlit UI** (in another terminal):
    ```bash
    streamlit run streamlit_app.py
    ```
    (Opens in your browser, usually `http://localhost:8501`)

### 5\. Usage

1.  Ensure both the FastAPI backend (`predict.py`) and the Streamlit frontend (`streamlit_app.py`) are running.
2.  Open your browser to the Streamlit application URL (e.g., `http://localhost:8501`).
3.  Input network traffic feature values into the form.
4.  Click "Predict Attack" to receive an instant classification of the traffic as `Normal Traffic` or `ATTACK DETECTED!` (with attack type and confidence).



<img width="1920" height="867" alt="Screenshot (77)" src="https://github.com/user-attachments/assets/b5b771cc-572e-4415-8bd4-50cc675f29b2" />

<img width="1920" height="873" alt="Screenshot (78)" src="https://github.com/user-attachments/assets/3946198a-87a6-47c8-8893-f302e5dd9381" />

-----

