import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_and_save_model():
    """
    Loads and cleans the KDD Cup 99 dataset, preprocesses features,
    trains a Decision Tree model, and saves the trained model
    along with preprocessing objects (scaler, label encoders, one-hot encoder).
    """
    # --- Configuration ---
    # Get the directory of the current script (train/train_model.py)
    script_dir = os.path.dirname(__file__)

    # Construct paths relative to the script's directory, then move up to project root, then into data
    # os.pardir is equivalent to '..'
    PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, os.pardir)) # Go up one level from 'train' to project root
    DATA_DIR = os.path.join(PROJECT_ROOT, "data") # Path to the data folder

    TRAIN_DATA_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt") # Full path to KDDTrain+.txt
    TEST_DATA_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")   # Full path to KDDTest+.txt
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")         # Path to the models folder from project root
    
    # Define the 43 column names for the KDD dataset
    # This is crucial for correctly loading the dataset without headers
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files",
        "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty_level"   # Target and auxiliary column
    ]

    categorical_cols = ["protocol_type", "service", "flag"]
    
    # CORRECTED: This list MUST include ALL 37 numeric features used by the model
    numeric_cols = [
        "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate"
    ]

    # --- 1. Load Data ---
    print(f"Loading data from {TRAIN_DATA_PATH} and {TEST_DATA_PATH}...")
    try:
        df_train = pd.read_csv(TRAIN_DATA_PATH, header=None, names=columns)
        df_test = pd.read_csv(TEST_DATA_PATH, header=None, names=columns)
    except FileNotFoundError:
        print(f"Error: Dataset files not found. Please ensure '{TRAIN_DATA_PATH}' and '{TEST_DATA_PATH}' exist.")
        return

    print(f"Train data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")

    # --- 2. Data Cleaning & Feature Engineering ---
    # Map attack labels to broader categories (as per KDD documentation or common practice)
    # This step is crucial for consistent target variable across different KDD versions
    attack_mapping = {
        'normal': 'normal',
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos',
        'apache2': 'dos', 'mailbomb': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
        'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
        'mscan': 'probe', 'saint': 'probe',
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
        'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
        'named': 'r2l', 'sendmail': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l',
        'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
        'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
    }

    df_train['attack_category'] = df_train['label'].replace(attack_mapping)
    df_test['attack_category'] = df_test['label'].replace(attack_mapping)

    # Drop the original 'label' and 'difficulty_level' columns
    df_train = df_train.drop(columns=['label', 'difficulty_level'])
    df_test = df_test.drop(columns=['label', 'difficulty_level'])

    # --- 3. Preprocessing: Encoding Categorical Features ---
    print("Encoding categorical features...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col]) # Use fitted encoder for test data
        label_encoders[col] = le # Store each encoder separately

    # --- 4. Preprocessing: Scaling Numeric Features ---
    print("Scaling numeric features...")
    scaler = MinMaxScaler()
    df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
    df_test[numeric_cols] = scaler.transform(df_test[numeric_cols]) # Use fitted scaler for test data

    # --- 5. Prepare Features (X) and Target (y) ---
    # The remaining columns are features except 'attack_category'
    features = [col for col in df_train.columns if col != 'attack_category']
    X_train = df_train[features]
    X_test = df_test[features]

    # One-hot encode the target variable 'attack_category'
    print("One-hot encoding target variable...")
    ohe_target = OneHotEncoder(sparse_output=False)
    y_train_encoded = ohe_target.fit_transform(df_train[['attack_category']])
    y_test_encoded = ohe_target.transform(df_test[['attack_category']])

    y_train = pd.DataFrame(y_train_encoded, columns=ohe_target.get_feature_names_out(['attack_category']))
    y_test = pd.DataFrame(y_test_encoded, columns=ohe_target.get_feature_names_out(['attack_category']))

    print(f"X_train shape: {X_train.shape} (Features used: {X_train.shape[1]})")
    print(f"y_train shape: {y_train.shape}")


    # --- 6. Model Training (Decision Tree) ---
    print("Training Decision Tree model...")
    dt_model = DecisionTreeClassifier(random_state=42)

    # Hyperparameter tuning with GridSearchCV (as in notebook)
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_leaf': [1, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_dt_model = grid_search.best_estimator_
    print(f"Best Decision Tree parameters: {grid_search.best_params_}")
    print(f"Best Decision Tree accuracy (CV): {grid_search.best_score_:.4f}")

    # --- 7. Save Model and Preprocessing Objects ---
    print("Saving model and preprocessing objects...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(best_dt_model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(ohe_target, os.path.join(MODELS_DIR, "ohe_target.pkl")) # Save the target OneHotEncoder

    for col, le in label_encoders.items():
        joblib.dump(le, os.path.join(MODELS_DIR, f"le_{col}.pkl"))

    print("✅ Training complete. Model and preprocessing objects saved successfully!")

if __name__ == "__main__":
    train_and_save_model()