import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import os
import numpy as np # Import numpy for explicit type checking

# --- Configuration ---
app = FastAPI(
    title="Intrusion Detection System API",
    description="API for predicting network attack types.",
    version="1.0.0"
)

# --- Define Pydantic Model for Input Data ---
class IDSInput(BaseModel):
    duration: float
    protocol_type: str
    service: str
    flag: str
    src_bytes: float
    dst_bytes: float
    land: int
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    num_compromised: int
    root_shell: int
    su_attempted: int
    num_root: int
    num_file_creations: int
    num_shells: int
    num_access_files: int
    num_outbound_cmds: int
    is_host_login: int
    is_guest_login: int
    count: int
    srv_count: int
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: int
    dst_host_srv_count: int
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float # Corrected: This field appears only once
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float

class IDSModelPredictor:
    """
    A class to load the trained IDS model and preprocessing objects,
    and handle the prediction logic.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.ohe_target = None
        self.feature_columns = [
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
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]
        self.categorical_cols = ["protocol_type", "service", "flag"]
        self.numeric_cols = [
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

        script_dir = os.path.dirname(__file__)
        PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, os.pardir))
        self.MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads the saved model, scaler, and label encoders."""
        print(f"Loading model and preprocessing objects from {self.MODELS_DIR}...")
        try:
            self.model = joblib.load(os.path.join(self.MODELS_DIR, "model.pkl"))
            print("INFO: Model loaded.")

            self.scaler = joblib.load(os.path.join(self.MODELS_DIR, "scaler.pkl"))
            print("INFO: Scaler loaded.")
            if hasattr(self.scaler, 'feature_names_in_'):
                print(f"DIAGNOSTIC: Scaler was fitted on these features ({len(self.scaler.feature_names_in_)}):")
                print(self.scaler.feature_names_in_)
            else:
                print("DIAGNOSTIC: Scaler does not have 'feature_names_in_' attribute (likely older sklearn version).")

            self.ohe_target = joblib.load(os.path.join(self.MODELS_DIR, "ohe_target.pkl"))
            print("INFO: Target OneHotEncoder loaded.")
            self.target_names = self.ohe_target.get_feature_names_out()
            print(f"DIAGNOSTIC: OHE Target names ({len(self.target_names)}): {self.target_names.tolist()}")


            for col in self.categorical_cols:
                self.label_encoders[col] = joblib.load(os.path.join(self.MODELS_DIR, f"le_{col}.pkl"))
                print(f"INFO: Label encoder for {col} loaded.")
            
            print("INFO: Model and preprocessing artifacts loaded successfully.")
        except FileNotFoundError as e:
            raise RuntimeError(f"Required model/preprocessing file not found: {e}. "
                               f"Ensure 'train/train_model.py' has been run and artifacts are in '{self.MODELS_DIR}'.")
        except Exception as e:
            raise RuntimeError(f"Error loading model artifacts: {e}")

    def preprocess_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the same preprocessing steps as during training to new input data.
        Handles unseen categories for LabelEncoder by setting them to a default value (-1).
        Ensures output DataFrame has consistent column names for the model.
        """
        processed_df = input_df.copy()

        # Apply Label Encoding to categorical columns
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            processed_df[col] = processed_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Separate numeric data for scaling
        numeric_data_to_scale = processed_df[self.numeric_cols]
        
        # DIAGNOSTIC PRINT: Check numeric data before scaling
        print(f"\nDIAGNOSTIC: numeric_data_to_scale shape: {numeric_data_to_scale.shape}")
        print(f"DIAGNOSTIC: numeric_data_to_scale columns: {numeric_data_to_scale.columns.tolist()}")
        print(f"DIAGNOSTIC: numeric_data_to_scale head:\n{numeric_data_to_scale.head(1)}")
        print(f"DIAGNOSTIC: numeric_data_to_scale has NaNs: {numeric_data_to_scale.isnull().any().any()}")


        # Apply Min-Max Scaling - scaler.transform returns a NumPy array
        scaled_numeric_array = self.scaler.transform(numeric_data_to_scale)
        
        # Convert scaled NumPy array back to DataFrame with original numeric column names
        scaled_numeric_df = pd.DataFrame(scaled_numeric_array, columns=self.numeric_cols, index=processed_df.index)
        
        # Combine processed categorical columns and scaled numeric columns
        for col in self.numeric_cols:
            processed_df[col] = scaled_numeric_df[col]

        # Finally, ensure the entire DataFrame has the correct feature_columns and order
        final_processed_df = processed_df[self.feature_columns]

        # DIAGNOSTIC PRINT: Check columns of the DataFrame right before passing to the model
        print(f"\nDIAGNOSTIC: Columns of final_processed_df ({len(final_processed_df.columns)}):")
        print(final_processed_df.columns.tolist())
        print(f"DIAGNOSTIC: final_processed_df shape: {final_processed_df.shape}")
        print(f"DIAGNOSTIC: final_processed_df head:\n{final_processed_df.head(1)}")
        print(f"DIAGNOSTIC: final_processed_df has NaNs: {final_processed_df.isnull().any().any()}")
        
        return final_processed_df

    def predict_attack(self, raw_input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives raw input data, preprocesses it, and makes a prediction.
        Returns the predicted attack class and its confidence.
        """
        # Convert dictionary input to a pandas DataFrame, ensuring correct columns and order
        input_df = pd.DataFrame([raw_input_data], columns=self.feature_columns)

        # Preprocess the input data
        processed_input = self.preprocess_input(input_df)

        # DIAGNOSTIC PRINT: Check processed_input right before model prediction
        print(f"\nDIAGNOSTIC: Input to model - type: {type(processed_input)}, shape: {processed_input.shape}")
        print(f"DIAGNOSTIC: Input to model - columns: {processed_input.columns.tolist()}")
        print(f"DIAGNOSTIC: Input to model - head:\n{processed_input.head(1)}")
        print(f"DIAGNOSTIC: Input to model - has NaNs: {processed_input.isnull().any().any()}")


        # Make prediction and get probabilities
        raw_prediction_proba = self.model.predict_proba(processed_input)
        
        # DIAGNOSTIC PRINT: Check raw_prediction_proba immediately after model call
        print(f"\nDIAGNOSTIC: raw_prediction_proba - type: {type(raw_prediction_proba)}")
        print(f"DIAGNOSTIC: raw_prediction_proba - content:\n{raw_prediction_proba}")

        # Process prediction_proba based on its structure
        if isinstance(raw_prediction_proba, list):
            # This handles cases like MultiOutputClassifier or OneVsRestClassifier where
            # predict_proba returns a list of arrays (one for each output/class).
            # Each inner array is typically [[proba_class_0, proba_class_1]].
            # We assume we need the probability of the "positive" class (index 1) for each.
            
            # Extract the probability for the "true" class (index 1) from each inner array
            # And then concatenate them to form a single 1D array of probabilities for all target classes
            combined_probabilities_1d = np.array([p[0, 1] for p in raw_prediction_proba])
            
            # Reshape to (1, num_classes) for consistency with standard predict_proba output
            prediction_proba = combined_probabilities_1d.reshape(1, -1)
            print(f"DIAGNOSTIC: Processed list of arrays into prediction_proba - shape: {prediction_proba.shape}")
        elif isinstance(raw_prediction_proba, np.ndarray):
            # This is the standard case for a single multi-class classifier's predict_proba
            prediction_proba = raw_prediction_proba
            print(f"DIAGNOSTIC: prediction_proba is standard NumPy array - shape: {prediction_proba.shape}")
        else:
            raise TypeError(f"Unexpected type for raw_prediction_proba: {type(raw_prediction_proba)}")

        # Get the predicted class index (highest probability)
        # prediction_proba should now be a (1, N_CLASSES) NumPy array
        predicted_class_idx = prediction_proba.argmax(axis=1)[0]

        # Convert the predicted index back to the original attack category name
        predicted_category = self.ohe_target.categories_[0][predicted_class_idx]

        # Get confidence for the predicted class
        # This will now correctly extract a scalar float
        confidence = prediction_proba[0, predicted_class_idx]

        return {
            "predicted_attack_type": predicted_category,
            "confidence": f"{confidence:.4f}"
        }

# Initialize the predictor globally so artifacts are loaded once at startup
try:
    predictor = IDSModelPredictor()
except RuntimeError as e:
    print(f"FATAL ERROR: {e}")
    raise

@app.post("/predict/", response_model=Dict[str, Any])
async def predict(data: IDSInput):
    """
    Receives network traffic features and predicts the attack type.
    """
    try:
        # Convert Pydantic model to a dictionary
        input_dict = data.dict()
        prediction_result = predictor.predict_attack(input_dict)
        return prediction_result
    except Exception as e:
        # Catch more general exceptions and re-raise as HTTPException for client
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")