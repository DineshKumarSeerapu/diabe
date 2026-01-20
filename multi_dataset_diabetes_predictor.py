
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    print("SHAP not available or failed to initialize. Continuing without explanations. If needed, try: pip install --upgrade shap numba llvmlite")
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")
import warnings
warnings.filterwarnings('ignore')

def main():
    # Example usage
    print("Initializing Multi-Dataset Diabetes Predictor...")
    predictor = MultiDatasetDiabetesPredictor(random_state=42)
    
    # Example with Pima Indian Diabetes Dataset (binary classification)
    print("\n--- Testing with Pima Indian Diabetes Dataset ---")
    pima_path = "pima_diabetes_dataset.csv"  # Using the correct dataset file name
    try:
        predictor.load_data(pima_path, dataset_type='pima')
        predictor.preprocess_data()
        predictor.train_models()
        metrics = predictor.evaluate_models()
        print("\nEvaluation Metrics:", metrics)
    except Exception as e:
        print(f"Error running with Pima dataset: {str(e)}")

if __name__ == "__main__":
    main()

class MultiDatasetDiabetesPredictor:
    """
    Advanced diabetes prediction system supporting multiple datasets:
    - Pima Indian Diabetes Dataset
    - Frankfurt Hospital Germany Dataset
    - Iraqi Patient Dataset for Diabetes (IPDD)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.ensemble_model = None
        self.scaler = None
        self.feature_names = None
        self.dataset_type = None
        self.target_type = 'binary'  # 'binary' or 'multiclass'
        self.X_train_processed = None
        self.shap_explainer = None
        self.feature_mappings = {}
        self.imputers = {}
        self.last_metrics = None  # Stores latest evaluation metrics

        # Define dataset specifications
        self.dataset_specs = {
            'pima': {
                'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                'target': 'Outcome',
                'type': 'binary',
                'missing_value_features': ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            },
            'frankfurt': {
                'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                'target': 'Outcome',
                'type': 'binary',
                'missing_value_features': ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            },
            'iraqi': {
                'features': ['Gender', 'Age', 'FBS', 'BUN', 'Cr', 'Chol', 'TG', 
                           'BMI', 'LDL', 'VLDL', 'HDL', 'HbA1C'],
                'target': 'Class',
                'type': 'multiclass',
                'missing_value_features': []
            }
        }

    def load_dataset(self, dataset_name, file_path=None):
        """Load specified dataset"""
        self.dataset_type = dataset_name.lower()

        if file_path:
            df = pd.read_csv(file_path)
        else:
            # Load from default files
            dataset_files = {
                'pima': 'pima_diabetes_dataset.csv',
                'frankfurt': 'frankfurt_diabetes_dataset.csv',
                'iraqi': 'iraqi_diabetes_dataset.csv'
            }
            df = pd.read_csv(dataset_files[self.dataset_type])

        spec = self.dataset_specs[self.dataset_type]
        self.target_type = spec['type']

        X = df[spec['features']]
        y = df[spec['target']]

        print(f"Loaded {dataset_name} dataset: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def load_combined_datasets(self, dataset_paths=None):
        """Load and combine multiple datasets with feature harmonization"""
        if dataset_paths is None:
            dataset_paths = {
                'pima': 'pima_diabetes_dataset.csv',
                'frankfurt': 'frankfurt_diabetes_dataset.csv',
                'iraqi': 'iraqi_diabetes_dataset.csv'
            }

        combined_X = []
        combined_y = []
        dataset_sources = []

        # Common features across Pima and Frankfurt
        common_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        for dataset_name, file_path in dataset_paths.items():
            df = pd.read_csv(file_path)
            spec = self.dataset_specs[dataset_name]

            if dataset_name in ['pima', 'frankfurt']:
                # Use common features for binary classification
                X_subset = df[common_features]
                y_subset = df[spec['target']]  # Binary: 0/1

                combined_X.append(X_subset)
                combined_y.extend(y_subset.tolist())
                dataset_sources.extend([dataset_name] * len(X_subset))

            elif dataset_name == 'iraqi':
                # Handle Iraqi dataset separately or create feature mapping
                X_iraqi = df[spec['features']]
                y_iraqi = df[spec['target']]  # Multiclass: 0/1/2

                # Convert Iraqi features to match common features where possible
                X_mapped = pd.DataFrame()
                X_mapped['Pregnancies'] = 0  # Not applicable for Iraqi dataset
                X_mapped['Glucose'] = X_iraqi['FBS'] * 18  # Convert mmol/L to mg/dL
                X_mapped['BloodPressure'] = 72  # Use average value
                X_mapped['SkinThickness'] = 20  # Use average value
                X_mapped['Insulin'] = 80  # Use average value
                X_mapped['BMI'] = X_iraqi['BMI']
                X_mapped['DiabetesPedigreeFunction'] = 0.5  # Use average value
                X_mapped['Age'] = X_iraqi['Age']

                # Convert multiclass to binary (0,1 -> 0, 2 -> 1)
                y_binary = (y_iraqi == 2).astype(int)

                combined_X.append(X_mapped)
                combined_y.extend(y_binary.tolist())
                dataset_sources.extend([dataset_name] * len(X_mapped))

        # Combine all datasets
        X_combined = pd.concat(combined_X, ignore_index=True)
        y_combined = pd.Series(combined_y)

        self.feature_names = common_features
        self.target_type = 'binary'
        self.dataset_type = 'combined'

        print(f"Combined dataset created: {X_combined.shape}")
        print(f"Combined target distribution: {y_combined.value_counts().to_dict()}")

        return X_combined, y_combined

    def preprocess_data(self, X, y, test_size=0.2):
        """Comprehensive data preprocessing pipeline"""
        # Handle missing values
        if self.dataset_type in ['pima', 'frankfurt']:
            # Replace 0s with NaN for impossible medical values
            spec = self.dataset_specs[self.dataset_type] if self.dataset_type != 'combined' else self.dataset_specs['pima']
            missing_features = spec['missing_value_features']

            for feature in missing_features:
                if feature in X.columns:
                    X[feature] = X[feature].replace(0, np.nan)

        # Impute missing values using different strategies
        self.imputers = {}
        for column in X.columns:
            if X[column].isnull().any():
                if X[column].dtype in ['int64', 'float64']:
                    # Use median for numerical features
                    self.imputers[column] = SimpleImputer(strategy='median')
                else:
                    # Use most frequent for categorical features
                    self.imputers[column] = SimpleImputer(strategy='most_frequent')

                X[column] = self.imputers[column].fit_transform(X[column].values.reshape(-1, 1)).flatten()

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y if self.target_type == 'binary' else None
        )

        return X_train, X_test, y_train, y_test

    def balance_data(self, X_train, y_train):
        """Balance dataset using advanced techniques"""
        if self.target_type == 'binary':
            try:
                # Use SMOTETomek for binary classification
                smote_tomek = SMOTETomek(random_state=self.random_state)
                X_balanced, y_balanced = smote_tomek.fit_resample(X_train, y_train)
                print(f"Data balanced using SMOTETomek: {X_balanced.shape}")
            except:
                # Fallback to SMOTE
                smote = SMOTE(random_state=self.random_state)
                X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
                print(f"Data balanced using SMOTE: {X_balanced.shape}")
        else:
            # For multiclass, use SMOTE
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            print(f"Data balanced using SMOTE: {X_balanced.shape}")

        return X_balanced, y_balanced

    def initialize_models(self):
        """Initialize ensemble models optimized for different dataset types"""
        base_models = {
            'XGBoost': xgb.XGBClassifier(
                learning_rate=0.01, n_estimators=500, max_depth=4,
                min_child_weight=6, subsample=0.8, reg_alpha=0.01,
                random_state=self.random_state, verbosity=0
            ),

            'LightGBM': lgb.LGBMClassifier(
                boosting_type='gbdt', num_leaves=15, n_estimators=300,
                learning_rate=0.05, feature_fraction=0.8,
                random_state=self.random_state, verbose=-1
            ),

            'GradientBoosting': GradientBoostingClassifier(
                learning_rate=0.05, n_estimators=300, max_depth=6,
                min_samples_split=10, subsample=0.8,
                random_state=self.random_state
            ),

            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_split=8,
                min_samples_leaf=4, random_state=self.random_state
            ),

            'AdaBoost': AdaBoostClassifier(
                n_estimators=200, learning_rate=0.05,
                random_state=self.random_state
            ),

            'LogisticRegression': LogisticRegression(
                C=1.0, random_state=self.random_state, max_iter=1000
            ),

            'SVM': SVC(
                C=1.0, kernel='rbf', probability=True,
                random_state=self.random_state
            )
        }

        # Adjust for multiclass if needed
        if self.target_type == 'multiclass':
            base_models['XGBoost'].set_params(objective='multi:softmax', num_class=3)

        self.models = base_models

    def train_individual_models(self, X_train, y_train):
        """Train individual models with cross-validation"""
        results = {}
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Initialize scalers for models that need them
        scaling_models = ['LogisticRegression', 'SVM']
        self.scaler = StandardScaler()

        for name, model in self.models.items():
            print(f"Training {name}...")

            cv_scores = []
            for train_idx, val_idx in kfold.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                if name in scaling_models:
                    scaler_fold = StandardScaler()
                    X_fold_train_scaled = scaler_fold.fit_transform(X_fold_train)
                    X_fold_val_scaled = scaler_fold.transform(X_fold_val)

                    model.fit(X_fold_train_scaled, y_fold_train)
                    pred = model.predict(X_fold_val_scaled)
                else:
                    model.fit(X_fold_train, y_fold_train)
                    pred = model.predict(X_fold_val)

                cv_scores.append(accuracy_score(y_fold_val, pred))

            # Final training on full data
            if name in scaling_models:
                X_scaled = self.scaler.fit_transform(X_train)
                model.fit(X_scaled, y_train)
            else:
                model.fit(X_train, y_train)

            results[name] = {
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'model': model
            }

            print(f"{name} CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        return results

    def create_ensemble_model(self, individual_results):
        """Create optimized ensemble model"""
        # Select top performers
        sorted_models = sorted(individual_results.items(), 
                             key=lambda x: x[1]['cv_mean'], reverse=True)

        # Use top 3-5 models based on performance
        top_models = sorted_models[:5]

        print(f"\nTop models selected for ensemble:")
        for name, results in top_models:
            print(f"{name}: {results['cv_mean']:.4f}")

        # Create weighted voting ensemble
        estimators = [(name, results['model']) for name, results in top_models]
        weights = [results['cv_mean'] for _, results in top_models]

        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )

        return self.ensemble_model

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive model evaluation"""
        # Handle scaling for specific models
        scaling_models = ['LogisticRegression', 'SVM']

        if model_name in scaling_models and self.scaler:
            X_test_eval = self.scaler.transform(X_test)
        else:
            X_test_eval = X_test

        y_pred = model.predict(X_test_eval)

        if self.target_type == 'binary':
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1_score': f1_score(y_test, y_pred, average='binary'),
                'auc_roc': roc_auc_score(y_test, y_pred_proba)
            }
        else:
            # Multiclass metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'auc_roc': None  # AUC not straightforward for multiclass
            }

        return metrics, y_pred

    def setup_shap_explainer(self, X_train, model):
        """Setup SHAP explainer for model interpretation"""
        if SHAP_AVAILABLE and model is not None:
            try:
                # Sample data for efficiency
                sample_size = min(100, len(X_train))
                X_sample = X_train.sample(sample_size, random_state=self.random_state)

                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba, X_sample
                )
                print("SHAP explainer initialized successfully")
            except Exception as e:
                print(f"SHAP explainer setup failed: {e}")
                self.shap_explainer = None
        else:
            self.shap_explainer = None

    def fit(self, dataset_name=None, file_path=None, use_combined=False):
        """Complete training pipeline"""
        print("Starting multi-dataset diabetes prediction model training...")

        # Load data
        if use_combined:
            X, y = self.load_combined_datasets()
        else:
            X, y = self.load_dataset(dataset_name, file_path)

        # Preprocess
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)

        # Balance data
        X_train_balanced, y_train_balanced = self.balance_data(X_train, y_train)

        # Convert to DataFrame if needed
        if isinstance(X_train_balanced, np.ndarray):
            self.X_train_processed = pd.DataFrame(X_train_balanced, columns=self.feature_names)
        else:
            self.X_train_processed = X_train_balanced

        y_train_processed = pd.Series(y_train_balanced)

        # Initialize and train models
        self.initialize_models()
        individual_results = self.train_individual_models(
            self.X_train_processed, y_train_processed
        )

        # Create ensemble
        ensemble = self.create_ensemble_model(individual_results)
        ensemble.fit(self.X_train_processed, y_train_processed)

        # Evaluate ensemble
        metrics, pred = self.evaluate_model(ensemble, X_test, y_test, "Ensemble")

        print(f"\nEnsemble Model Performance:")
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric.upper()}: {value:.4f}")

        # Cache metrics for API access
        self.last_metrics = metrics

        # Setup SHAP
        self.setup_shap_explainer(self.X_train_processed, ensemble)

        return self

    def predict_single(self, input_data, return_proba=True):
        """Make prediction for single instance"""
        if self.ensemble_model is None:
            raise ValueError("Model must be trained first")

        # Convert to DataFrame if needed
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data

        # Handle missing features for different datasets
        if self.dataset_type == 'combined':
            # Ensure all features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Default value

        # Apply imputation if needed
        for column in input_df.columns:
            if column in self.imputers:
                input_df[column] = self.imputers[column].transform(
                    input_df[column].values.reshape(-1, 1)
                ).flatten()

        # Make prediction
        prediction = self.ensemble_model.predict(input_df)[0]

        if return_proba:
            prediction_proba = self.ensemble_model.predict_proba(input_df)[0]
            return prediction, prediction_proba

        return prediction

    def _get_model_feature_importances(self):
        """Extract feature importances from the ensemble's component models.

        Returns a dict {feature_name: importance} normalized to sum to 1.
        If not available, returns uniform importances.
        """
        importances = None
        # Try each estimator in the ensemble for feature_importances_
        if self.ensemble_model is not None:
            collected = []
            # Fitted estimators (VotingClassifier.estimators_ is a list of estimators)
            if hasattr(self.ensemble_model, 'estimators_'):
                for est in self.ensemble_model.estimators_:
                    if hasattr(est, 'feature_importances_'):
                        collected.append(np.array(est.feature_importances_, dtype=float))
            # Original estimators as (name, est) tuples (fallback)
            if not collected and hasattr(self.ensemble_model, 'estimators'):
                for _, est in getattr(self.ensemble_model, 'estimators', []):
                    if hasattr(est, 'feature_importances_'):
                        collected.append(np.array(est.feature_importances_, dtype=float))
            if collected:
                arr = np.vstack(collected)
                mean_imp = arr.mean(axis=0)
                importances = mean_imp

        if importances is None and self.feature_names is not None:
            importances = np.ones(len(self.feature_names), dtype=float)

        if importances is None:
            return {}
        # Normalize
        s = importances.sum()
        if s > 0:
            importances = importances / s
        return {f: float(w) for f, w in zip(self.feature_names, importances)}

    def _prep_input_df(self, input_data):
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        # Ensure required features exist
        if self.feature_names is not None:
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            input_df = input_df[self.feature_names]

        return input_df

    def _explain_with_shap(self, input_df, max_features=5):
        """Attempt SHAP-based explanation, return text or raise Exception."""
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            shap_values = self.shap_explainer.shap_values(input_df)
            # Handle binary vs multiclass
            if isinstance(shap_values, list):
                if self.target_type == 'binary':
                    shap_values = shap_values[1]  # Positive class
                else:
                    pred_class = self.ensemble_model.predict(input_df)[0]
                    shap_values = shap_values[pred_class]

            # Build explanation from SHAP
            feature_effects = {}
            row_vals = np.array(shap_values[0]).ravel()
            for i, feature in enumerate(self.feature_names):
                if i < len(row_vals):
                    feature_effects[feature] = float(row_vals[i])

            sorted_features = sorted(
                feature_effects.items(), key=lambda x: abs(x[1]), reverse=True
            )
            explanation = "Key factors influencing this prediction (SHAP):\n"
            for feature, impact in sorted_features[:max_features]:
                direction = "increases" if impact > 0 else "decreases"
                explanation += f"• {feature}: {direction} risk (impact: {abs(impact):.3f})\n"
            return explanation

        raise RuntimeError("SHAP explainer not available")

    def _explain_with_lime(self, input_df, max_features=5):
        """LIME-based explanation using LimeTabularExplainer."""
        if not LIME_AVAILABLE:
            raise RuntimeError("LIME is not available")
        if self.X_train_processed is None or self.X_train_processed.empty:
            raise RuntimeError("Training data not available for LIME explainer")

        explainer = LimeTabularExplainer(
            training_data=self.X_train_processed.values,
            feature_names=self.feature_names,
            class_names=['No Diabetes', 'Diabetes'] if self.target_type == 'binary' else ['Non-Diabetic', 'Pre-Diabetic', 'Diabetic'],
            mode='classification'
        )

        # Define prediction function compatible with LIME
        def predict_fn(X):
            # Ensure DataFrame with correct columns
            X_df = pd.DataFrame(X, columns=self.feature_names)
            return self.ensemble_model.predict_proba(X_df)

        exp = explainer.explain_instance(
            data_row=input_df.iloc[0].values,
            predict_fn=predict_fn,
            num_features=min(max_features, len(self.feature_names))
        )

        weights = exp.as_list()
        explanation = "Key factors influencing this prediction (LIME):\n"
        for name, weight in weights:
            direction = "increases" if weight > 0 else "decreases"
            explanation += f"• {name}: {direction} risk (impact: {abs(weight):.3f})\n"
        return explanation

    def generate_local_explanation(self, input_data, method='auto', max_features=5):
        """Unified per-instance explanation with method selection: 'shap', 'lime', or 'auto'."""
        input_df = self._prep_input_df(input_data)

        methods = []
        if method == 'auto':
            # Prefer SHAP if available; else LIME; else heuristic
            methods = ['shap', 'lime', 'heuristic']
        else:
            methods = [method, 'heuristic']

        last_err = None
        for m in methods:
            try:
                if m == 'shap':
                    return self._explain_with_shap(input_df, max_features)
                elif m == 'lime':
                    return self._explain_with_lime(input_df, max_features)
                elif m == 'heuristic':
                    # Heuristic fallback explanation
                    if self.X_train_processed is not None and not self.X_train_processed.empty:
                        means = self.X_train_processed.mean()
                        stds = self.X_train_processed.std(ddof=0).replace(0, 1.0)
                    else:
                        means = pd.Series(0, index=self.feature_names)
                        stds = pd.Series(1.0, index=self.feature_names)
                    z = ((input_df.iloc[0] - means) / stds).abs()
                    fi = self._get_model_feature_importances()
                    contrib = {f: float(fi.get(f, 1.0) * z.get(f, 0.0)) for f in self.feature_names}
                    sorted_features = sorted(contrib.items(), key=lambda x: x[1], reverse=True)
                    explanation = "Key factors influencing this prediction (heuristic):\n"
                    for feature, score in sorted_features[:max_features]:
                        explanation += f"• {feature}: higher influence (score: {score:.3f})\n"
                    return explanation
            except Exception as e:
                last_err = e
                continue

        return f"Explanation generation failed: {str(last_err) if last_err else 'Unknown error'}"

    # Backward compatibility
    def generate_explanation(self, input_data, max_features=5):
        return self.generate_local_explanation(input_data, method='auto', max_features=max_features)
        try:
            # Training distribution from X_train_processed
            if self.X_train_processed is not None and not self.X_train_processed.empty:
                means = self.X_train_processed.mean()
                stds = self.X_train_processed.std(ddof=0).replace(0, 1.0)
            else:
                # Fallback safe defaults
                means = pd.Series(0, index=self.feature_names)
                stds = pd.Series(1.0, index=self.feature_names)

            # Standardized deviation
            z = ((input_df.iloc[0] - means) / stds).abs()

            # Feature importances
            fi = self._get_model_feature_importances()
            contrib = {}
            for f in self.feature_names:
                w = fi.get(f, 1.0)
                contrib[f] = float(w * z.get(f, 0.0))

            sorted_features = sorted(contrib.items(), key=lambda x: x[1], reverse=True)
            explanation = "Key factors influencing this prediction (heuristic):\n"
            for feature, score in sorted_features[:max_features]:
                explanation += f"• {feature}: higher influence (score: {score:.3f})\n"
            return explanation
        except Exception as e:
            return f"Explanation generation failed: {str(e)}"

    def compute_metrics(self, dataset_name: str | None = None):
        """Compute evaluation metrics using the original dataset(s).

        This uses the already-trained ensemble model and cached preprocessing artifacts
        (feature_names, imputers, scaler). It reloads the dataset from CSVs and evaluates.

        Parameters
        - dataset_name: optional override; if None, uses self.dataset_type

        Returns
        - dict of metrics (accuracy, precision, recall, f1_score, auc_roc or None)
        """
        if self.ensemble_model is None:
            raise ValueError("Model must be trained before computing metrics")

        target_dataset = (dataset_name or self.dataset_type or 'combined').lower()

        # Load data according to dataset
        if target_dataset == 'combined':
            X, y = self.load_combined_datasets()
        else:
            X, y = self.load_dataset(target_dataset)

        # Align features to training feature set
        if self.feature_names is not None:
            for f in self.feature_names:
                if f not in X.columns:
                    X[f] = 0
            X = X[self.feature_names]

        # Apply imputers if available
        for column in X.columns:
            if column in self.imputers:
                X[column] = self.imputers[column].transform(X[column].values.reshape(-1, 1)).flatten()

        # Evaluate using stored scaler rules inside evaluate_model
        metrics, _ = self.evaluate_model(self.ensemble_model, X, y, "Ensemble")
        self.last_metrics = metrics
        return metrics
