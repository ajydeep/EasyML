import logging
import pandas as pd
import numpy as np
from scipy import stats
import re
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.svm import SVC, SVR
import optuna
import json
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, 
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

class AutoMLPipeline:
    def __init__(self):
        # Set up logging
        self._setup_logging()
        
        # Initialize model dictionaries
        self._initialize_models()
        
        # Initialize state variables
        self.best_model = None
        self.best_score = float('-inf')
        self.problem_type = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.selected_features = None
        
        logging.info("AutoMLPipeline initialized successfully")

    def _setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('automl.log')
            ]
        )

    def _initialize_models(self):
        """Initialize classification and regression models"""
        self.classifiers = {
            'random_forest': RandomForestClassifier,
            'k_neighbors': KNeighborsClassifier,
            'decision_tree': DecisionTreeClassifier,
            'naive_bayes': GaussianNB,
            'logistic_regression': LogisticRegression,
            'svm': SVC
        }

        self.regressors = {
            'k_neighbors': KNeighborsRegressor,
            'linear_regression': LinearRegression,
            'random_forest': RandomForestRegressor,
            'decision_tree': DecisionTreeRegressor,
            'lasso': Lasso,
            'ridge': Ridge,
            'svr': SVR
        }


        # Define hyperparameter search spaces
        self._initialize_hyperparameter_spaces()

    def _initialize_hyperparameter_spaces(self):
        """Initialize hyperparameter search spaces for models"""
        self.hyperparameter_spaces = {
            'random_forest': {
                'classification': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'regression': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'k_neighbors': {
                'classification': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                'regression': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                }
            },
            'decision_tree': {
                'classification': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'regression': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'naive_bayes': {
                'classification': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                }
            }
        }

    def set_problem_type(self, problem_type: str) -> None:
        """Set the problem type for the pipeline"""
        if problem_type not in ['classification', 'regression']:
            raise ValueError("Problem type must be either 'classification' or 'regression'")
        self.problem_type = problem_type
        logging.info(f"Problem type set to: {problem_type}")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from various file formats with enhanced error handling"""
        try:
            file_extension = file_path.split('.')[-1].lower()
            logging.info(f"Loading data from file: {file_path}")

            if file_extension == 'csv':
                data = pd.read_csv(file_path)
            elif file_extension in ['json', 'jsonl']:
                data = pd.read_json(file_path)
            elif file_extension in ['xlsx', 'xls']:
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Log data quality metrics
            logging.info(f"Loaded data shape: {data.shape}")
            logging.info(f"Missing values: {data.isnull().sum().sum()}")
            logging.info(f"Duplicate rows: {data.duplicated().sum()}")

            return data

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def clean_data(self, data: pd.DataFrame, datetime_columns: list = None) -> pd.DataFrame:
        """Clean and preprocess the input data"""
        try:
            logging.info("Starting data cleaning process")
            df = data.copy()

            # Remove empty rows and columns
            initial_shape = df.shape
            df = df.dropna(how='all', axis=0)
            df = df.dropna(how='all', axis=1)
            logging.info(f"Removed empty rows/columns. Shape changed from {initial_shape} to {df.shape}")

            # Handle numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_columns:
                df[col] = self._clean_numeric_column(df[col])

            # Handle categorical columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                df[col] = df[col].str.replace(r'[$,\s]', '', regex=True)
                numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                if numeric_count / len(df) > 0.5:  # If more than 50% are numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = self._clean_numeric_column(df[col])
                else:
                    df[col] = self._clean_categorical_column(df[col])

            # Handle datetime columns if specified
            if datetime_columns:
                df = self._process_datetime_columns(df, datetime_columns)

            # Handle missing values
            df = self._handle_missing_values(df)

            logging.info("Data cleaning completed successfully")
            return df

        except Exception as e:
            logging.error(f"Error in data cleaning: {str(e)}")
            raise

    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean numeric column by removing currency symbols and handling percentages"""
        try:
            if series.dtype in ['int64', 'float64']:
                # Remove currency symbols and commas
                if series.astype(str).str.contains(r'[$,]').any():
                    series = series.astype(str).str.replace('$', '').str.replace(',', '')
                    series = pd.to_numeric(series, errors='coerce')

                # Handle percentages
                if series.astype(str).str.contains('%').any():
                    series = series.astype(str).str.rstrip('%').astype(float) / 100

            return series
        except Exception as e:
            logging.error(f"Error cleaning numeric column: {str(e)}")
            return series

    def _clean_categorical_column(self, series: pd.Series) -> pd.Series:
        """Clean categorical column by standardizing strings and handling special cases"""
        try:
            if series.dtype == 'object':
                # Basic string cleaning
                series = series.astype(str).str.strip().str.lower()
                
                # Remove special characters
                series = series.str.replace(r'[^\w\s]', '', regex=True)
                
                # Handle special cases (emails, phone numbers, etc.)
                if series.str.contains('@').any():
                    series = series.apply(self._validate_email)
                elif series.str.contains(r'\d{3,}').any():
                    series = series.apply(self._standardize_phone)

            return series
        except Exception as e:
            logging.error(f"Error cleaning categorical column: {str(e)}")
            return series

    @staticmethod
    def _validate_email(email: str) -> str:
        """Validate email addresses"""
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return email if isinstance(email, str) and re.match(pattern, email) else np.nan

    @staticmethod
    def _standardize_phone(phone: str) -> str:
        """Standardize phone number format"""
        if not isinstance(phone, str):
            return phone
        nums = re.sub(r'\D', '', phone)
        return nums[-10:] if len(nums) >= 10 else nums

    def _process_datetime_columns(self, df: pd.DataFrame, datetime_columns: list) -> pd.DataFrame:
        """Process datetime columns and create relevant features"""
        for col in datetime_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                logging.info(f"Processed datetime column: {col}")
            except Exception as e:
                logging.error(f"Error processing datetime column {col}: {str(e)}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate imputation methods"""
        try:
            # For numeric columns, use KNN imputation
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_columns) > 0:
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

            # For categorical columns, use mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

            logging.info("Missing values handled successfully")
            return df
        except Exception as e:
            logging.error(f"Error handling missing values: {str(e)}")
            raise

    def preprocess_data(self, data: pd.DataFrame, target_column: str) -> tuple:
        """Preprocess data for model training"""
        try:
            logging.info("Starting data preprocessing")
            
            # Validate inputs
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            if self.problem_type is None:
                raise ValueError("Problem type must be set before preprocessing")

            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Handle categorical features
            X = self._encode_categorical_features(X)
            
            # Scale features
            X = self._scale_features(X)
            
            # Perform feature selection
            X = self.feature_selection(X, y)
            
            # Encode target for classification
            if self.problem_type == 'classification':
                self.label_encoders['target'] = LabelEncoder()
                y = self.label_encoders['target'].fit_transform(y)
            
            logging.info(f"Preprocessing completed. Feature shape: {X.shape}")
            return X, y
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise

    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder"""
        categorical_features = X.select_dtypes(include=['object']).columns
        for column in categorical_features:
            try:
                self.label_encoders[column] = LabelEncoder()
                X[column] = self.label_encoders[column].fit_transform(X[column])
                logging.info(f"Encoded categorical feature: {column}")
            except Exception as e:
                logging.error(f"Error encoding feature {column}: {str(e)}")
        return X

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler"""
        try:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns
            )
            logging.info("Feature scaling completed")
            return X_scaled
        except Exception as e:
            logging.error(f"Error in feature scaling: {str(e)}")
            raise

    def feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Perform feature selection using multiple methods"""
        try:
            logging.info("Starting feature selection")
            feature_importance = pd.DataFrame(index=X.columns)
            
            # Method 1: Mutual Information
            mi_scores = (mutual_info_classif(X, y) if self.problem_type == 'classification' 
                        else mutual_info_regression(X, y))
            feature_importance['mutual_info'] = mi_scores
            
            # Method 2: Model-based selection
            model = (RandomForestClassifier(n_estimators=100, random_state=42) if self.problem_type == 'classification'
                    else RandomForestRegressor(n_estimators=100, random_state=42))
            model.fit(X, y)
            feature_importance['model_importance'] = model.feature_importances_
            
            # Method 3: Correlation with target
            correlations = X.corrwith(pd.Series(y)).abs()
            feature_importance['correlation'] = correlations
            
            # Calculate mean importance
            feature_importance['mean_importance'] = feature_importance.mean(axis=1)
            
            # Select top features
            n_features = min(10, len(X.columns))
            self.selected_features = feature_importance.nlargest(n_features, 'mean_importance').index
            
            logging.info(f"Selected {len(self.selected_features)} features")
            return X[self.selected_features]
            
        except Exception as e:
            logging.error(f"Error in feature selection: {str(e)}")
            raise

    def optimize_hyperparameters(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> dict:
        """Optimize model hyperparameters using Optuna"""
        try:
            logging.info(f"Starting hyperparameter optimization for {model_name}")
            
            def objective(trial):
                param_space = self.hyperparameter_spaces.get(model_name, {}).get(self.problem_type, {})
                logging.info(f"Model: {model_name}, Parameter space: {param_space}")
                
                if not param_space:
                    logging.warning(f"No hyperparameter space defined for {model_name} with {self.problem_type}")
                    return self._evaluate_default_model(model_name, X, y)
                
                params = {}
                for param_name, param_values in param_space.items():
                    if isinstance(param_values, list):
                        if all(isinstance(x, (int, float)) for x in param_values):
                            if all(isinstance(x, int) for x in param_values):
                                params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                            else:
                                params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                        else:
                            params[param_name] = trial.suggest_categorical(param_name, param_values)
                    elif isinstance(param_values, tuple):
                        if len(param_values) == 2:
                            params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
                        elif len(param_values) == 3:
                            params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1], step=param_values[2])
                
                ModelClass = (self.classifiers.get(model_name) if self.problem_type == 'classification' 
                            else self.regressors.get(model_name))
                
                if ModelClass is None:
                    raise ValueError(f"Model '{model_name}' not found for {self.problem_type}")
                
                # Handle None values for max_depth
                if 'max_depth' in params and params['max_depth'] == -1:
                    params['max_depth'] = None
                    
                model = ModelClass(**params)
                score = cross_val_score(
                    model, X, y, 
                    cv=5, 
                    scoring='accuracy' if self.problem_type == 'classification' else 'r2'
                ).mean()
                
                return score

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=10)
            
            logging.info(f"Best parameters found: {study.best_params}")
            return study.best_params
            
        except Exception as e:
            logging.error(f"Error in hyperparameter optimization: {str(e)}")
            raise

    def _evaluate_default_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate model with default parameters"""
        ModelClass = (self.classifiers[model_name] if self.problem_type == 'classification' 
                else self.regressors[model_name])
        model = ModelClass()  # Create instance with default parameters
        return cross_val_score(
            model, X, y, 
            cv=5, 
            scoring='accuracy' if self.problem_type == 'classification' else 'r2'
        ).mean()

    def _get_trial_params(self, trial, param_space: dict) -> dict:
        """Get parameters for Optuna trial"""
        params = {}
        for param_name, param_range in param_space.items():
            if isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            elif isinstance(param_range, tuple):
                if len(param_range) == 2:
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                elif len(param_range) == 3:
                    params[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1], step=param_range[2]
                    )
        return params

    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate model performance with multiple metrics"""
        try:
            metrics = {}
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X, y, 
                cv=5, 
                scoring='accuracy' if self.problem_type == 'classification' else 'r2'
            )
            metrics['cv_score_mean'] = cv_scores.mean()
            metrics['cv_score_std'] = cv_scores.std()
            
            # Train-test split evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics based on problem type
            if self.problem_type == 'classification':
                metrics.update(self._get_classification_metrics(model, X_test, y_test, y_pred))
            else:
                metrics.update(self._get_regression_metrics(y_test, y_pred))
            
            logging.info(f"Model evaluation completed: {metrics['cv_score_mean']:.4f}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise

    def _get_classification_metrics(self, model, X_test, y_test, y_pred) -> dict:
        """Get classification-specific metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
        
        return metrics

    def _get_regression_metrics(self, y_test, y_pred) -> dict:
        """Get regression-specific metrics"""
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train and evaluate all models"""
        try:
            logging.info("Starting model training and evaluation")
            model_classes = self.classifiers if self.problem_type == 'classification' else self.regressors
            
            for name, ModelClass in model_classes.items():
                logging.info(f"Training {name}")
                
                # Initialize model with default parameters first
                model = ModelClass()
                
                # Optimize hyperparameters if available
                if name in self.hyperparameter_spaces:
                    best_params = self.optimize_hyperparameters(name, X, y)
                    model = ModelClass(**best_params)  # Create new instance with optimized parameters
                
                # Evaluate model
                metrics = self.evaluate_model(model, X, y)
                
                # Update best model if necessary
                if metrics['cv_score_mean'] > self.best_score:
                    self.best_score = metrics['cv_score_mean']
                    self.best_model = model
                    self.best_metrics = metrics
                    logging.info(f"New best model found: {name}")
                
            logging.info("Model training and evaluation completed")
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise

    def save_model(self, output_path: str) -> None:
        """Save model and related metadata"""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Prepare model information
            model_info = {
                'problem_type': self.problem_type,
                'best_score': self.best_score,
                'best_model_name': type(self.best_model).__name__,
                'selected_features': list(self.selected_features),
                'metrics': self.best_metrics,
                'feature_importance': (
                    self.best_model.feature_importances_.tolist()
                    if hasattr(self.best_model, 'feature_importances_')
                    else None
                )
            }
            
            # Save model information
            with open(f"{output_path}/model_info.json", "w") as f:
                json.dump(model_info, f, indent=4)
                
            logging.info(f"Model saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    def run_pipeline(self, file_path: str, target_column: str, output_path: str) -> object:
        """Run the complete AutoML pipeline"""
        try:
            logging.info("Starting AutoML pipeline")
            
            # Load data
            data = self.load_data(file_path)
            
            # Clean data
            clean_data = self.clean_data(data)
            
            # Preprocess data
            X, y = self.preprocess_data(clean_data, target_column)
            
            # Train and evaluate models
            self.train_and_evaluate(X, y)
            
            # Save model
            self.save_model(output_path)
            
            logging.info("Pipeline completed successfully")
            return self.best_model
            
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model"""
        try:
            if self.best_model is None:
                raise ValueError("No model has been trained yet")
            
            # Preprocess input data
            X = X[self.selected_features]
            X = self._encode_categorical_features(X)
            X = self._scale_features(X)
            
            # Make predictions
            predictions = self.best_model.predict(X)
            
            # Decode predictions for classification
            if self.problem_type == 'classification' and 'target' in self.label_encoders:
                predictions = self.label_encoders['target'].inverse_transform(predictions)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise

'''pipeline=AutoMLPipeline()
pipeline.set_problem_type('regression')
file_path = 'business.csv'
best_model = pipeline.run_pipeline(
                            file_path=file_path,
                            target_column='price',
                            output_path='sample_data'
                        )'''
