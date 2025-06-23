import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import optuna
import joblib
from custom_transformers import CorrelationFeatureSelector

# Load data
train = pd.read_csv('data/application_train.csv', encoding = 'ISO-8859-1')
agg_bureau = pd.read_csv('data/agg_bureau_selected.csv')
agg_pos = pd.read_csv('data/agg_pos_selected.csv')
agg_prev = pd.read_csv('data/agg_prev_selected.csv')
agg_credit = pd.read_csv('data/agg_credit_selected.csv')

train_merged = (train
    .merge(agg_bureau, on='SK_ID_CURR', how='left')
    .merge(agg_pos, on='SK_ID_CURR', how='left')
    .merge(agg_prev, on='SK_ID_CURR', how='left')
    .merge(agg_credit, on='SK_ID_CURR', how='left')
)

X = train_merged.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = train_merged['TARGET']

# Identify feature types
numerical_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

# Define pipeline building function (from your notebook)
def build_pipeline(numerical_features, categorical_features, params=None):
    if params is None:
        params = {}

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('feature_selector', CorrelationFeatureSelector(threshold=0.9))
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(
            is_unbalance=True,
            metric='auc',
            objective='binary',
            **params
        ))
    ])

    return model_pipeline

# Optuna objective function (from your code)
def objective(trial, X, y, numerical_features, categorical_features):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42
    }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline(numerical_features, categorical_features, params)
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba)

    return auc_score

# Optimize model (from your code)
def optimize_model(X, y, numerical_features, categorical_features, n_trials=50):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    objective_func = lambda trial: objective(trial, X, y, numerical_features, categorical_features)
    study.optimize(objective_func, n_trials=n_trials)

    print('Best trial:')
    trial = study.best_trial
    print(f'  AUC Score: {trial.value}')
    print('  Best hyperparameters:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    best_params = trial.params
    best_params['random_state'] = 42
    best_pipeline = build_pipeline(numerical_features, categorical_features, best_params)
    return best_pipeline

# Train and save the model
lgbm_optuna = optimize_model(X, y, numerical_features, categorical_features, n_trials=50)
lgbm_optuna.fit(X, y)
joblib.dump(lgbm_optuna, 'lgbm_optuna.joblib')
print("Optimized model saved as 'lgbm_optuna.joblib'")