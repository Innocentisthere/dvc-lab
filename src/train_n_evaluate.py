import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature


def train_random_forest(**kwargs):
    ti = kwargs['ti']
    X_train, X_val, y_train, y_val = ti.xcom_pull(task_ids='split_data')
    power_trans = ti.xcom_pull(task_ids='preprocess_data', key='power_trans')

    # Параметры для GridSearchCV
    params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Настройка MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:8081")
    mlflow.set_experiment("insurance_rf_simple")

    with mlflow.start_run():
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, params, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train.ravel())

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        y_pred_orig = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        y_val_orig = power_trans.inverse_transform(y_val)

        # Метрики
        rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        mae = mean_absolute_error(y_val_orig, y_pred_orig)
        r2 = r2_score(y_val_orig, y_pred_orig)

        # Логирование
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        # Сохранение модели
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        with open("lr_medical_charges.pkl", "wb") as file:
            joblib.dump(rf, file)

    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://",
                                                                                                   "") + '/model'  # путь до эксперимента с лучшей моделью
    print(path2model)