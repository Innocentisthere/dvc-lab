import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature
import yaml


def train_random_forest():
    train_df = pd.read_csv("data/split/train.csv", delimiter=",")
    x_train = train_df.drop(columns=['charges'])
    y_train = train_df['charges']

    test_df = pd.read_csv("data/split/test.csv", delimiter=",")
    x_test = test_df.drop(columns=['charges'])
    y_test = test_df['charges']

    params = yaml.safe_load(open("params.yaml"))["train"]
    
    # Параметры для GridSearchCV
    grid_params = {
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "min_samples_split": params["min_samples_split"],
        "min_samples_leaf": params["min_samples_leaf"],
        "ccp_alpha": params["ccp_alpha"]
    }

    mlflow.set_experiment("insurance_random_forest")

    with mlflow.start_run():
        rf = RandomForestRegressor(random_state=params["random_state"])
        grid_search = GridSearchCV(rf, grid_params, cv=params["cv"], n_jobs=-1)
        grid_search.fit(x_train, y_train.ravel())

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)

        # Метрики
        rmse = np.sqrt(mean_squared_error(y_test, y_pred.reshape(-1, 1)))
        mae = mean_absolute_error(y_test, y_pred.reshape(-1, 1))
        r2 = r2_score(y_test, y_pred.reshape(-1, 1))

        # Логирование
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        # Сохранение модели
        signature = infer_signature(x_train, best_model.predict(x_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        with open("rand_forest_insurance.pkl", "wb") as file:
            joblib.dump(rf, file)


if __name__ == "__main__":
    train_random_forest()
