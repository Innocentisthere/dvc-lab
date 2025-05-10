import os
import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def scale_split_data(df):
    params = yaml.safe_load(open("params.yaml"))["scale_split"]

    scaler = StandardScaler()

    x = df.drop(columns=['charges'])
    y = df['charges'].values.reshape(-1, 1)

    x_scaled = scaler.fit_transform(x)


    x_train, x_val, y_train, y_val = train_test_split(
        x_scaled,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"]
    )

    train_columns = list(x.columns) + ['charges']
    df_train = pd.DataFrame(
        np.hstack([x_train, y_train]),
        columns=train_columns
    )

    df_test = pd.DataFrame(
        np.hstack([x_val, y_val]),
        columns=train_columns
    )

    return df_train, df_test


if __name__=="__main__":
    os.makedirs(os.path.join("data", "split"), exist_ok=True)
    dataset = pd.read_csv("data/features/insurance_featurized.csv", delimiter=",")

    train, test = scale_split_data(dataset)

    train.to_csv("data/split/train.csv", index=False)
    test.to_csv("data/split/test.csv", index=False)
