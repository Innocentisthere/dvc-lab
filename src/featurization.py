import os

import pandas as pd


def get_features(dataset):

    df = dataset.copy()
    cat_columns = df.select_dtypes(include=object)

    # OneHotEncoding
    for col in cat_columns:
        to_add = pd.get_dummies(df[col], drop_first=True, dtype=float, prefix=col)
        df = pd.concat((df, to_add), axis=1)
        df = df.drop([col], axis=1)

    # Кодирование категорий BMI и других фичей
    df['bmi_category'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 25, 30, float('inf')],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )

    to_add = pd.get_dummies(df["bmi_category"], dtype=float, prefix="bmi")
    df = pd.concat((df, to_add), axis=1)
    df = df.drop(["bmi_category", "bmi"], axis=1)

    return df


if __name__=="__main__":
    os.makedirs(os.path.join("data", "features"), exist_ok=True)
    dataset = pd.read_csv("data/insurance.csv", delimiter=",")
    features = get_features(dataset)
    features.to_csv("data/features/insurance_featurized.csv", index=False)


