import pandas as pd


TIMESTAMP_COLUMN = "trans_date_trans_time"

TIME_FEATURES = [
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_day_of_week",
]


def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    """Load training and testing datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def create_time_features(df):
    """Create useful features from transaction timestamp."""
    df = df.copy()

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN]
    )

    df["transaction_hour"] = (
        df[TIMESTAMP_COLUMN].dt.hour
    )

    df["transaction_day"] = (
        df[TIMESTAMP_COLUMN].dt.day
    )

    df["transaction_month"] = (
        df[TIMESTAMP_COLUMN].dt.month
    )

    df["transaction_day_of_week"] = (
        df[TIMESTAMP_COLUMN].dt.dayofweek
    )

    return df


if __name__ == "__main__":
    train, test = load_data()

    train = create_time_features(train)
    test = create_time_features(test)

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    print("\nNew time-based features:")
    print(TIME_FEATURES)