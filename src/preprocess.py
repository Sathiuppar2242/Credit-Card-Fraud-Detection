import pandas as pd

TIME_FEATURES = ["transaction_hour", "transaction_day", "transaction_month", "transaction_day_of_week"]


def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    """Load training and testing datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def create_time_features(df):
    """Create useful features from transaction timestamp."""
    df = df.copy()

    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )

    df["transaction_hour"] = (
        df["trans_date_trans_time"].dt.hour
    )

    df["transaction_day"] = (
        df["trans_date_trans_time"].dt.day
    )

    df["transaction_month"] = (
        df["trans_date_trans_time"].dt.month
    )

    df["transaction_day_of_week"] = (
        df["trans_date_trans_time"].dt.dayofweek
    )

    return df


if __name__ == "__main__":
    train, test = load_data()

    train = create_time_features(train)
    test = create_time_features(test)

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    print("\nNew time-based features:")
    print(
        [
            "transaction_hour",
            "transaction_day",
            "transaction_month",
            "transaction_day_of_week",
        ]
    )
