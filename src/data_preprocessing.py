def preprocess_data(df):
    """
    Function to preprocess data by normalizing it.

    :param df: pandas DataFrame with raw data.
    :return: pandas DataFrame with processed (normalized) data.
    """
    processed_df = df.copy()

    # Normalize the data
    # For each column, subtract the minimum and divide by the range.
    for column in processed_df.columns:
        min_value = processed_df[column].min()
        max_value = processed_df[column].max()
        processed_df[column] = (processed_df[column] - min_value) / (max_value - min_value)

    return processed_df
