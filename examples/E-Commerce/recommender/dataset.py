import pandas as pd
import tensorflow as tf

def create_tf_dataset(path: str) -> tf.raw_ops.TensorSliceDataset:
    """
    Crea el dataset de "Online-Retail.csv" que se encuentra en path.
    
    References
    ----------
    - https://www.tensorflow.org/tutorials/load_data/csv
    """
    data = pd.read_csv(
        path,
        dtype={'CustomerID': str, 'StockCode': str},
        parse_dates=['InvoiceDate'],
    )
    data = data.dropna()
    # Actual items have a "StockCode" of length = 5
    data = data[data['StockCode'].apply(len) == 5]
    data['timestamp'] = (data['InvoiceDate'] - pd.Timestamp('1970-01-01')) / pd.Timedelta('1s')

    # Just to test
    data = data.sample(n=20_000, random_state=42)

    # To remove clients with few interactions
    customers = data['CustomerID'].value_counts()
    customers = {k: v for (k, v) in customers.items() if v > 2}
    data = data[data['CustomerID'].isin(customers)]

    features_dict = {
        'user_id': data['CustomerID'].values,
        'item_id': data['StockCode'].values,
        'item_description': data['Description'].values,
        'timestamp': data['timestamp'].values,
        'country': data['Country'].values
    }

    dataset = tf.data.Dataset.from_tensor_slices(features_dict)

    return dataset
