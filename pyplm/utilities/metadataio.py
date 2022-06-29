import pandas as pd
import ast

def get_metadata_df(group, dataset_name):
    metadata_dataset = dataset_name + '_metadata'
    metadata = group[metadata_dataset].asstr()[()]
    metadata = [ast.literal_eval(md) for md in metadata]
    df = pd.DataFrame(metadata)
    return df