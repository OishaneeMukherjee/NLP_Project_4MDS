import pandas as pd

def load_resume_data(path):
    df = pd.read_csv(path)
    df.dropna(subset=['Resume_str', 'Category'], inplace=True)
    return df
