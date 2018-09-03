import pickle
import pandas as pd

def converter(path):
    with open(path, 'rb') as f:
        raw = pickle.load(f, fix_imports=True, encoding='latin1')
    tickers = list(raw.keys())
    ric_tickers = [ticker for ticker in tickers if '.' in ticker]  # RICs contain a dot
    ric_tickers = list(set(ric for ric in ric_tickers if len(ric.split(".")[1]) > 0
                         and len(ric.split(".")[1]) <= 2))
    data_frame = pd.DataFrame(ric_tickers)
    data_frame.to_csv("ric_tickers.csv")

if __name__ == '__main__':
    mapping_path = 'source/abbreviation_pairs.pkl'
    converter(mapping_path)