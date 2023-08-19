import pandas as pd

# convert a pickel file to csv
def convert_pickle_to_csv(pickle_file, csv_file):
    df = pd.read_pickle(pickle_file)
    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    convert_pickle_to_csv('../dataset/df.pkl', '../dataset/df.csv') 