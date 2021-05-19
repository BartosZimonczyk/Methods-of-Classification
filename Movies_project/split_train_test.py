import pandas
import sklearn.model_selection as sml
import argparse

parser = argparse.ArgumentParser(description='Funkcja dzielaca dane na zbrior uczacy i testowy')
parser.add_argument('--file', required = True, help = 'Plik z danymi')
parser.add_argument('--size', type = float, required = True, help = 'Odsetek danych trafiajacy do zbioru uczacego')
args = parser.parse_args()

df = pandas.read_csv(args.file)
train, test = sml.train_test_split(df, test_size = args.size, stratify=df['userId'])
train.to_csv('train_ratings.csv', index=False)
test.to_csv('test_ratings.csv', index=False)