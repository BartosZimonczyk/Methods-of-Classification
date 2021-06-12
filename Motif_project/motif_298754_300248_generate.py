import argparse
import json

import numpy as np


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="param_file.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output

def generate(k, w, alpha, Theta, ThetaB):
    # we are generating random vector of {0, 1} with probability 1-alpha that we have 1
    # this vector detects if for given row we will generate data from background or from the motif
    background = np.random.uniform(size=k) > alpha
    X = np.zeros((k, w), dtype=int)
    for row in range(k):
        if background[row]:
            X[row, :] = np.random.choice(np.arange(1, 5), size=w, p=ThetaB)
        if not background[row]:
            for column in range(w):
                X[row, column] = np.random.choice(np.arange(1, 5), size=1, p=Theta[:, column])
    
    return X
    
    
param_file, output_file = ParseArguments()

with open(param_file, 'r') as inputfile:
    params = json.load(inputfile)

the_w = params['w']
the_k = params['k']
the_alpha = params['alpha']
the_theta = np.asarray(params['Theta'])
the_thetaB = np.asarray(params['ThetaB'])

the_X = generate(
        the_k,
        the_w,
        the_alpha,
        the_theta,
        the_thetaB,
    )

generated_data = {'alpha': the_alpha, 'X': the_X.tolist()}

with open(output_file, 'w') as outfile:
    json.dump(generated_data, outfile)
