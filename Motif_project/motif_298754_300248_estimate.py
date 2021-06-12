import argparse
import json

import numpy as np
from tqdm import tqdm


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="generated_data.json", required=False, help='File with data  (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False, help='File to save our parameters into %(default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False, help='Estimate alpha?  (default: %(default)s)')
    parser.add_argument('--MC-runs', default=1000, required=False, help='Number of MC runs to perform (default: %(default)s)')
    args = parser.parse_args()
    
    return args.input, args.output, args.estimate_alpha, args.MC_runs

def Q_zero_one(x, alpha, w, theta, thetaB):
    # returns desired values of Q_i(0) and Q_i(1)
    p_motif = alpha * np.prod(theta[x - 1, np.arange(w)])
    p_background = (1 - alpha) * np.prod(thetaB[x - 1])
    const = p_motif + p_background

    return p_background/const, p_motif/const # Qi(0), Qi(1)

# since testing mode is not used, this function is not use either
# this legacy function is kept "just in case"
def total_variation(real_theta, real_thetaB, est_theta, est_thetaB):
    dtb = np.abs(real_thetaB - est_thetaB).sum() / 2
    dt = np.abs(real_theta - est_theta).sum() / 2

    return (dt + dtb) / (real_theta.shape[1] + 1)

the_theta = None
the_thetaB = None

def estimate_params(fX, fk, fw, falpha, alpha_known=True, eps=1e-2, verbouse=False, testing_mode=False):
    # testing mode and verbouse are legacy of testing phase
    # they are not used in standard usage of this function
    # but shall not be deleted
    
    # initiating theta for t = 0
    # fisrt thetas are randomly distributed, but with given condition that they sum up to 1
    t = 0
    new_theta = np.random.uniform(size=(4, fw))
    new_theta = (new_theta.T / np.sum(new_theta, axis=1)).T
    new_thetaB = np.random.uniform(size=4)
    new_thetaB = new_thetaB / np.sum(new_thetaB)
    if not alpha_known:
        falpha = np.random.uniform(size=1)
    
    thetas = []
    thetasB = []
    thetas.append(new_theta)
    thetasB.append(new_thetaB)
    # our stop criterion is Fob norm < eps on both thetas
    theta_diff = np.linalg.norm(thetas[-1])
    thetaB_diff = np.linalg.norm(thetasB[-1])
    
    if testing_mode:
        tvs = []
        norms = []
        normsB = []
    
    while theta_diff > eps or thetaB_diff > eps:
        # EM algorithm

        # (E)xpectation step
        Q_zero = np.zeros(fk)
        Q_one = np.zeros(fk)
        for i in range(fk):
            Q_zero[i], Q_one[i] = Q_zero_one(fX[i, :], falpha, fw, thetas[-1], thetasB[-1])

        # prep for new iteration
        new_theta = np.zeros((4, fw))
        new_thetaB = np.zeros(4)


        # (M)aximization step
        # here we will write estimators that maximize big Q function
        
        # estimating thetaB
        for j in range(4):
            current_sum = 0
            for i in range(fk):
                current_sum += Q_zero[i] * (fX[i, :] == j+1).sum()
            current_sum /= fw * Q_zero.sum()
            new_thetaB[j] = current_sum
        
        # estimating theta
        for j in range(4):
            for l in range(fw):
                current_sum = 0
                for i in range(fk):
                    current_sum += Q_one[i] * (fX[i, l] == j+1)
                current_sum /= Q_one.sum()
                new_theta[j, l] = current_sum

        # part where we are estimating alpha
        # Q_i(1) is the probability that our data row is from motif
        # so to find this probability, we can just estimate it by summing up all of the Q(1) and dividing by sum of all Qs
        if not alpha_known:
            falpha = Q_one.sum()/(Q_one.sum() + Q_zero.sum())

        # prep for new iteration
        thetas.append(new_theta)
        thetasB.append(new_thetaB)
        theta_diff = np.linalg.norm(thetas[-1] - thetas[-2])
        thetaB_diff = np.linalg.norm(thetasB[-1] - thetasB[-2])

        if testing_mode:
            tvs.append(total_variation(the_theta, the_thetaB, thetas[-1], thetasB[-1]))
            norms.append(theta_diff)
            normsB.append(thetaB_diff)

        if verbouse:
            print(f'Iteration: {t+1}\tCurrent theta difference {theta_diff}\tCurrent thetaB difference {thetaB_diff}')
            print(thetas[-1])
            print(thetasB[-1])
            print(falpha)
        t += 1
    
    if testing_mode:
        return thetas[-1], thetasB[-1], falpha, tvs, norms[1:], normsB[1:]
    
    return thetas[-1], thetasB[-1], falpha
    
    
input_file, output_file, estimate_alpha, mc_runs = ParseArguments()
mc_runs = int(mc_runs)

alpha_known = None
if estimate_alpha == 'yes':
    alpha_known = True
elif estimate_alpha == 'no':
    alpha_known = False
else:
    raise TypeError('You should pass "yes" or "no" as estimate_alpha argument.')

with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)

the_alpha = data['alpha']
the_X = np.asarray(data['X'])
the_k, the_w = the_X.shape

# doing 1000 MC runs in order to find the best starting thetas
estimated_theta = np.zeros((mc_runs, 4, the_w))
estimated_thetaB = np.zeros((mc_runs, 4))
alphas = np.zeros(mc_runs)
for i in tqdm(range(mc_runs), desc='MC runs'):
    # X = generate(the_k, the_w, the_alpha, the_theta, the_thetaB)
    estimated_theta[i, :, :], estimated_thetaB[i, :], alphas[i] = estimate_params(the_X, the_k, the_w, the_alpha, alpha_known=alpha_known)

result_et = estimated_theta.mean(axis=0)
result_etB = estimated_thetaB.mean(axis=0)
result_alpha = alphas.mean()

estimated_params = {
    "alpha" : result_alpha,            # "przepisujemy" to alpha, one nie bylo estymowane 
    "Theta" : result_et.tolist(),   # westymowane
    "ThetaB" : result_etB.tolist()  # westymowane
    }

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)
