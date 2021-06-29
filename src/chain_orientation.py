import notears.notears.utils as ut
import numpy as np
import pandas as pd


def variance_based_order(X, coinflipifundecided=True):
    v = X.var(0, keepdims=True).round(6)
    pairs = v > v.T
    # pairs where later variables have higher variance
    triu = np.triu(pairs).sum()
    # pairs where earlier variables have higher variance
    tril = np.tril(pairs).sum()
    if triu > tril:
        return '->'
    elif triu < tril:
        return '<-'
    else:
        if coinflipifundecided:
            return np.random.choice(['->', '<-'])
        return '--'


def coefficient_based_order(Xin, coinflipifundecided=True):
    X = np.copy(Xin - Xin.mean(0, keepdims=True))
    cov = np.cov(X.T)
    v = np.abs(np.diagonal(cov, 1)).ravel()
    var = np.diagonal(cov)
    # regression coefficients Xi -> Xi+1
    forward_coefs = v / var[:-1]
    # versus those for Xi+1 -> Xi
    backward_coefs = (v / var[1:])[::-1]

    # see which sequence of regression coefficients is "more increasing"
    # that's the direction of increasing raw variance
    def increasing(ain):
        a = ain.reshape(1, -1)
        comps = a >= a.T
        # triu ~ pairs (early, later) where early < later
        # tril ~ pairs (early, later) where early > later
        return np.triu(comps).sum() - np.tril(comps).sum()

    forward = increasing(forward_coefs)
    backward = increasing(backward_coefs)

    if forward > backward:
        return '->'
    elif forward < backward:
        return '<-'
    else:
        if coinflipifundecided:
            return np.random.choice(['->', '<-'])
        return '--'


if __name__ == "__main__":
    n = 1000
    total = 1000

    noises = ['gauss  ',
              'exp    ',
              'gumbel ',
              'uniform']

    log_names = ['noise_type',
                 'noise_scale',
                 'n_nodes',
                 'edge_range',
                 'n_rep',
                 'setting',
                 'result']
    log = {i: [] for i in log_names}
    for w_range in [(.5, 2.), (.5, .9), (.1, .9)]:
        for d in (3, 5, 10):
            w_ranges = ((-w_range[1], -w_range[0]), w_range)
            B_true = np.eye(d, k=1)
            res = {}
            for noise in noises:
                res[noise] = {
                    'Xraw_var_+': 0,
                    'Xraw_cof_+': 0,
                    'Xstd_var_0': 0,
                    'Xstd_cof_+': 0,
                    'Xmoj_var_0': 0,
                    'Xmoj_cof_+': 0,
                }
                for k in range(total):
                    noise_scales = np.random.uniform(.5, 2, size=d)
                    W_true = ut.simulate_parameter(B_true, w_ranges=w_ranges)

                    # Mooij rescaling
                    W_mooij = np.copy(
                        W_true /
                        np.sqrt((W_true**2).sum(0, keepdims=True) + 1))

                    X = ut.simulate_linear_sem(W_true,
                                               n,
                                               sem_type=noise.strip(),
                                               noise_scale=noise_scales)

                    X_mooij = ut.simulate_linear_sem(W_mooij,
                                                     n,
                                                     sem_type=noise.strip(),
                                                     noise_scale=noise_scales)

                    # X is -> simulated, randomly flip target in order
                    target = np.random.choice(['->', '<-'])
                    if target == '<-':
                        X = X[:, ::-1]
                        X_mooij = X_mooij[:, ::-1]

                    Xstd = np.copy(X - X.mean(0, keepdims=True))
                    Xstd /= np.std(Xstd, 0, keepdims=True)

                    if variance_based_order(X) == target:
                        res[noise]['Xraw_var_+'] += 1
                    if coefficient_based_order(X) == target:
                        res[noise]['Xraw_cof_+'] += 1
                    if variance_based_order(Xstd) == target:
                        res[noise]['Xstd_var_0'] += 1
                    if coefficient_based_order(Xstd) == target:
                        res[noise]['Xstd_cof_+'] += 1
                    if variance_based_order(X_mooij) == target:
                        res[noise]['Xmoj_var_0'] += 1
                    if coefficient_based_order(X_mooij) == target:
                        res[noise]['Xmoj_cof_+'] += 1

                for k, v in res[noise].items():
                    log['noise_type'].append(noise)
                    log['noise_scale'].append(noise_scales)
                    log['n_nodes'].append(d)
                    log['edge_range'].append(w_ranges)
                    log['n_rep'].append(total)
                    log['setting'].append(k)
                    log['result'].append(v)

    # format results
    df = pd.DataFrame(log)
    df['fraction_correct'] = df['result'] / df['n_rep']
    df['setting'].replace({
        'Xraw_var_+': 'Variance sorting on    1) raw data ',
        'Xraw_cof_+': 'Coefficient sorting on 1) raw data ',
        'Xstd_var_0': 'Variance sorting on    2) standardized data ',
        'Xstd_cof_+': 'Coefficient sorting on 2) standardized data ',
        'Xmoj_var_0': 'Variance sorting on    3) Mooij-scaled data ',
        'Xmoj_cof_+': 'Coefficient sorting on 3) Mooij-scaled data '
    }, inplace=True)
    df = df.loc[:, ['edge_range', 'setting', 'n_nodes', 'fraction_correct']]
    df = df.groupby(by=['n_nodes', 'edge_range', 'setting']).mean()
    print('Aggregated chain orientation results')
    print(df)
