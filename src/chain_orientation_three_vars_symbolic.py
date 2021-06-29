import numpy as np
from sympy import simplify, sqrt, symbols
from sympy.stats import Normal, covariance as cov, variance as var


def regcoeffs(x, y, z):
    covxy = cov(x, y)
    covyz = cov(y, z)
    varx = var(x)
    vary = var(y)
    varz = var(z)
    # forward
    f1 = simplify(covxy / varx)
    f2 = simplify(covyz / vary)
    # backward
    b1 = simplify(covyz / varz)
    b2 = simplify(covxy / vary)
    return f1, f2, b1, b2


if __name__ == "__main__":
    ab, bc, a, b, c = symbols([
        "beta_{A_to_B}",
        "beta_{B_to_C}",
        "sigma_A",
        "sigma_B",
        "sigma_C"])

    Na = Normal('Na', 0, 1)
    Nb = Normal('Nb', 0, 1)
    Nc = Normal('Nc', 0, 1)

    # SEM
    # A -> B -> C

    # raw
    A = a * Na
    B = ab * A + b * Nb
    C = bc * B + c * Nc

    # standardized
    As = A / sqrt(var(A))
    Bs = B / sqrt(var(B))
    Cs = C / sqrt(var(C))

    # scale-harmonized
    Am = a * Na
    Bm = (ab / (ab**2 + 1)**(1/2)) * Am + b * Nb
    Cm = (bc / (bc**2 + 1)**(1/2)) * Bm + c * Nc

    # forward/backward coefficients in raw setting
    f1, f2, b1, b2 = regcoeffs(A, B, C)

    # forward/backward coefficients in standardized setting
    f1s, f2s, b1s, b2s = regcoeffs(As, Bs, Cs)

    # forward/backward coefficients in scale-harmonized setting
    f1m, f2m, b1m, b2m = regcoeffs(Am, Bm, Cm)

    for weight_range in [(0.5, 2),
                         (0.5, .9),
                         (.1, .9)]:
        raw = {
            'f1<f2,b1>b2': 0,
            'f1>f2,b1<b2': 0,
            'other': 0
        }
        std = {
            'f1<f2,b1>b2': 0,
            'f1>f2,b1<b2': 0,
            'other': 0
        }
        moj = {
            'f1<f2,b1>b2': 0,
            'f1>f2,b1<b2': 0,
            'other': 0
        }
        for _ in range(100000):
            # draw model parameters
            a_to_b, b_to_c = np.random.uniform(*weight_range, size=2)
            sA, sB, sC = np.random.uniform(0.5, 2, size=3)
            a_to_b *= np.random.choice([-1, 1])
            b_to_c *= np.random.choice([-1, 1])

            subs = {
                ab: a_to_b,
                bc: b_to_c,
                a: sA,
                b: sB,
                c: sC,
            }

            # raw
            if (abs(f1.subs(subs)) < abs(f2.subs(subs))
                    and abs(b1.subs(subs)) > abs(b2.subs(subs))):
                raw['f1<f2,b1>b2'] += 1
            elif (abs(f1.subs(subs)) > abs(f2.subs(subs))
                  and abs(b1.subs(subs)) < abs(b2.subs(subs))):
                raw['f1>f2,b1<b2'] += 1
            else:
                raw['other'] += 1

            # standardized
            if (abs(f1s.subs(subs)) < abs(f2s.subs(subs))
                    and abs(b1s.subs(subs)) > abs(b2s.subs(subs))):
                std['f1<f2,b1>b2'] += 1
            elif (abs(f1s.subs(subs)) > abs(f2s.subs(subs))
                    and abs(b1s.subs(subs)) < abs(b2s.subs(subs))):
                std['f1>f2,b1<b2'] += 1
            else:
                std['other'] += 1

            # scale-harmonized
            if (abs(f1m.subs(subs)) < abs(f2m.subs(subs))
                    and abs(b1m.subs(subs)) > abs(b2m.subs(subs))):
                moj['f1<f2,b1>b2'] += 1
            elif (abs(f1m.subs(subs)) > abs(f2m.subs(subs))
                    and abs(b1m.subs(subs)) < abs(b2m.subs(subs))):
                moj['f1>f2,b1<b2'] += 1
            else:
                moj['other'] += 1

        print('weight_range', weight_range)

        raw['correct'] = raw['f1<f2,b1>b2'] + raw['other'] / 2
        print('raw\t\t', raw)

        std['correct'] = std['f1<f2,b1>b2'] + std['other'] / 2
        print('standardized\t', std)

        moj['correct'] = moj['f1<f2,b1>b2'] + moj['other'] / 2
        print('Mooij-scaled\t', moj)
        print()
