import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from math import sqrt, pi

from argparse import ArgumentParser

def logposterior(x, mu, sigma, p):
    return -np.sqrt(2 * pi * sigma ** 2) - 0.5 / sigma ** 2 * (x - mu) ** 2 + np.log(p)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('test')
    args = argparser.parse_args()

    total = 0
    counts = [0, 0]
    sums = [0, 0]
    sums2 = [0, 0]

    points = [[], []]

    for line in sys.stdin:
        vec = line.strip().split('\t')
        x, y = float(vec[0]), int(vec[1])
        points[y].append(x)
        counts[y] += 1
        sums[y] += x
        sums2[y] += x * x
        total += 1
    p = (float(counts[1]) / total)
    mu_0 =  (sums[0]/counts[0])
    sigma_0 = sqrt((sums2[0] - 2 * mu_0 * sums[0] + counts[0] * mu_0 ** 2) / counts[0])
    mu_1 =  (sums[1]/counts[1])
    sigma_1 = sqrt((sums2[1] - 2 * mu_1 * sums[1] + counts[1] * mu_1 ** 2) / counts[1])
    print "p=%f" % p
    print "mu_0=%f" % mu_0
    print "sigma_0=%f" % sigma_0
    print "mu_1=%f" % mu_1
    print "sigma_1=%f" % sigma_1

    x = np.arange(mu_0 - 3 * sigma_0, mu_0 + 3 * sigma_0, 0.01)

    plt.figure()
    plt.plot(x, stats.norm.pdf(x, loc=mu_0, scale=sigma_0))
    plt.hist(points[0], normed=True, bins=100)
    plt.savefig('gaussian-0.png')


    x = np.arange(mu_1 - 3 * sigma_1, mu_1 + 3 * sigma_1, 0.01)
    plt.figure()
    plt.plot(x, stats.norm.pdf(x, loc=mu_1, scale=sigma_1))
    plt.hist(points[1], normed=True, bins=100)
    plt.savefig('gaussian-1.png')

    predictions = [[], []]
    correct, total = 0, 0
    with open(args.test) as test_file:
        for line in test_file:
            vec = line.strip().split('\t')
            x, y = float(vec[0]), int(vec[1])
            predictions[1].append(y)
            p0, p1 = logposterior(x, mu_0, sigma_0, 1 - p), logposterior(x, mu_1, sigma_1, p)
            prediction = int(p1 > p0)
            if prediction == y:
                correct += 1
            total += 1
    print "Error rate: %f" % (1 - float(correct)/total)


