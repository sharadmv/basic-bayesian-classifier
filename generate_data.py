import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

prior = [0.3, 0.7]
parameters = [[30, 4], [16, 3]]

def gen():
    c = np.random.binomial(1, prior[1])
    mu, sigma = parameters[c]
    return [np.random.normal(loc=mu, scale=sigma), c]

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("N", type=int, help="number of points")
    argparser.add_argument("--plot", action='store_true')

    args = argparser.parse_args()

    samples = []
    for _ in range(args.N):
        samples.append(gen())

    if args.plot:
        binwidth = 1
        c0 = [sample[0] for sample in samples if sample[1] == 0]
        c1 = [sample[0] for sample in samples if sample[1] == 1]
        plt.figure()
        bins = np.arange(min(c0 + c1),max(c0 + c1)+binwidth,binwidth)
        plt.hist(c0, bins=bins, alpha=0.5, label='Salmon')
        plt.hist(c1, bins=bins, alpha=0.5, label='Trout')
        plt.legend()
        plt.savefig("../sharadvikram.com/public/img/bayes-training_data.png")

    for sample in samples:
        print '\t'.join(map(str, sample))


