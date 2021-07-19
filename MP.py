import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform
from scipy.stats import multivariate_normal as mvnorm


class MCMC:
    def __init__(self, target, step):
        self.target = target
        self.step = step
        self.dim = None
        self.samples = None
        self.weights = None
        self.norms = None

    def sampling(self, size, initial):
        self.dim = len(initial)
        samples = [initial]
        weights = [1]
        while len(weights) < size + 2:
            new = samples[-1] + mvnorm.rvs(mean=np.zeros(self.dim), cov=self.step ** 2)
            if uniform.rvs() <= self.target(new) / self.target(samples[-1]):
                samples.append(new)
                weights.append(1)
            else:
                weights[-1] += 1

        self.samples = np.array(samples[1:-1])
        self.weights = np.array(weights[1:-1])
        print('ESS/size/niter: {:.0f}/{}/{}'
              .format(1 / ((self.weights / self.weights.sum()) ** 2).sum(), size, self.weights.sum()))

    def dist(self, xi, k):
        distances = np.abs(np.abs(xi) - self.norms)
        return np.argsort(distances)[:k]

    def draw(self, x, k):
        self.norms = np.sqrt(np.sum(self.samples ** 2, axis=1))
        min_norm = self.norms.min()
        num = np.int64((x[1] - x[0]) / min_norm)
        print('Number: {}'.format(num))
        x = np.linspace(x[0], x[1], num + 1)
        X = np.zeros([x.size, self.dim])
        X[:, 0] = x
        proposal = self.target(self.samples) / (self.weights / self.weights.mean())
        proposalX = np.zeros_like(x)
        for i, xi in enumerate(x):
            index = self.dist(xi, k)
            proposalX[i] = proposal[index].mean()

        fig, ax = plt.subplots()
        ax.plot(x, proposalX, c='r', label='proposal')
        ax.plot(x, self.target(X), c='b', label='target')
        ax.legend()
        ax.set_title('{}-D target and MCMC proposal (averaging)'.format(self.dim))
        plt.show()


def main(dim, step, size):
    target = mvnorm(mean=np.zeros(dim)).pdf
    mcmc = MCMC(target, step=step)
    mcmc.sampling(size=size, initial=np.zeros(dim))
    mcmc.draw(x=(0, 4), k=1)


if __name__ == '__main__':
    main(dim=2, step=1, size=100000)
    main(dim=3, step=1, size=100000)
    main(dim=4, step=1, size=100000)
    main(dim=5, step=1, size=100000)
