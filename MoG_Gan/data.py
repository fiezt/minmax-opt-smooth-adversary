import numpy as np
import matplotlib.pyplot as plt

"""
    generate 2d gaussian around a circle
"""
class gaussian_data_generator(object):
    def __init__(self):

        n = 8
        radius = 1
        std = 0.05
        delta_theta = 2*np.pi / n

        centers_x = []
        centers_y = []
        for i in range(n):
            centers_x.append(radius*np.cos(i*delta_theta))
            centers_y.append(radius*np.sin(i*delta_theta))

        centers_x = np.expand_dims(np.array(centers_x), 1)
        centers_y = np.expand_dims(np.array(centers_y), 1)

        p = [1./n for _ in range(n)]

        self.p = p
        self.size = 2
        self.n = n
        self.std = std
        self.centers = np.concatenate([centers_x, centers_y], 1)

    def random_distribution(self, p=None):
        if p is None:
            p = [np.random.uniform() for _ in range(self.n)]
            p = p / np.sum(p)
        self.p = p

    def uniform_distribution(self):
        p = [1./self.n for _ in range(self.n)]
        self.p = p

    def sample(self, N):
        
        """
        batch_size=N
        skel = np.array([[np.sin(t), np.cos(t)]
                     for t in np.linspace(0,2*np.pi,9)[:-1]])
        mixture = np.random.choice(range(8), batch_size)
        real = skel[mixture] + sigma*np.random.randn(batch_size, 2)

        return real
        """
        
        n = self.n
        std = self.std
        centers = self.centers

        ith_center = np.random.choice(n, N,p=self.p)
        sample_centers = centers[ith_center, :]
        sample_points = np.random.normal(loc=sample_centers, scale=std)
        return sample_points.astype('float32')


def plot(points):
    plt.scatter(points[:, 0], points[:, 1], c=[0.3 for i in range(1000)], alpha=0.5)
    #plt.show()
    plt.close()


def noise_sampler(N, z_dim):
    # random noise for generator
    return np.random.normal(size=[N, z_dim]).astype('float32')



def main():
    gen = gaussian_data_generator()
    sample_points = gen.sample(1000)
    plot(sample_points)

if __name__ == '__main__':
    main()
