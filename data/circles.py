import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt

class Circle(object):
    def __init__(self, 
                 size, 
                 radius):
        
        assert(size % 2 == 0)
        assert(radius % 2 == 0)

        self.size = size
        self.radius = radius
        self.x = self.gen_x()

    def gen_x(self):
        x = np.zeros([self.size, self.size])
        xx,yy = np.mgrid[:self.size, :self.size]
        circle = (xx - self.size/2)**2 + (yy - self.size/2)**2 <= self.radius**2
        return np.array(np.logical_or(x, circle), dtype=np.float32)
    
    def next_batch(self, 
                   batch_size):

        x = np.stack([self.x] * batch_size, axis=0)
        y = np.zeros([batch_size, self.size, self.size])

        gt = randint(low=(self.size-self.radius)/2,
                     high=(self.size + self.radius)/2 + 1,
                     size=(batch_size, 2))

        gt = np.insert(gt, 0, np.arange(batch_size), axis=1)

        y[tuple(np.array(gt).T)] = 1.0
        return x, y

    def plot_batch(self, batch, batch_number):
        plt.imshow(batch[0][batch_number,:,:])
        plt.imshow(batch[1][batch_number,:,:], alpha=0.5)
        plt.show()

if __name__== "__main__":
    c = Circle(28, 12)
    for j in range(200):
        batch = c.next_batch(200)
