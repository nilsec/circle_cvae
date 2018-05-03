import numpy as np
import time
from numpy.random import randint
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


class Circle(object):
    def __init__(self, 
                 size, 
                 radius):
        
        assert(size % 2 == 0)
        assert(radius % 2 == 0)

        self.size = size
        self.radius = radius
        self.x = self.gen_x()

    def gen_x(self, x=None, y=None):
        if x == None:
            xc = self.size/2
        if y == None:
            yc = self.size/2

        x = np.zeros([self.size, self.size])
        xx,yy = np.mgrid[:self.size, :self.size]
        circle = (xx - xc)**2 + (yy - yc)**2 <= self.radius**2
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


class MovingCircle(object):
    def __init__(self,
                 size,
                 radius):

        self.size = size
        self.radius = radius

    def gen_x(self, gt, batch_size):
        zz, yy, xx = np.mgrid[:batch_size, :self.size, :self.size]        
        x = np.zeros([batch_size, self.size, self.size])

        cylinder = (((xx.T - gt.T[1,:])**2 + (yy.T - gt.T[0,:])**2 )<= self.radius**2).T
        return cylinder
         
    def next_batch(self, batch_size):
        gt = randint(low=0,
                     high=self.size,
                     size=(batch_size, 2))

        offset = randint(low=-self.radius/2,
                         high=self.radius/2 + 1,
                         size=(batch_size, 2))

        x = self.gen_x(gt, batch_size)

        gt += offset
        gt[np.where(gt>=self.size)] = self.size - 1
        gt[np.where(gt<0)] = 0
        gt = np.insert(gt, 0, np.arange(batch_size), axis=1)
        y = np.zeros([batch_size, self.size, self.size])
        y[tuple(np.array(gt).T)] = 1.0

        return x,y
        
    def plot_batch(self, batch, batch_number):
        plt.imshow(batch[0][batch_number,:,:])
        plt.imshow(batch[1][batch_number,:,:], alpha=0.5)
        plt.show()


if __name__== "__main__":
    # 0.0005 seconds per call
    t0 = time.time()
    c = Circle(28, 12)
    for j in range(1000):
        c.next_batch(200)    
    t1 = time.time()
    print("Circle avg. batch gen time: {}".format((t1-t0)/1000.))
    
    # 0.006 seconds per call ~ 10 times slower
    c = MovingCircle(28, 12)
    t0 = time.time()
    for j in range(1000):
        c.next_batch(200)
    t1 = time.time()
    print("Moving Circle avg. batch gen time: {}".format((t1 - t0)/1000.))
