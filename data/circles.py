import numpy as np
from numpy.random import randint

class Circle(object):
    def __init__(self, 
                 width, 
                 height, 
                 radius):
        assert(width % 2 == 0)
        assert(height % 2 == 0)
        assert(radius % 2 == 0)

        self.width = width
        self.height = height
        self.radius = radius
        self.x = self.gen_x()

    def gen_x(self):
        x = np.zeros([self.height, self.width])
        xx,yy = np.mgrid[:self.height, :self.width]
        circle = (xx - self.height/2)**2 + (yy - self.width)**2 <= self.radius**2
        return np.array(np.logical_or(x, circle), dtype=np.float32)
    
    def next_batch(self, 
                   batch_size,
                   ):
        x = np.stack([self.x] * batch_size, axis=0)
        print(np.shape(x))
        y = np.zeros([batch_size, self.width, self.height])
        gt = randint(low=-self.radius/2,
                        high=self.radius/2 + 1,
                        size=(batch_size, 1,1))
        #y[gt] = 1.0
        return x


if __name__== "__main__":
    c = Circle(28, 28, 12)
    print(c.next_batch(2))
