import numpy as np
from acoustics import generator


def make_noise(noisedb, noiseshape, color='whitenormal'):
  noiselev = 10**(noisedb/20)
  if color=='whitenormal':
    noise = np.random.normal(loc=0, scale=1, size=noiseshape)*noiselev
  else:
    noise = np.zeros(noiseshape, dtype='float32')
    for i in range(noiseshape[0]):
      for j in range(noiseshape[1]):
        noise[i, j, :] = generator.noise(noiseshape[2], color='color')*noiselev
  return noise