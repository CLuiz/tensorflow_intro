import numpy as np
from scipy import signal as sg

h = [2,1,0]
x = [3,4,5]

y1 = np.convolve(x,h)
print(f'Standard numpy implementation of convolve: {y1}')
print(f'Compare with the following values in Python: {y1[0]} ; {y1[1]}; {y1[2]}; {y1[3]}; {y1[4]}')

# Effects of using zero padding
h = [1,2,5,4]
x = [6,2]

y2 = np.convolve(x, h, 'full')
print(f'"Full" numpy implementation of convolve: {y2}')

# Effect of 'same' in numpy convolve
y3 = np.convolve(x, h, 'same')
print(f'"Same" numpy implementation of convolve: {y3}')

# Effect of 'valid' in numpy convolve
y4 = np.convolve(x, h, 'valid')
print(f'"Valid" numpy implementation of convolve: {y4}')

# 2d convolutional operations with scipy.signal
I = [[255,   7,   3],
     [212, 240,   4],
     [218, 216, 230],]
g = [[-1,1]]

print('Without zero padding: \n')
print(f'{sg.convolve(I, g, "valid")} \n')

# The 'valid' argument states that the output consists only of
# those elements that do not rely on the zero-padding.
print('With zero padding: \n')
print(f'{sg.convolve(I, g)} \n')
