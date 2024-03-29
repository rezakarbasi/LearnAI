import numpy as np

a = np.array([[0, 1], [2, 3]])
b = np.array([[2, 2], [1, 0]])

print('\ndot function product two array mathematically\n', a.dot(b))
print('\n\nand * operator products each element of array separately\n', a*b)

print('\n\nshape of the array is ', np.shape(a))
print('\n\nnp.random.random(()) makes a random array : \n', np.random.random((2, 3)))
