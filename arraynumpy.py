### Array Attributes and Methods
### Numpy arrays

#import numpy
import numpy as np
#convert list to array
my_list = [1,2,3]
np.array(my_list)
# output : array([1, 2, 3])


# two dimentional array
my_list = [[1,2,3],[4,5,6],[7,8,9]]
np.array(my_list)
# output : array([[1, 2, 3],[4, 5, 6], [7, 8, 9]])


#arange
np.arange(0,10)
# output: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.arange(0,10,2)
# output: array([0, 2, 4, 6, 8])


# zeros and ones
np.zeros(3)
# output: array([0., 0., 0.])
np.zeros((3,3))
# output: array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])


#linspace
#Return evenly spaced numbers over a specified interval
np.linspace(0,10,5)
#output: array([ 0. ,  2.5,  5. ,  7.5, 10. ])


#eye
#returns a 2-D array with 1's as the diagonal and 0's elsewhere
np.eye(5)
#output: array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.],
#              [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])


#random.rand
#Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
np.random.rand(4)
#output: array([0.34875947, 0.60102242, 0.91969882, 0.48373634])


np.random.rand(4,5)
#output:array([[0.03111796, 0.63832928, 0.20898114, 0.15069603, 0.08955784],
#       [0.68028316, 0.34605671, 0.3816151 , 0.82935361, 0.21525307],
#       [0.7339131 , 0.26519259, 0.30119238, 0.66117268, 0.91405936],
#       [0.325582  , 0.59615388, 0.98757414, 0.60669361, 0.38189827]])


#random.randn
#Return a sample (or samples) from the “standard normal” distribution
#The standard normal distribution, also called the z-distribution, is a special normal distribution
#where the mean is 0 and the standard deviation is 1
np.random.randn(2)
#output:array([-0.42454629,  1.17887657])


np.random.randn(10000).mean()
#output:-0.003716264376782245


np.random.randn(10000).std()
#output:0.9982119352866324


#random.randint
#Return random integers from low (inclusive) to high (exclusive)
np.random.randint(0,40)
#output:10


np.random.randint(1,15,6)
#output:array([13,  2, 10,  9,  1,  6])

#reshape
#allows us to reshape an array in Python
arr = np.arange(25)
randarr = np.random.randint(0,50,10)
arr.reshape(5,5)
#output: array([[ 0,  1,  2,  3,  4],
#       [ 5,  6,  7,  8,  9],
#       [10, 11, 12, 13, 14],
#       [15, 16, 17, 18, 19],
#       [20, 21, 22, 23, 24]])


np.random.randint(0,50,15).reshape(3,5)
#output: #array([[ 3, 38, 48, 37, 48],
#       [36, 11, 36, 42, 36],
#       [39, 46, 34, 23, 40]])


#shape
# used to fetch the dimensions of Pandas and NumPy type objects
arr.shape
# the output is vector
#output: (25,)


arr.reshape(1,25).shape
# the output is matrix
#output: (1, 25)


#dtype
#create a data type object
arr.dtype
#output: dtype('int32')


new_arr = np.array([1,2,3,4],dtype = 'float64')
new_arr.dtype
#output: dtype('float64')


new_arr = np.array([1.6,2.3,5.1,7.9], dtype = 'int32')
print(new_arr)
#output: array([1, 2, 5, 7])

#max
randarr.max()
#output: 35



#min
randarr.min()
#output: 11


#argmax
#Returns the indices of the maximum values along an axis
randarr.argmax()
#output: 6


#argmin
randarr.argmin()
#output: 0

### Numpy indexing and Selection
# indexing and selection in vectors are same as python slicing
arr[:5]
#output: array([0, 1, 2, 3, 4])


arr[3:6]
#output: array([3, 4, 5])

#Broadcasting
#lists in python don't have broadcasting
arr[0:5] = 100
print(arr)
#output: array([100, 100, 100, 100, 100,   5,   6,   7,   8,   9,  10,  11,  12,
#        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24])


slice_of_arr = arr[0:6]
slice_of_arr[:] = 99
print(arr)
#output: array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#       17, 18, 19, 20, 21, 22, 23, 24])


#copy()
arr = np.arange(25)
copy_of_arr = arr.copy()[0:6]
copy_of_arr[:] = 99
print(arr)
#output: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#       17, 18, 19, 20, 21, 22, 23, 24])

#### indexing a 2D array
arr_2d = np.array([[5,10,15],[20,25,30],[35,40,45]])
print(arr_2d[1])
#output: array([20, 25, 30])


arr_2d[1][2]
#output: 30

arr_2d.transpose()
#output: array([[ 5, 20, 35],
#       [10, 25, 40],
#       [15, 30, 45]])

arr = np.arange(0,100).reshape(10,10)
print(arr)
# output : array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
#       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
#       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
#       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
#       [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
#       [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
#       [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
arr[1:5]
#output: array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
#       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
#       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])

arr[1:5][2]
#output: array([30, 31, 32, 33, 34, 35, 36, 37, 38, 39])


arr[2][1]
#output: 21


arr[3:,5:]
#output: array([[35, 36, 37, 38, 39],
#       [45, 46, 47, 48, 49],
#       [65, 66, 67, 68, 69],
#       [75, 76, 77, 78, 79],
#       [85, 86, 87, 88, 89],
#       [95, 96, 97, 98, 99]])



arr[3:][5:]
#output: array([[80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
#       [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])


arr[3:,5:][0:2]
#output: array([[35, 36, 37, 38, 39],
#       [45, 46, 47, 48, 49]])


arr[3:,5:][0,2]
#output: 37

#in 2D matrix : arr[:,:] in which first is for rows and second is for columns
arr[4:6,1:6]
#output: array([[41, 42, 43, 44, 45],
#       [51, 52, 53, 54, 55]])
