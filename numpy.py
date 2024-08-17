'''Source : W3Schools'''
'''Why is NumPy Faster Than Lists?
NumPy arrays are stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently.
This behavior is called locality of reference in computer science.
This is the main reason why NumPy is faster than lists. Also it is optimized to work with latest CPU architectures.'''

'''Which Language is NumPy written in?
NumPy is a Python library and is written partially in Python, but most of the parts that require fast computation are written in C or C++.'''

arr = numpy.array([1, 2, 3, 4, 5])

arr = np.array((1, 2, 3, 4, 5))

arr = np.array([[1, 2, 3], [4, 5, 6]])

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(arr.ndim)

print('2nd element on 1st row: ', arr[0, 1])

print('5th element on 2nd row: ', arr[1, 4])

print('Last element from 2nd dim: ', arr[1, -1])

arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5:2])
[2,4]

print(arr[0:2, 1:4])
'''
i - integer
b - boolean
u - unsigned integer
f - float
c - complex float
m - timedelta
M - datetime
O - object
S - string
U - unicode string
V - fixed chunk of memory for other type ( void )
'''

arr = np.array([1, 2, 3, 4], dtype='S')

newarr = arr.astype('i')

x = arr.copy()

x = arr.view()

x = arr.copy()
y = arr.view()
print(x.base) # return none if it owns the data
print(y.base)

print(arr.shape)

newarr = arr.reshape(4, 3)

newarr = arr.reshape(2, 3, 2)

# reshape returns a view of the array

newarr = arr.reshape(2, 2, -1)

for x in arr:
  print(x)

for x in arr:
  for y in x:
    print(y)

for x in np.nditer(arr):
  print(x)

'''We can use op_dtypes argument and pass it the expected datatype to change the datatype of elements while iterating.
NumPy does not change the data type of the element in-place (where the element is in array) so it needs some other space to perform this action,
that extra space is called buffer, and in order to enable it in nditer() we pass flags=['buffered']. '''
for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x)

for x in np.nditer(arr[:, ::2]):
  print(x)

for idx, x in np.ndenumerate(arr):
  print(idx, x)

arr = np.concatenate((arr1, arr2))

arr = np.concatenate((arr1, arr2), axis=1) // join along rows

'''Stacking is same as concatenation, the only difference is that stacking is done along a new axis.
We can concatenate two 1-D arrays along the second axis which would result in putting them one over the other, ie. stacking.
We pass a sequence of arrays that we want to join to the stack() method along with the axis. If axis is not explicitly passed it is taken as 0.'''
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.stack((arr1, arr2), axis=1)

arr = np.hstack((arr1, arr2))

arr = np.vstack((arr1, arr2))

arr = np.dstack((arr1, arr2))

arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)
[array([1, 2]), array([3, 4]), array([5, 6])]
newarr = np.array_split(arr, 4)
[array([1, 2]), array([3, 4]), array([5]), array([6])]

x = np.where(arr == 4)

x = np.where(arr%2 == 0)

x = np.searchsorted(arr, 7) //does a binary search

x = np.searchsorted(arr, 7, side='right') #Find the indexes where the value 7 should be inserted, starting from the right

x = np.searchsorted(arr, [2, 4, 6]) #Find the indexes where the values 2, 4, and 6 should be inserted

np.sort(arr) #If you use the sort() method on a 2-D array, both arrays will be sorted:

#Filtering 
x = [True, False, True, False]
newarr = arr[x]
[41, 43]

filter_arr = arr > 42
