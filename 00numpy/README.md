- [NumPy](#numpy)
  - [Why Numpy?](#why-numpy)
    - [Vectorization](#vectorization)
  - [Array Creation](#array-creation)
    - [1) Python sequences to NumPy Arrays - `np.array()`](#1-python-sequences-to-numpy-arrays---nparray)
      - [type conversation](#type-conversation)
    - [2) Functions for Generating General Arrays](#2-functions-for-generating-general-arrays)
      - [`arange(start, end [exclusive], step)`](#arangestart-end-exclusive-step)
      - [`numpy.linspace(s,en,equally spaced between)`](#numpylinspacesenequally-spaced-between)
      - [`zeros()`](#zeros)
      - [`np.ones()`](#npones)
      - [`np.diag()` , `np.identity()` and `np.eye()`](#npdiag--npidentity-and-npeye)
      - [`np.full`](#npfull)
      - [`np.rand` , `randint()` and `np.randn`](#nprand--randint-and-nprandn)
        - [`random.rand()`](#randomrand)
        - [`random.randn()`](#randomrandn)
        - [`randint()`](#randint)
        - [NumPy Random Seed](#numpy-random-seed)
  - [Array Dimension](#array-dimension)
    - [Some vocabulary](#some-vocabulary)
    - [What is the Shape of an Array](#what-is-the-shape-of-an-array)
    - [What is the Axis of a Numpy Array](#what-is-the-axis-of-a-numpy-array)
    - [Difference between array shape `(n,)` vs `(1,n)` vs `(n,1)` 🚀🚀🚀](#difference-between-array-shape-n-vs-1n-vs-n1-)
    - [Using `axis` keyword in aggregation functions](#using-axis-keyword-in-aggregation-functions)
    - [Axis concept in 🚀  `Pandas` 🚀](#axis-concept-in---pandas-)
  - [N-dimensional arrays 🌟🌟🌟](#n-dimensional-arrays-)
    - [Understanding Nature of 3d Array 🌟🌟🌟](#understanding-nature-of-3d-array-)
    - [Working With 3d Array](#working-with-3d-array)
    - [Converting to 1D array with `flatten()` method](#converting-to-1d-array-with-flatten-method)
  - [Indexing and Masking](#indexing-and-masking)
    - [Indexing](#indexing)
    - [slice operator `[begin:end]`](#slice-operator-beginend)
      - [Stride](#stride)
      - [More In Depth - shape,dimensionality](#more-in-depth---shapedimensionality)
      - [Negative slicing of NumPy arrays](#negative-slicing-of-numpy-arrays)
    - [Masking](#masking)
    - [Selecting values from your array that fulfill certain conditions](#selecting-values-from-your-array-that-fulfill-certain-conditions)
      - [Indexing with a mask can be very useful to assign a new value to a sub-array](#indexing-with-a-mask-can-be-very-useful-to-assign-a-new-value-to-a-sub-array)
      - [`np.where()` to select elements or indices from an array.](#npwhere-to-select-elements-or-indices-from-an-array)
  - [Mathematical Operations](#mathematical-operations)
    - [Matrix Arithmetic](#matrix-arithmetic)
    - [with scalars](#with-scalars)
    - [Transcendental functions](#transcendental-functions)
    - [Shape Mismatch](#shape-mismatch)
    - [Dot Product](#dot-product)
    - [Matrix Aggregation](#matrix-aggregation)
    - [Logical Operations](#logical-operations)
  - [Statistics](#statistics)
  - [Miscellaneous Operations](#miscellaneous-operations)
    - [Unique Items and Count](#unique-items-and-count)
    - [Reversing Rows and Columns](#reversing-rows-and-columns)
  - [Broadcasting](#broadcasting)
  - [Shape Manipulation - Transposing, Reshaping, Stacking etc...](#shape-manipulation---transposing-reshaping-stacking-etc)
    - [Flatten](#flatten)
    - [Reshape](#reshape)
    - [Transpose](#transpose)
    - [Adding a Dimension](#adding-a-dimension)
    - [Stacking of Array](#stacking-of-array)
  - [🌟Vectorization🌟](#vectorization-1)
    - [Machine Learning context](#machine-learning-context)

# NumPy


```python
import numpy as np
# jupyter nbconvert --to markdown numpy.ipynb --output README.md
# <div align="center"><img src="img/ndim.png" alt="dfs" width="800px"></div>
```

## Why Numpy?

- performs fast operations (because of Vectorization)
- `numpy` arrays can be treated as vectors and matrices from linear algebra (Vectorization)


```python
l = [1,2,3,4,5,6,7,8,9,10]
```


```python
%timeit [i**2 for i in l]
```

    3.52 µs ± 1.05 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)



```python
arr = np.array(l)
arr
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])




```python
%timeit arr**2
```

    975 ns ± 122 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


### Vectorization


```python
# adding 1 to each element of this vector
l + 1
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-43-9fd71c034c35> in <module>
          1 # adding 1 to each element of this vector
    ----> 2 l + 1


    TypeError: can only concatenate list (not "int") to list



```python
# but it's possible in numpy
arr + 1
```




    array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11])



## Array Creation

### 1) Python sequences to NumPy Arrays - `np.array()`


```python
a1D = np.array([1, 2, 3, 4])
print(a1D)
print()
# 2D
a2D = np.array([[1, 2], [3, 4]])
print(a2D)
```

    [1 2 3 4]

    [[1 2]
     [3 4]]


#### type conversation

- `np.array(list,dtype='float')`
- `arr.astype('int')`


```python
### int to float
l2 = [[1,2,3],
		[4,5,0]]
a2 = np.array(l2,dtype='float')
a2
```




    array([[1., 2., 3.],
           [4., 5., 0.]])




```python
# float to int
a3 = a2.astype('int')
a3
```




    array([[1, 2, 3],
           [4, 5, 0]])




```python
a4 = np.array(l2,dtype="bool")
a4
```




    array([[ True,  True,  True],
           [ True,  True, False]])




```python
# to list
l3 = a3.tolist()
print(type(a3))
print(type(l3))

```

    <class 'numpy.ndarray'>
    <class 'list'>


### 2) Functions for Generating General Arrays


#### `arange(start, end [exclusive], step)`

`numpy.arange` creates arrays with regularly incrementing values. `arange` is an array-valued version of the built-in Python `range` function


```python
print(np.arange(10))  # 0.... n-1

print()

print(np.arange(1,10)) #start, end (exclusive)

print()

print(np.arange(1, 10, 2))  # start, end (exclusive), step

```

    [0 1 2 3 4 5 6 7 8 9]

    [1 2 3 4 5 6 7 8 9]

    [1 3 5 7 9]


It also works with floats:


```python
print(np.arange(1.0, 5.0))
print(np.arange(1, 5, 0.5))

```

    [1. 2. 3. 4.]
    [1.  1.5 2.  2.5 3.  3.5 4.  4.5]


#### `numpy.linspace(s,en,equally spaced between)`

`numpy.linspace`  will create arrays with a specified number of elements, and spaced equally between the specified beginning and end values.


```python
print(np.linspace(1, 10, 5))  # start, end, number of points
print(np.linspace(0, 5/3, 6))
```

    [ 1.    3.25  5.5   7.75 10.  ]
    [0.         0.33333333 0.66666667 1.         1.33333333 1.66666667]


####  `zeros()`

Many other NumPy functions create `ndarrays`.

Here's a 3x4 matrix full of ones:


```python
print(np.zeros(10))
print("2d..............")
print(np.zeros((3,3)))
print("3d.............")
print(np.zeros((2,3,3)))
```

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    2d..............
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    3d.............
    [[[0. 0. 0.]
      [0. 0. 0.]
      [0. 0. 0.]]

     [[0. 0. 0.]
      [0. 0. 0.]
      [0. 0. 0.]]]


#### `np.ones()`


```python
print(np.ones(10))
print()
print(np.ones((3,3)) * 5)
```

    [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

    [[5. 5. 5.]
     [5. 5. 5.]
     [5. 5. 5.]]


#### `np.diag()` , `np.identity()` and `np.eye()`


```python
np.diag([1,2,3,4])
```




    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])




```python
np.identity(4)
```




    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])




```python
c = np.eye(4)  # Return a 2-D array with ones on the diagonal and zeros elsewhere.
print(c)
print()
# 3 is number of rows, 2 is number of columns, index of diagonal start with 0
d = np.eye(3, 2)
print(d)

```

    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    [[1. 0.]
     [0. 1.]
     [0. 0.]]


#### `np.full`

Creates an array of the given shape initialized with the given value. Here's a 3x4 matrix full of `π`.


```python
np.full((3, 4), np.pi)
```




    array([[3.14159265, 3.14159265, 3.14159265, 3.14159265],
           [3.14159265, 3.14159265, 3.14159265, 3.14159265],
           [3.14159265, 3.14159265, 3.14159265, 3.14159265]])



#### `np.rand` , `randint()` and `np.randn`

We can generate an array of random numbers using `rand()`, `randn()` or `randint()` functions.

##### `random.rand()`

- Using `random.rand()`, we can generate an array of random numbers of the shape we pass to it from uniform distribution over 0 to 1.


```python
# For example, say we want a one-dimensional array of 4 objects that are uniformly
# distributed from 0 to 1, we can do this:

r = np.random.rand(4)
print(r)

# And if we want a two-dimensional array of 3rows and 2columns:
print()
r = np.random.rand(3, 2)
print(r)
```

    [0.9448474  0.93929097 0.4827775  0.65302109]

    [[0.26754535 0.03942572]
     [0.31236268 0.85961221]
     [0.99364858 0.35495844]]


##### `random.randn()`

- Using `randn()`, we can generate random samples from **Standard, normal or Gaussian distributioncentered around 0** . For example, let’s generate 7 random numbers:


```python
np.random.randn(3, 4)

```




    array([[-0.05719047,  1.82009335,  1.10657625,  0.229511  ],
           [ 0.04197836,  2.06061915, -0.27225969, -1.41491265],
           [-0.11479308, -0.92311692, -1.05784058, -1.25812064]])



To give you a feel of what these distributions look like, let's use matplotlib:


```python
import matplotlib.pyplot as plt
```


```python
plt.hist(np.random.rand(100000), density=True, bins=100,
         histtype="step", color="blue", label="rand")
plt.hist(np.random.randn(100000), density=True, bins=100,
         histtype="step", color="red", label="randn")
plt.axis([-2.5, 2.5, 0, 1.1])
plt.legend(loc="upper left")
plt.title("Random distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

```



![png](README_files/README_47_0.png)



##### `randint()`

- Lastly, we can use the `randint()` function to generate an array of integers.
- The `randint()` function can take up to 3 arguments;
    - the low(inclusive),
    - high(exclusive)
    - size of the array.


```python
print(np.arange(10)) # Remember!! not random but sequential
print(np.random.randint(20)) #generates a random integer exclusive of 20
print()
print(np.random.randint(2, 20))#generates a random integer including 2 but excluding 20
print()
print(np.random.randint(2, 20, 7))#generates 7 random integers including 2 but excluding 20
print()
print(np.random.randint(2, 20, (2,3)))#generates 2D array of shape 2,3
```

    17

    12

    [10  8 18 18  5 16 15]

    [[13  8 13]
     [ 5  8  9]]


##### NumPy Random Seed

If we set the np.random.seed(a_fixed_number) every time you call the numpy's other random function, the result will be the same:


```python
r= np.random.randint(10,size=(2,3))
print(r)
# changes each time
```

    [[2 3 8]
     [1 3 3]]



```python
np.random.seed(43)
r= np.random.randint(10,size=(2,3))
print(r) # fixed
```

    [[4 0 1]
     [5 0 3]]



```python
print('Printing as a single array :','\n',a[1:,0:2,0:2].flatten())
```

    Printing as a single array :
     [ 7  8  9 10 13 14 15 16 13 14 15 16]


## Array Dimension

### Some vocabulary



* In NumPy, each dimension is called an **axis**.
* The number of axes is called the **rank**.
    * For example, the above 3x4 matrix is an array of rank 2 (it is 2-dimensional).
    * The first axis has length 3, the second has length 4.
* An array's list of axis lengths is called the **shape** of the array.
    * For example, the above matrix's shape is `(3, 4)`.
    * The rank is equal to the shape's length.
* The **size** of an array is the total number of elements, which is the product of all axis lengths (eg. 3*4=12)


Basic Attributes of the ndarray Class in numpy:

<div align="center"><img src="img/ndim.png" alt="dfs" width="800px"></div>

[https://medium.com/analytics-vidhya/axes-and-dimensions-in-numpy-and-pandas-array-a2490f72631c](https://medium.com/analytics-vidhya/axes-and-dimensions-in-numpy-and-pandas-array-a2490f72631c)

### What is the Shape of an Array

<div align="center"><img src="img/shape_1.png" alt="dfs"></div>


```python
a2D = np.array([[67, 63, 87],
               [77, 69, 59],
               [85, 87, 99],
               [79, 72, 71],
               [63, 89, 93],
               [68, 92, 78]])
print('array type: ', type(a2D))
print('dimension: ', a2D.ndim)
print('array element type: ', a2D.dtype)
print("total item: ", a2D.size)
print("shape: ", a2D.shape)
print("shape is indexable: ", a2D.shape[0], "~", a2D.shape[1])
```

    array type:  <class 'numpy.ndarray'>
    dimension:  2
    array element type:  int32
    total item:  18
    shape:  (6, 3)
    shape is indexable:  6 ~ 3


Numpy has a function called `shape` which returns the shape of an array. The shape is a tuple of integers. **These numbers denote the lengths of the corresponding array dimension.**

The `shape` of this array is a tuple with the **number of elements per axis (dimension)**. In our example, the shape is equal to `(6, 3)`, i.e. we have 6 elements (at axis=0) and 3 elements (at axis=1).

### What is the Axis of a Numpy Array

In Mathematics/Physics, dimension or dimensionality is informally defined as the minimum number of coordinates needed to specify any point within a space. But in Numpy, according to the numpy doc, it’s the same as axis/axes:

> In Numpy dimensions are called axes. The number of axes is rank.

In the most simple terms, when you have more than 1-dimensional array than the concept of the Axis is comes at all. For example, the 2-D array has 2 Axis.

The first axis ( i.e. axis-0 ) is running vertically downwards across rows, and the second (axis-1) running horizontally across columns.

<div align="center"><img src="img/shape_2.png" alt="dfs" width="800px"></div>

**Basically simplest to remember it as `0=down` and `1=across`.**

*So a mean calculation on axis-0 will be the mean of all the rows in each column, and a mean on axis-1 will be a mean of all the columns in each row.*

Also explaining more, by definition, the axis number of the dimension is the index of that dimension within the array’s shape. **It is also the position used to access that dimension during indexing**.

For example, if a 2D array a has shape `(5,6)`, then you can access `a[0,0]` up to `a[4,5]`. `Axis 0` is thus the first dimension (the "rows"), and `axis 1` is the second dimension (the "columns").

### Difference between array shape `(n,)` vs `(1,n)` vs `(n,1)` 🚀🚀🚀

<div align = "center" > <img src = "img/shape_2.png" alt = "dfs" width = "500px" > </div >


```python
s = np.array([1, 2, 3]) #1D array
print(s)
# 3 element across axis=0
print("ndim:", s.ndim, ", shape: ", s.shape)
# vs
r = np.array([[1, 2, 3]])  #2D single Row array
print(r)
# 1 element across axis=0, 3 element axis=1
print("ndim:", r.ndim, ", shape: ", r.shape)
# vs
c= np.array([ #2D single Column array
	[1],
	[2],
	[3]
])
print(c)
# 3 element across axis=0, 1 element axis=1
print("ndim:",c.ndim,", shape: ", c.shape)
print()

```

    [1 2 3]
    ndim: 1 , shape:  (3,)
    [[1 2 3]]
    ndim: 2 , shape:  (1, 3)
    [[1]
     [2]
     [3]]
    ndim: 2 , shape:  (3, 1)



`(3,)` Python here tells us the object has three items along the first axis i.e. trailing comma is needed in Python to
indicate that the purpose is a tuple with only one element.


`(n,)` is called a rank 1 array. It doesn't behave consistently as a row vector or column vector which makes some of its operation and effect not intuitive. If we take transpose of this `(n,)` data structure, it will look exactly the same and the dot product will give you a number and not a matrix.

The vector of shape `(n,1)` or `(1,n)` row or column vectors are much more intuitive and consistent.


```python
a= np.array([1,2,3,4])
b= np.array([10,20,30,40])
```


```python
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
try:
	regr.fit(a,b)
except ValueError as v:
	print(v)
```

    Expected 2D array, got 1D array instead:
    array=[1 2 3 4].
    Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.



```python
a=a.reshape(-1,1)
try:
	regr.fit(a, b)
except ValueError as v:
	print(v)
```


```python

```

### Using `axis` keyword in aggregation functions

  - [Matrix Aggregation](#matrix-aggregation)

### Axis concept in 🚀  `Pandas` 🚀

Same Array and Axis concept apply to `Pandas` as well Which is `0=down` and `1=across`.

So a mean calculation on axis-0 will be the mean of all the rows in each column, and a mean on axis-1 will be a mean of all the columns in each row.


```python
import pandas as pd
df = pd.DataFrame([[10, 20, 30, 40], [2, 2, 2, 2], [3, 3, 3, 3]], columns=[
                       "col1", "col2", "col3", "col4"])
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



So if I call df.mean(axis=1), we'll get a mean across the rows:


```python
df.mean(axis=1)
```




    0    25.0
    1     2.0
    2     3.0
    dtype: float64



## N-dimensional arrays 🌟🌟🌟

### Understanding Nature of 3d Array 🌟🌟🌟

NumPy can do everything we’ve mentioned in any number of dimensions. Its central data structure is called ndarray (N-Dimensional Array) for a reason.


<div align="center"><img src="img/numpy_3d.png" alt="Itrtype" width="800px"></div>

In a lot of ways, dealing with a new dimension is just adding a comma to the parameters of a NumPy function:

<div align="center"><img src="img/numpy_3d_1.png" alt="Itrtype" width="800px"></div>

> **Note: Keep in mind that when you print a 3-dimensional NumPy array, the text output visualizes the array differently than shown here. NumPy’s order for printing n-dimensional arrays is that the last axis (axis=2) is looped over the fastest, while the first is the slowest(axis=0). Which means that `np.ones((4,3,2))` would be printed as:**



```python
np.zeros((4,3,2)) # should've print 4,3 matrix of 2 channel
```




    array([[[0., 0.],
            [0., 0.],
            [0., 0.]],

           [[0., 0.],
            [0., 0.],
            [0., 0.]],

           [[0., 0.],
            [0., 0.],
            [0., 0.]],

           [[0., 0.],
            [0., 0.],
            [0., 0.]]])



This is definitely not a 4d Array. The meaning of the dimensions comes from the application and user, not from Python/numpy. Images are often `(height, width, channels)`. Computationally it may be convenient to keep the 3 (or 4) elements of a channel for one pixel together, that is, make it that last dimension. So `(2,4,3)` shape could be thought of as a `(2,4)` image with `3` colors (rgb). The normal `numpy` print isn't the best for visualizing that.

**But if the image is something of `(400, 600, 3)` shape, we don't want a 'print' of the array. We want a plot or image display, a picture, that renders that last dimension as colors.**

If the image is colored, then each pixel is represented by three numbers - a value for each of red, green, and blue. In that case we need a 3rd dimension (because each cell can only contain one number). So a colored image is represented by an ndarray of dimensions: (height x width x 3).

<div align="center"><img src="img/numpy-color-image.png" alt="Itrtype" width="800px"></div>

[why-is-the-print-result-of-3d-arrays-different-from-the-mental-visualisation-of](https://stackoverflow.com/questions/58354395/why-is-the-print-result-of-3d-arrays-different-from-the-mental-visualisation-of)

[https://jalammar.github.io/visual-numpy/](https://jalammar.github.io/visual-numpy/)




```python
import cv2
import matplotlib.pyplot as plt
```


```python
img_cv2 = cv2.imread("img/dog.jpg")
img_plt = plt.imread("img/dog.jpg")

```


```python
print(type(img_cv2))
print(img_cv2.shape)
print()
print(type(img_plt))
print(img_plt.shape)
```

    <class 'numpy.ndarray'>
    (2820, 3760, 3)

    <class 'numpy.ndarray'>
    (2820, 3760, 3)



```python
plt.imshow(img_cv2)
```




    <matplotlib.image.AxesImage at 0x1940b79de20>





![png](README_files/README_87_1.png)




```python
img_cv2[0:2]
```




    array([[[199, 245, 252],
            [199, 245, 252],
            [199, 245, 252],
            ...,
            [ 76, 186, 186],
            [ 76, 186, 186],
            [ 76, 186, 186]],

           [[199, 245, 252],
            [199, 245, 252],
            [199, 245, 252],
            ...,
            [ 76, 186, 186],
            [ 76, 186, 186],
            [ 76, 186, 186]]], dtype=uint8)




```python
img_plt[0:2]
```




    array([[[252, 245, 199],
            [252, 245, 199],
            [252, 245, 199],
            ...,
            [186, 185,  76],
            [186, 185,  76],
            [186, 185,  76]],

           [[252, 245, 199],
            [252, 245, 199],
            [252, 245, 199],
            ...,
            [186, 185,  76],
            [186, 185,  76],
            [186, 185,  76]]], dtype=uint8)




```python
# Convert Colorspaces
img_rgb = cv2.cvtColor(img_cv2,cv2.COLOR_BGR2RGB)
print(img_rgb.shape)
```

    (2820, 3760, 3)



```python
plt.imshow(img_rgb)
```




    <matplotlib.image.AxesImage at 0x1940b843b20>





![png](README_files/README_91_1.png)



### Working With 3d Array


```python
a = np.array([
    [
     [1, 2],
     [3, 4],
     [5, 6]
    ],
    [
     [7, 8],
     [9, 10],
     [11, 12]
    ],
    [[13, 14], [15, 16], [17, 18]],
    [[13, 14], [15, 16], [17, 18]]
    ])
a

```




    array([[[ 1,  2],
            [ 3,  4],
            [ 5,  6]],

           [[ 7,  8],
            [ 9, 10],
            [11, 12]],

           [[13, 14],
            [15, 16],
            [17, 18]],

           [[13, 14],
            [15, 16],
            [17, 18]]])




```python
print(a.shape)
print(a.ndim)
```

    (4, 3, 2)
    3



```python
print(a[0],'\n')
print(a[0, 0], '\n')  # ~ a[0][0] chanel 0, row 0
# chanel 0, first row, first column value
print(a[0,0,0],'\n')
print(a[1],'\n')
```

    [[1 2]
     [3 4]
     [5 6]]

    [1 2]

    1

    [[ 7  8]
     [ 9 10]
     [11 12]]




```python
print("first row from each channel:",'\n')
print(a[:,0,:])
```

    first row from each channel:

    [[ 1  2]
     [ 7  8]
     [13 14]
     [13 14]]



```python
# First one rows for second and third channel:
print(a[1:,:1,:])
print('\n',"watch out!!",'\n')
# First one rows for second and third channel:
print(a[1:,0,:],'\n')

```

    [[[ 7  8]]

     [[13 14]]

     [[13 14]]]

     watch out!!

    [[ 7  8]
     [13 14]
     [13 14]]



### Converting to 1D array with `flatten()` method



## Indexing and Masking

### Indexing

Remember that numpy indices start from `0` and the element at any particular index can be found by `n-1`. For
instance, you access the rst element by referencing the cell at `a[0]` and the second element at `a[1]`.

<div align="center"><img src="img/numpy-array-slice.png" alt="dfs" width="800px"></div>

Unlike accessing arrays in, say, JavaScript, numpy arrays have a powerful selection notation that you can use
to read data in a variety of ways. For instance, we can use commas to select along multiple axes `a[i,j]`.

<div align="center"><img src="img/indexing.jpg" alt="dfs" width="800px"></div>

If a 2D array a has `shape(4,3)`, then you can access `a[0,0]` up to `a[3,2]`. `Axis 0` is thus the first dimension (the "rows"), and `axis 1` is the second dimension (the "columns").


```python
a = np.array([
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9],
    [10, 11, 12]
])
print(a.shape)

```

    (4, 3)



```python
# first row
print(a[0])
# second last row
print(a[1])
# first element
print(a[0,0])
# last element
print(a[3,2])
```

    [1 2 3]
    [4 5 6]
    1
    12


### slice operator `[begin:end]`

`:` is the delimiter of the slice syntax to select a sub-part of a sequence, like: `[begin:end]`.


```python
a = np.array([
    [1, 2, 3],
   	[4, 5, 6],
   	[7, 8, 9],
    [10, 11, 12]
])
a[1:2]

```




    array([[4, 5, 6]])



What about selecting all the first items along the second axis(axis=1) in a? Use the `:` operator:


```python
a[:,0]
```




    array([ 1,  4,  7, 10])



`arr[start_row_idx : end_row_idx + 1, start_col_idx : end_col_idx + 1]`


```python
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])
```

<div align="center"><img src="img/index_slice.jpg" alt="Itrtype" width="800px" ></div>


```python
print(a[0:2, 0:2])
print()
print(a[:2, :2])
# diff. ways of indexing first two rows
print()
print(a[0:2])
print()
print(a[:2, :])
print()
print(a[:2])
print()
print(a[:, 2:])  # all rows of last two rows

```

    [[1 2]
     [5 6]]

    [[1 2]
     [5 6]]

    [[1 2 3 4]
     [5 6 7 8]]

    [[1 2 3 4]
     [5 6 7 8]]

    [[1 2 3 4]
     [5 6 7 8]]

    [[ 3  4]
     [ 7  8]
     [11 12]
     [15 16]]


#### Stride
We can also select regularly spaced elements by specifying a step size a er a second `:`. For example, to select
the first and third element in a we can type:


```python
a[0:-1:2]
```




    array([[1, 2, 3],
           [7, 8, 9]])




```python
# or, simply:
a[::2]
```




    array([[1, 2, 3],
           [7, 8, 9]])



#### More In Depth - shape,dimensionality

- `[0]~[0,:]` gets **all the element** of **_1st row_**
- `[:1,:]` gets **all the element** of **_all the rows till 1st row_**


```python
a = np.array([
	[1, 2, 4],
	[5, 6, 7],
	[8, 9, 10]
])
print(a[0])
print("*"*40)
v1 = a[0, :] # gets all the element of 1st row, with all the columns
print(v1)
print("shape: ",v1.shape)
print("dimension: ",v1.ndim)
print("*"*40)
v2 = a[:1,:] # get all the element of all the rows till 1st row, with all the columns
print(v2)
print("shape: ", v2.shape)
print("dimension: ",v2.ndim)

```

    [1 2 4]
    ****************************************
    [1 2 4]
    shape:  (3,)
    dimension:  1
    ****************************************
    [[1 2 4]]
    shape:  (1, 3)
    dimension:  2



```python
v3 = a[:2, :]  # get  all the `rows` till 2nd, with all the columns
print(v3)
```

    [[1 2 4]
     [5 6 7]]



```python
v1= a[:,1]
print(v1)
print(v1.shape)
print("vs")
v2 = a[:, 1:2]
print(v2)
print(v2.shape)
```

    [2 6 9]
    (3,)
    vs
    [[2]
     [6]
     [9]]
    (3, 1)



```python
print(a[1,:])
print("vs")
print(a[1:2,:])
```

    [5 6 7]
    vs
    [[5 6 7]]


#### Negative slicing of NumPy arrays


```python
# last column only
print(a[:,-1])
print()
print(a[:,2])
```

    [ 4  7 10]

    [ 4  7 10]



```python
#  last two  column only
ans = a[:,-2:]
print(ans)
print()
print(ans[1,0])
```

    [[ 2  4]
     [ 6  7]
     [ 9 10]]

    6


If, however, we wanted to extract from the end, we would have to explicitly provide a negative step-size otherwise the result would be an empty list.


```python
print(a[:,-1:-3:-1])
```

    [[ 4  2]
     [ 7  6]
     [10  9]]



```python
print('Reversed array :','\n',a[::-1,::-1])
```

    Reversed array :
     [[10  9  8]
     [ 7  6  5]
     [ 4  2  1]]


### Masking

NumPy arrays can be indexed with slices, but also with boolean or integer arrays **(masks)**. This method is called **fancy indexing**. It creates copies not views.


```python
a = np.array([[1 , 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [1 , 2, 3, 4]])

a[1:3,1:3] = 0
print(a)
```

    [[ 1  2  3  4]
     [ 5  0  0  8]
     [ 9  0  0 12]
     [ 1  2  3  4]]



```python
mask = a>5
print(mask)
```

    [[False False False False]
     [False False False  True]
     [ True False False  True]
     [False False False False]]



```python
# get all values greater then 5
a[mask]
```




    array([ 8,  9, 12])



### Selecting values from your array that fulfill certain conditions


```python
# print all of the values in the array that are less than 5.
print(a[a<5])

# numbers that are equal to or greater than 5
print()
five_up = (a >= 5)
print(a[five_up])

# elements that are divisible by 2:
print()
divisible_by_2 = a[a%2==0]
print(divisible_by_2)

# elements that satisfy two conditions using the & and | operators:
print()
c = a[(a > 2) & (a < 11)]
print(c)
```

    [1 2 3 4 0 0 0 0 1 2 3 4]

    [ 5  8  9 12]

    [ 2  4  0  0  8  0  0 12  2  4]

    [3 4 5 8 9 3 4]


#### Indexing with a mask can be very useful to assign a new value to a sub-array


```python
a = np.random.randint(0, 20, 15)
print(a)
mask = (a % 2 == 0)
a[mask] = -1
a
```

    [18 14 14  7 11  1 10  4  1 10 10 18 17  5  2]





    array([-1, -1, -1,  7, 11,  1, -1, -1,  1, -1, -1, -1, 17,  5, -1])



#### `np.where()` to select elements or indices from an array.


```python
# print the indices of elements that are, for example, less than 5:
a = np.array([[1 , 5, 6, 4],
              [5, 1, 7, 8],
              [9, 10, 11, 1]])
b = np.where(a<5)
print(b)
```

    (array([0, 0, 1, 2], dtype=int64), array([0, 3, 1, 3], dtype=int64))


In this example, a tuple of arrays was returned: one for each dimension.
The first array represents the row indices where these values are found, and
the second array represents the column indices where the values are found.

## Mathematical Operations

### Matrix Arithmetic

We can add and multiply matrices using `arithmetic operators (+-*/)` if the two matrices are the **same size**. NumPy handles those as **`position-wise`** operations:

<div align="center"><img src="img/Matrix Arithmetic.jpg" alt="Itrtype" width="800px"></div>

We can get away with doing these arithmetic operations on matrices of **different size** only if the different dimension is one (e.g. the matrix has only one column or one row), in which case NumPy uses its **`broadcast`** rules for that operation:

<div align="center"><img src="img/Matrix Arithmetic 2.jpg" alt="Itrtype" width="800px"></div>


```python
a = np.array([10,20,30,40])
b = np.arange(1,5)
print(a)
print(b)
```

    [10 20 30 40]
    [1 2 3 4]



```python
a = np.array([14, 23, 32, 41])
b = np.array([5,  4,  3,  2])
print("a + b  =", a + b)
print("a - b  =", a - b)
print("a * b  =", a * b)
print("a / b  =", a / b)
print("a // b  =", a // b)
print("a % b  =", a % b)
print("a ** b =", a ** b)

```

    a + b  = [19 27 35 43]
    a - b  = [ 9 19 29 39]
    a * b  = [70 92 96 82]
    a / b  = [ 2.8         5.75       10.66666667 20.5       ]
    a // b  = [ 2  5 10 20]
    a % b  = [4 3 2 1]
    a ** b = [537824 279841  32768   1681]


### with scalars


```python
a = np.array([1, 2, 3, 4])  # create an array
a*2
```




    array([2, 4, 6, 8])




```python
a= a ** 2
a
```




    array([ 1,  4,  9, 16], dtype=int32)




```python
# Masking
a>15
```




    array([False,  True,  True,  True])



### Transcendental functions


```python
np.log(b)
```




    array([0.        , 0.69314718, 1.09861229, 1.38629436])




```python
np.sin(a)
```




    array([-0.54402111,  0.91294525, -0.98803162,  0.74511316])



### Shape Mismatch


```python
a = np.arange(4)

a + np.array([1, 2])

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-26-e480c525d1d6> in <module>
          1 a = np.arange(4)
          2
    ----> 3 a + np.array([1, 2])


    ValueError: operands could not be broadcast together with shapes (4,) (2,)



```python
a = np.array([[1,2],[3,4]])
b = np.array([1,1])
a + b

```




    array([[2, 3],
           [4, 5]])



### Dot Product

To multiply an `m×n` matrix by an `n×p` matrix, the `n`s must be the same,
and the result is an `m×p` matrix.

A key distinction to make with arithmetic is the case of matrix multiplication using the dot product. NumPy gives every matrix a dot() method we can use to carry-out dot product operations with other matrices:

<div align="center"><img src="img/dot.jpg" alt="Itrtype" width="800px"></div>

I’ve added matrix dimensions at the bottom of this figure to stress that the two matrices have to have the same dimension on the side they face each other with. You can visualize this operation as looking like this:

<div align="center"><img src="img/dot1.jpg" alt="Itrtype" width="800px"></div>



```python
# A = np.random.randint(0,5,(3,4))
# B = np.random.randint(0,5,(4,2))
A = np.array([1,2,3])
B = np.array([[1,10],[100,1000],[10000,100000]])
print(A)
print()
print(B)
```

    [1 2 3]

    [[     1     10]
     [   100   1000]
     [ 10000 100000]]



```python
# Dot product
A.dot(B)
```




    array([ 30201, 302010])



### Matrix Aggregation

We can aggregate matrices the same way we aggregated vectors:

<div align="center"><img src="img/agg.jpg" alt="Itrtype" width="800px"></div>

Not only can we aggregate all the values in a matrix, but we can also aggregate across the rows or columns by using the `axis` parameter:

<div align="center"><img src="img/agg1.jpg" alt="Itrtype" width="800px"></div>


```python
print("1:------------\n",np.sum(B))
print("2:------------\n",np.sum(B,axis=0))
print("3:------------\n",np.sum(B,axis=1))

print("4:------------\n",np.sqrt(B))
print("5:------------\n",np.mean(B))
print("6:------------\n",np.mean(B,axis=0))
```

    1:------------
     111111
    2:------------
     [ 10101 101010]
    3:------------
     [    11   1100 110000]
    4:------------
     [[  1.           3.16227766]
     [ 10.          31.6227766 ]
     [100.         316.22776602]]
    5:------------
     18518.5
    6:------------
     [ 3367. 33670.]


**NUMPY SUM WITH AXIS = 0**

When we set axis = 0, the function actually sums down the columns.

<div align="center"><img src="img/sum_0.png" alt="dfs"></div>

Remember that axis 0 refers to the vertical direction across the rows. That means that the code np.sum(2d_array, axis = 0) collapses the rows during the summation.So when we set axis = 0, we’re not summing across the rows. Rather we collapse axis 0.

**NUMPY SUM WITH AXIS = 1**

axis 1 refers to the horizontal direction across the columns. That means that the code np.sum(np_array_2d, axis = 1) collapses the columns during the summation.
<div align="center"><img src="img/sum_1.png" alt="dfs"></div>

So Generally
If you do `.sum(axis=n)`, for example, then dimension `n` is collapsed and deleted, with each value in the new matrix equal to the sum of the corresponding collapsed values. For example, if `b` has shape `(5,6,7,8)`, and you do `c = b.sum(axis=2)`, then axis 2 (dimension with size 7) is collapsed, and the result has shape `(5,6,8)`.


<div align="center"><img src="img/sum_all.png" alt="dfs"></div>

Similarly, while executing another Numpy aggregation function `np.mean()`, we can specify what axis we want to calculate the values across.


### Logical Operations


```python
np.all([True, True, False])

```




    False




```python
np.any([True, False, False])

```




    True




```python
#Note: can be used for array comparisions
a = np.zeros((50, 50))
np.any(a != 0)

```




    False




```python
np.all(a == a)

```




    True




```python
a = np.array([1, 2, 3, 2])
b = np.array([2, 2, 3, 2])
c = np.array([6, 4, 4, 5])
((a <= b) & (b <= c)).all()

```




    True



## Statistics


```python
x = np.array([1, 2, 3, 1])
y = np.array([[1, 2, 3], [5, 6, 1]])
x.mean()
```




    1.75




```python
np.median(x)
```




    1.5




```python
np.median(y, axis=-1)  # last axis
```




    array([2., 5.])




```python
x.std()          # full population standard dev.
```




    0.82915619758885



## Miscellaneous Operations

### Unique Items and Count


```python
a= [[1,4,5,2,2,5],
   [4,4,1,7,4,5]]
u_val,count = np.unique(a,return_counts=True)
```


```python
print(u_val)
print(count)
```

    [1 2 4 5 7]
    [2 2 4 3 1]


### Reversing Rows and Columns


```python
a =np.array([[1,2,3],
	[5,6,7]])
```


```python
a[::-1]
```




    array([[5, 6, 7],
           [1, 2, 3]])




```python
a[::-1, ::-1]
```




    array([[7, 6, 5],
           [3, 2, 1]])



## Broadcasting

Broadcasting is a process performed by NumPy that allows mathematical operations to work with objects that don't necessarily have compatible dimensions.

- First rule of Numpy: 2 Array can perform operaiton only when they have same shapes
- `Broadcasting` let two Arrays of different shapes to do some operaitons.
    - The `small` Array will repeat itself, and convert to the same shape as of another array.

<div align="center"><img src="img/Matrix Arithmetic 2.jpg" alt="Itrtype" width="800px"></div>

<div align="center"><img src="img/broadcast.jpg" alt="Itrtype" width="600px"></div>

<div align="center"><img src="img/broadcasting.png" alt="Itrtype" width="600px"></div>


```python
A = np.array([[1,2,1],[2,3,1],[3,4,1]])
a = np.array([[1,2,3]])
```


```python
A + 4
```




    array([[5, 6, 5],
           [6, 7, 5],
           [7, 8, 5]])




```python
A + a
```




    array([[2, 4, 4],
           [3, 5, 4],
           [4, 6, 4]])




```python
a.T
```




    array([[1],
           [2],
           [3]])




```python
A + a.T
```




    array([[2, 3, 2],
           [4, 5, 3],
           [6, 7, 4]])



## Shape Manipulation - Transposing, Reshaping, Stacking etc...


```python
A = np.array([[1,2,3],[4,5,6]])
```




    array([[1, 2, 3],
           [4, 5, 6]])



### Flatten


```python
A.flatten()
```




    array([1, 2, 3, 4, 5, 6])



### Reshape


```python
print(A.reshape(2,3))
print()
print(A.reshape(3,2))
```

    [[1 2 3]
     [4 5 6]]

    [[1 2]
     [3 4]
     [5 6]]


<div align="center"><img src="img/reshape.jpg" alt="dfgdfg" width="800px"></div>

### Transpose

A common need when dealing with matrices is the need to **rotate** them. This is often the case when we need to take the dot product of two matrices and need to align the dimension they share. NumPy arrays have a convenient property called T to get the `transpose` of a matrix:

<div align="center"><img src="img/trans.jpg" alt="dfgdfg" width="800px"></div>


```python
print(A.flatten())
X = A.reshape(2,3)
Y = A.reshape(3,2)
print()
print("X:---------------\n",X)
print()
print("Y:---------------\n",Y)
print()
print("X.T=Y:-----------\n",X.T)
print()
print("Y.T=X:-----------\n",Y.T)
```

    [1 2 3 4 5 6]

    X:---------------
     [[1 2 3]
     [4 5 6]]

    Y:---------------
     [[1 2]
     [3 4]
     [5 6]]

    X.T=Y:-----------
     [[1 4]
     [2 5]
     [3 6]]

    Y.T=X:-----------
     [[1 3 5]
     [2 4 6]]


### Adding a Dimension

Indexing with the np.newaxis object allows us to add an axis to an array

`newaxis` is used to increase the dimension of the existing array by one more dimension, when used once. Thus,

1D array will become 2D array

2D array will become 3D array

3D array will become 4D array and so on



```python
z = np.array([1, 2, 3])
z

```




    array([1, 2, 3])




```python
z[:, np.newaxis]

```




    array([[1],
           [2],
           [3]])



### Stacking of Array

<div align="center"><img src="img/stack.jpg" alt="dfgdfg" width="800px"></div>


```python
a = np.arange(0,5)
b = np.arange(5,10)
print('Array 1 :','\n',a)
print('Array 2 :','\n',b)
print('Vertical stacking :','\n',np.vstack((a,b)))
print('Horizontal stacking :','\n',np.hstack((a,b)))
```

    Array 1 :
     [0 1 2 3 4]
    Array 2 :
     [5 6 7 8 9]
    Vertical stacking :
     [[0 1 2 3 4]
     [5 6 7 8 9]]
    Horizontal stacking :
     [0 1 2 3 4 5 6 7 8 9]


## 🌟Vectorization🌟

- performing operation directly on Arrays

Vectorization is the process of modifying code to utilize array operation methods. Array operations can be computed internally by NumPy using a lower-level language, which leads to many benefits:

- Vectorized code tends to execute much faster than equivalent code that uses loops (such as for-loops and while-loops). Usually a lot faster. Therefore, vectorization can be very important for machine learning, where we often work with large datasets

- Vectorized code can often be more compact. Having fewer lines of code to write can potentially speed-up the code-writing process, make code more readable, and reduce the risk of errors


```python
# find the distance between any two points (x1, y1) and (x2, y2)
p1 = np.array([1,2,3,4])  # [x1,x2,x3.....,xn]
p2 = np.array([5,5,3,4])  # [y1,y2,x3.....,xn]
```


```python
s=0
for i in range(3):
    s += (p2[i] - p1[i])**2

print(s**0.5)
```

    5.0



```python
# efficient
def point_distance(p1,p2):
    return np.sqrt(np.sum((p2-p1)**2))

print(point_distance(p1,p2))
```

    5.0


### Machine Learning context

Let's imagine a machine learning problem where we use a linear regression algorithm to model the cost of electricity.

Let's denote our model features as `x1,x2...xn`. Features could represent things like **the amount of available wind energy**, **the current gas price**, and **the current load on the grid**.

After we train the algorithm, we obtain model parameters, `θ0,θ1,θ2...θn`. These model parameters constitute the _weights_ that should be used for each feature.

For instance, `x2` might represent the price of gas. The model might find that gas prices are particularly decisive in determining the price of electricity. The corresponding weight of `θ2` would then be expected to be much larger in magnitude than other weights for less important features. The result (hypothesis/prediction) returned by our linear regression model for a given set of x is a linear expression:

`h=θo+x1.θ1+x2.θ2+...+xn.θn`

Furthermore, let's assume we have a set of `m` test examples. In other words, we have `m` sets of `x` for which we would like to obtain the model's prediction. The linear expression, `h`, is to be calculated for each of the test examples. There will be a total of `m` individual hypothesis outputs.

First, define a `10x4` array (x) in which each row is a training set. Here, m=10 and n=4:



```python
x = np.arange(1,41).reshape(10,4)
# x is now a range of 40 numbers reshaped to be 10 rows by 4 columns.
print('x:\n', x)
```

    x:
     [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]
     [13 14 15 16]
     [17 18 19 20]
     [21 22 23 24]
     [25 26 27 28]
     [29 30 31 32]
     [33 34 35 36]
     [37 38 39 40]]


Now, add a column of ones to represent `x0`, known in machine learning as the `bias` term. `x` is now a `10x5` array:


```python
# print(np.full((4,1),1)) # uncomment before run
ones = np.full((10,1),1)
x = np.hstack((ones,x))
print('x:\n', x)
# Using np.full, we created a 10x1 array full of ones then horizontally stacked it (np.hstack) to the front of x.
print('shape : \n', x.shape)
print('x.shape[0] : \n', x.shape[0])
print('x.shape[1] : \n', x.shape[1])
```

    [[1]
     [1]
     [1]
     [1]]
    x:
     [[ 1  1  2  3  4]
     [ 1  5  6  7  8]
     [ 1  9 10 11 12]
     [ 1 13 14 15 16]
     [ 1 17 18 19 20]
     [ 1 21 22 23 24]
     [ 1 25 26 27 28]
     [ 1 29 30 31 32]
     [ 1 33 34 35 36]
     [ 1 37 38 39 40]]
    shape :
     (10, 5)
    x.shape[0] :
     10
    x.shape[1] :
     5


Now let's initialize our model parameters as a `5x1` array


```python
theta = np.arange(1,6).reshape(5,1)
print('theta:\n', theta)
```

    theta:
     [[1]
     [2]
     [3]
     [4]
     [5]]


Armed with our matrix `x` and vector `θ`, we'll proceed to define vectorized and non-vectorized versions of evaluating the linear expressions to compare the computation time.


```python
#Non-vectorized version
def non_vectorized_output(x, theta):
    h = []
    for i in range(x.shape[0]):  # number of elements in axis=0
        total = 0
        for j in range(x.shape[1]):  # number of elements in axis=1
            total = total + x[i, j] * theta[j, 0]
            # ∑Xeach_row_all_colum_el.θeach_row`~~`h=θo+x1.θ1+x2.θ2+...+xn.θn`
        h.append(total)
    return h

#Vectorized version
def vectorized_output(x, theta):
    h = np.matmul(x, theta) # NumPy's matrix multiplication function
    return h

```


```python
print(vectorized_output(x,theta))
```

    [[ 41]
     [ 97]
     [153]
     [209]
     [265]
     [321]
     [377]
     [433]
     [489]
     [545]]



```python
nv_time = %timeit -o non_vectorized_output(x, theta)
```

    80.5 µs ± 11 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)



```python
v_time = %timeit -o vectorized_output(x, theta)
```

    4.62 µs ± 280 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)



```python
print('Non-vectorized version:', f'{1E6 * nv_time.average:0.2f}', 'microseconds per execution, average')

print('Vectorized version:', f'{1E6 * v_time.average:0.2f}', 'microseconds per execution, average')

print('Computation was', "%.0f" % (nv_time.average / v_time.average), 'times faster using vectorization')
```

    Non-vectorized version: 80.53 microseconds per execution, average
    Vectorized version: 4.62 microseconds per execution, average
    Computation was 17 times faster using vectorization


Note that in both examples, NumPy's vectorized calculations significantly outperformed native Python calculations using loops. The improved performance is substantial.

However, vectorization does have potential disadvantages. Vectorized code can be less intuitive to those who do not know how to read it. It can also be more memory intensive.
