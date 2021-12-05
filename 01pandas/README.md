- [Pandas](#pandas)
  - [`Series` objects](#series-objects)
    - [Creating a `Series`](#creating-a-series)
    - [Similar to a 1D `ndarray`](#similar-to-a-1d-ndarray)
    - [Indexing](#indexing)
      - [Slicing a `Series` also slices the index labels:](#slicing-a-series-also-slices-the-index-labels)
      - [Init from `dict`](#init-from-dict)
    - [Automatic alignment](#automatic-alignment)
    - [Init with a scalar](#init-with-a-scalar)
  - [`DataFrame` objects](#dataframe-objects)
    - [Creating a `DataFrame`](#creating-a-dataframe)
      - [from numpy array](#from-numpy-array)
      - [dictionary of `Series` objects:](#dictionary-of-series-objects)
      - [`DataFrame(columns=[],index=[])` constructor](#dataframecolumnsindex-constructor)
    - [Indexing, Masking, Query](#indexing-masking-query)
      - [Extracting Columns](#extracting-columns)
      - [Extracting Rows](#extracting-rows)
        - [`loc()` - label location](#loc---label-location)
        - [`iloc()` - integer location](#iloc---integer-location)
      - [Masking - Boolean Indexing](#masking---boolean-indexing)
      - [Querying a `DataFrame`](#querying-a-dataframe)
    - [Adding and removing columns](#adding-and-removing-columns)
      - [direct assignment](#direct-assignment)
      - [`insert(position,column,value)`](#insertpositioncolumnvalue)
      - [`assign()`: Assigning new columns](#assign-assigning-new-columns)
      - [`drop` and `pop`](#drop-and-pop)
    - [Handy Methods and Properties](#handy-methods-and-properties)
      - [`shape` , `dtypes` , `info()`, `describe()`](#shape--dtypes--info-describe)
      - [`head` and `tail`](#head-and-tail)
      - [`columns`](#columns)
      - [`values` : returns a numpy array](#values--returns-a-numpy-array)
      - [`unique` and `nunique`](#unique-and-nunique)
      - [`value_counts()`](#value_counts)
      - [Sorting a `DataFrame`](#sorting-a-dataframe)
      - [`apply`](#apply)
    - [Saving & loading files](#saving--loading-files)
      - [Saving](#saving)
      - [Loading](#loading)
    - [Operations on `DataFrame`s](#operations-on-dataframes)
    - [Aggregating with `groupby`](#aggregating-with-groupby)
    - [Handling Missing Data](#handling-missing-data)

# Pandas

- library for Data Analysis and Manipulation

**Why Pandas?**

- provides ability to work with Tabular data
  - `Tabular Data` : data that is organized into tables having rows and cols


```python
import pandas as pd
import numpy as np
# jupyter nbconvert --to markdown pandas.ipynb --output README.md
```

## `Series` objects

- A `Series` object is 1D array that can hold/store data.

### Creating a `Series`


```python
l = ["C", "C++", "Python", "Javascript"]
s = pd.Series(l)
s
```




    0             C
    1           C++
    2        Python
    3    Javascript
    dtype: object



### Similar to a 1D `ndarray`

`Series` objects behave much like one-dimensional NumPy `ndarray`s, and you can often pass them as parameters to NumPy functions:


```python
import numpy as np

s = pd.Series([2,4,6,8])
np.exp(s)
```




    0       7.389056
    1      54.598150
    2     403.428793
    3    2980.957987
    dtype: float64



Arithmetic operations on `Series` are also possible, and they apply *elementwise*, just like for `ndarray`s:


```python
s + [1000,2000,3000,4000]
```




    0    1002
    1    2004
    2    3006
    3    4008
    dtype: int64



Similar to NumPy, if you add a single number to a `Series`, that number is added to all items in the `Series`. This is called * broadcasting*:


```python
s + 1000
```




    0    1002
    1    1004
    2    1006
    3    1008
    dtype: int64



The same is true for all binary operations such as `*` or `/`, and even conditional operations:


```python
s < 0
```




    0    False
    1    False
    2    False
    3    False
    dtype: bool



### Indexing


```python
s2 = pd.Series(l, index=["a", "b", "c", "d"])
s2
```




    a             C
    b           C++
    c        Python
    d    Javascript
    dtype: object



You can then use the `Series` just like a `dict`:


```python
s2["b"]
```




    'C++'



You can still access the items by integer location, like in a regular array:


```python
s2[1]
```




    'C++'



To make it clear when you are accessing by label or by integer location, it is recommended to always use the `loc` attribute when accessing by label, and the `iloc` attribute when accessing by integer location:


```python
s2.loc["b"]
```




    'C++'




```python
s2.iloc[1]
```




    'C++'



#### Slicing a `Series` also slices the index labels:


```python
s2.iloc[1:3]
```




    b       C++
    c    Python
    dtype: object



This can lead to unexpected results when using the default numeric labels, so be careful:


```python
surprise = pd.Series([1000, 1001, 1002, 1003])
surprise
```




    0    1000
    1    1001
    2    1002
    3    1003
    dtype: int64




```python
surprise_slice = surprise[2:]
surprise_slice
```




    2    1002
    3    1003
    dtype: int64



Oh look! The first element has index label `2`. The element with index label `0` is absent from the slice:


```python
try:
    surprise_slice[0]
except KeyError as e:
    print("Key error:", e)
```

    Key error: 0


But remember that you can access elements by integer location using the `iloc` attribute. This illustrates another reason why it's always better to use `loc` and `iloc` to access `Series` objects:


```python
surprise_slice.iloc[0]
```




    1002



#### Init from `dict`

You can create a `Series` object from a `dict`. The keys will be used as index labels:


```python
weights = {"a": 68, "b": 83, "c": 86, "d": 68}
s3 = pd.Series(weights)
s3
```




    a    68
    b    83
    c    86
    d    68
    dtype: int64



You can control which elements you want to include in the `Series` and in what order by explicitly specifying the desired `index`:


```python
s4 = pd.Series(weights, index = ["c", "a"])
s4
```




    c    86
    a    68
    dtype: int64



### Automatic alignment

When an operation involves multiple `Series` objects, `pandas` automatically aligns items by matching index labels.


```python
s2 = pd.Series([1,2,3], index=["a", "b", "c"])
s3 = pd.Series([10,20,40], index=["a", "b", "d"])

print(s2.keys())
print(s3.keys())

s2 + s3

```

    Index(['a', 'b', 'c'], dtype='object')
    Index(['a', 'b', 'd'], dtype='object')





    a    11.0
    b    22.0
    c     NaN
    d     NaN
    dtype: float64



The resulting `Series` contains the union of index labels from `s2` and `s3`. Since `"d"` is missing from `s2` and `"c"` is missing from `s3`, these items have a `NaN` result value. (ie. Not-a-Number means *missing*).

Automatic alignment is very handy when working with data that may come from various sources with varying structure and missing items. But if you forget to set the **right index labels**, you can have surprising results:


```python
s5 = pd.Series([1000,1000,1000,1000])
s2 + s5
```




    a   NaN
    b   NaN
    c   NaN
    0   NaN
    1   NaN
    2   NaN
    3   NaN
    dtype: float64



Pandas could not align the `Series`, since their labels do not match at all, hence the full `NaN` result.

### Init with a scalar

You can also initialize a `Series` object using a scalar and a list of index labels: all items will be set to the scalar.


```python
meaning = pd.Series(42, ["a", "b", "c"])
meaning
```




    a    42
    b    42
    c    42
    dtype: int64




```python
s6 = pd.Series([83, 68], index=["a", "b"], name="weights")
s6
```




    a    83
    b    68
    Name: weights, dtype: int64



## `DataFrame` objects

- A DataFrame object represents a 2d labelled array, with cell values, column names and row index labels
- You can see `DataFrame`s as dictionaries of `Series`.


### Creating a `DataFrame`

#### from numpy array


```python
import numpy as np
```


```python
arr = np.random.randint(10,100,size=(6,4))
arr
```




    array([[30, 27, 82, 14],
           [94, 66, 75, 56],
           [53, 19, 72, 20],
           [32, 91, 10, 14],
           [88, 65, 70, 49],
           [31, 57, 27, 95]])




```python
df = pd.DataFrame(data=arr)
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>27</td>
      <td>82</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>94</td>
      <td>66</td>
      <td>75</td>
      <td>56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>19</td>
      <td>72</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>91</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>88</td>
      <td>65</td>
      <td>70</td>
      <td>49</td>
    </tr>
    <tr>
      <th>5</th>
      <td>31</td>
      <td>57</td>
      <td>27</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>




```python
type(df)
```




    pandas.core.frame.DataFrame




```python
df[2]
```




    0    82
    1    75
    2    72
    3    10
    4    70
    5    27
    Name: 2, dtype: int32




```python
type(df[2])
```




    pandas.core.series.Series




```python
type(df[0])
```




    pandas.core.series.Series



#### dictionary of `Series` objects:


```python
people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children": pd.Series([0, 3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}
people = pd.DataFrame(people_dict)
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>68</td>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>83</td>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>112</td>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



A few things to note:
* the `Series` were automatically aligned based on their index,
* missing values are represented as `NaN`,
* `Series` names are ignored (the name `"year"` was dropped),
* `DataFrame`s are displayed nicely in Jupyter notebooks, woohoo!

You can access columns pretty much as you would expect. They are returned as `Series` objects:


```python
people["birthyear"]
```




    alice      1985
    bob        1984
    charles    1992
    Name: birthyear, dtype: int64



You can also get multiple columns at once:


```python
people[["birthyear", "hobby"]]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>hobby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>Biking</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>Dancing</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



It is also possible to create a `DataFrame` with a dictionary (or list) of dictionaries (or list):


```python
people = pd.DataFrame({
    "birthyear": {"alice": 1985, "bob": 1984, "charles": 1992},
    "hobby": {"alice": "Biking", "bob": "Dancing"},
    "weight": {"alice": 68, "bob": 83, "charles": 112},
    "children": {"bob": 3, "charles": 0}
})
people

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>hobby</th>
      <th>weight</th>
      <th>children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>Biking</td>
      <td>68</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>Dancing</td>
      <td>83</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>NaN</td>
      <td>112</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



#### `DataFrame(columns=[],index=[])` constructor

If you pass a list of columns and/or index row labels to the `DataFrame` constructor, it will guarantee that these columns and/or rows will exist, in that order, and no other column/row will exist. For example:


```python
d2 = pd.DataFrame(
        people_dict,
        columns=["birthyear", "weight", "height"],
        index=["bob", "alice", "eugene"]
     )
d2
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>weight</th>
      <th>height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bob</th>
      <td>1984.0</td>
      <td>83.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>alice</th>
      <td>1985.0</td>
      <td>68.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>eugene</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Another convenient way to create a `DataFrame` is to pass all the values to the constructor as an `ndarray`, or a list of lists, and specify the column names and row index labels separately:


```python
values = [
            [1985, np.nan, "Biking",   68],
            [1984, 3,      "Dancing",  83],
            [1992, 0,      np.nan,    112]
         ]
d3 = pd.DataFrame(
        values,
        columns=["birthyear", "children", "hobby", "weight"],
        index=["alice", "bob", "charles"]
     )
d3
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



To specify missing values, you can either use `np.nan` or NumPy's masked arrays:


```python
masked_array = np.ma.asarray(values, dtype=np.object)
masked_array[(0, 2), (1, 2)] = np.ma.masked
d3 = pd.DataFrame(
        masked_array,
        columns=["birthyear", "children", "hobby", "weight"],
        index=["alice", "bob", "charles"]
     )
d3
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>3</td>
      <td>Dancing</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>0</td>
      <td>NaN</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



Instead of an `ndarray`, you can also pass a `DataFrame` object:


```python
d4 = pd.DataFrame(
         d3,
         columns=["hobby", "children"],
         index=["alice", "bob"]
     )
d4
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



### Indexing, Masking, Query

#### Extracting Columns


```python
arr = np.random.randint(10, 100, size=(6, 4))
df = pd.DataFrame(data=arr,columns=["a", "b", "c", "d"])
# df.columns = ["a", "b", "c", "d"]
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>71</td>
      <td>34</td>
      <td>51</td>
      <td>84</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td>36</td>
      <td>56</td>
      <td>54</td>
    </tr>
    <tr>
      <th>2</th>
      <td>95</td>
      <td>46</td>
      <td>40</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>95</td>
      <td>41</td>
      <td>46</td>
      <td>62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80</td>
      <td>86</td>
      <td>79</td>
      <td>59</td>
    </tr>
    <tr>
      <th>5</th>
      <td>36</td>
      <td>39</td>
      <td>96</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['c']
```




    0    51
    1    56
    2    40
    3    46
    4    79
    5    96
    Name: c, dtype: int32



multiple columns can be extracted at once:


```python
df[['b','c','a']]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>c</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34</td>
      <td>51</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36</td>
      <td>56</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46</td>
      <td>40</td>
      <td>95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41</td>
      <td>46</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>86</td>
      <td>79</td>
      <td>80</td>
    </tr>
    <tr>
      <th>5</th>
      <td>39</td>
      <td>96</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
</div>



#### Extracting Rows


```python
arr = np.random.randint(10, 100, size=(6, 4))
df = pd.DataFrame(data=arr)
df.columns = ["a", "b", "c", "d"]
df.index = "p q r s t u".split()
df

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p</th>
      <td>64</td>
      <td>81</td>
      <td>95</td>
      <td>79</td>
    </tr>
    <tr>
      <th>q</th>
      <td>17</td>
      <td>52</td>
      <td>83</td>
      <td>41</td>
    </tr>
    <tr>
      <th>r</th>
      <td>81</td>
      <td>76</td>
      <td>54</td>
      <td>29</td>
    </tr>
    <tr>
      <th>s</th>
      <td>96</td>
      <td>58</td>
      <td>22</td>
      <td>98</td>
    </tr>
    <tr>
      <th>t</th>
      <td>83</td>
      <td>27</td>
      <td>67</td>
      <td>95</td>
    </tr>
    <tr>
      <th>u</th>
      <td>16</td>
      <td>34</td>
      <td>60</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



##### `loc()` - label location

The `loc` attribute lets you access rows instead of columns. The result is a `Series` object in which the `DataFrame`'s column names are mapped to row index labels:


```python
df.loc["p"]
```




    a    64
    b    81
    c    95
    d    79
    Name: p, dtype: int32



##### `iloc()` - integer location

You can also access rows by integer location using the `iloc` attribute:


```python
df.iloc[2]
```




    a    81
    b    76
    c    54
    d    29
    Name: r, dtype: int32



You can also get a slice of rows, and this returns a `DataFrame` object:


```python
df.iloc[1:3]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>q</th>
      <td>17</td>
      <td>52</td>
      <td>83</td>
      <td>41</td>
    </tr>
    <tr>
      <th>r</th>
      <td>81</td>
      <td>76</td>
      <td>54</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[1:3][['a','b']]

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>q</th>
      <td>17</td>
      <td>52</td>
    </tr>
    <tr>
      <th>r</th>
      <td>81</td>
      <td>76</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[1:3,:2]

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>q</th>
      <td>17</td>
      <td>52</td>
    </tr>
    <tr>
      <th>r</th>
      <td>81</td>
      <td>76</td>
    </tr>
  </tbody>
</table>
</div>



#### Masking - Boolean Indexing


```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p</th>
      <td>64</td>
      <td>81</td>
      <td>95</td>
      <td>79</td>
    </tr>
    <tr>
      <th>q</th>
      <td>17</td>
      <td>52</td>
      <td>83</td>
      <td>41</td>
    </tr>
    <tr>
      <th>r</th>
      <td>81</td>
      <td>76</td>
      <td>54</td>
      <td>29</td>
    </tr>
    <tr>
      <th>s</th>
      <td>96</td>
      <td>58</td>
      <td>22</td>
      <td>98</td>
    </tr>
    <tr>
      <th>t</th>
      <td>83</td>
      <td>27</td>
      <td>67</td>
      <td>95</td>
    </tr>
    <tr>
      <th>u</th>
      <td>16</td>
      <td>34</td>
      <td>60</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
df > 30
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>q</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>r</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>s</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>t</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>u</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
mask = df > 30
df[mask]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p</th>
      <td>64.0</td>
      <td>81.0</td>
      <td>95.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>q</th>
      <td>NaN</td>
      <td>52.0</td>
      <td>83.0</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>r</th>
      <td>81.0</td>
      <td>76.0</td>
      <td>54.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>s</th>
      <td>96.0</td>
      <td>58.0</td>
      <td>NaN</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>t</th>
      <td>83.0</td>
      <td>NaN</td>
      <td>67.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>u</th>
      <td>NaN</td>
      <td>34.0</td>
      <td>60.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



This is most useful when combined with boolean expressions:


```python
df['a'] <50
```




    p    False
    q     True
    r    False
    s    False
    t    False
    u     True
    Name: a, dtype: bool




```python
df[df["a"] < 50] # only getting q and u as both is True
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>q</th>
      <td>17</td>
      <td>52</td>
      <td>83</td>
      <td>41</td>
    </tr>
    <tr>
      <th>u</th>
      <td>16</td>
      <td>34</td>
      <td>60</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df["a"] < 50][['a','d']]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>q</th>
      <td>17</td>
      <td>41</td>
    </tr>
    <tr>
      <th>u</th>
      <td>16</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



#### Querying a `DataFrame`

The `query()` method lets you filter a `DataFrame` based on a query expression:


```python
people.query("age > 30 and pets == 0")
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>height</th>
      <th>weight</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
      <th>body_mass_index</th>
      <th>overweight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>181</td>
      <td>83</td>
      <td>34</td>
      <td>True</td>
      <td>0.0</td>
      <td>25.335002</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Adding and removing columns


```python
people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children": pd.Series([0, 3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}
people = pd.DataFrame(people_dict)
people

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>68</td>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>83</td>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>112</td>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### direct assignment


```python
people["age"] = 2021 - people["birthyear"]  # adds a new column "age"
people["over 30"] = people["age"] > 30      # adds another column "over 30"
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>age</th>
      <th>over 30</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>68</td>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>36</td>
      <td>True</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>83</td>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>37</td>
      <td>True</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>112</td>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>29</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
birthyears
```




    alice      1985
    bob        1984
    charles    1992
    Name: birthyear, dtype: int64



When you add a new colum, it must have the same number of rows. Missing rows are filled with NaN, and extra rows are ignored:


```python
people["pets"] = pd.Series({"bob": 0, "charles": 5, "eugene":1})  # alice is missing, eugene is ignored
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>68</td>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>36</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>83</td>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>37</td>
      <td>True</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>112</td>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>29</td>
      <td>False</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



#### `insert(position,column,value)`

When adding a new column, it is added at the end (on the right) by default. You can also insert a column anywhere else using the `insert()` method:


```python
people.insert(1, "height", [172, 181, 185])
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>height</th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>68</td>
      <td>172</td>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>36</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>83</td>
      <td>181</td>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>37</td>
      <td>True</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>112</td>
      <td>185</td>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>29</td>
      <td>False</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



#### `assign()`: Assigning new columns

You can also create new columns by calling the `assign()` method. Note that this returns a new `DataFrame` object, **the original is not modified:**


```python
people.assign(
    body_mass_index = people["weight"] / (people["height"] / 100) ** 2,
    has_pets = people["pets"] > 0
)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>height</th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
      <th>body_mass_index</th>
      <th>has_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>68</td>
      <td>172</td>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>36</td>
      <td>True</td>
      <td>NaN</td>
      <td>22.985398</td>
      <td>False</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>83</td>
      <td>181</td>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>37</td>
      <td>True</td>
      <td>0.0</td>
      <td>25.335002</td>
      <td>False</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>112</td>
      <td>185</td>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>29</td>
      <td>False</td>
      <td>5.0</td>
      <td>32.724617</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
people # the original is not modified
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>height</th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>68</td>
      <td>172</td>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>36</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>83</td>
      <td>181</td>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>37</td>
      <td>True</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>112</td>
      <td>185</td>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>29</td>
      <td>False</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



Note that you cannot access columns created within the same assignment:


```python
try:
    people.assign(
        body_mass_index = people["weight"] / (people["height"] / 100) ** 2,
        overweight = people["body_mass_index"] > 25 # body_mass_index is not defined at this point
    )
except KeyError as e:
    print("Key error:", e)
```

    Key error: 'body_mass_index'


The solution is to split this assignment in two consecutive assignments:


```python
d6 = people.assign(body_mass_index = people["weight"] / (people["height"] / 100) ** 2)
d6.assign(overweight = d6["body_mass_index"] > 25)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>height</th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
      <th>body_mass_index</th>
      <th>overweight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>68</td>
      <td>172</td>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>36</td>
      <td>True</td>
      <td>NaN</td>
      <td>22.985398</td>
      <td>False</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>83</td>
      <td>181</td>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>37</td>
      <td>True</td>
      <td>0.0</td>
      <td>25.335002</td>
      <td>True</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>112</td>
      <td>185</td>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>29</td>
      <td>False</td>
      <td>5.0</td>
      <td>32.724617</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Having to create a temporary variable `d6` is not very convenient. You may want to just chain the assigment calls, but it does not work because the `people` object is not actually modified by the first assignment:


```python
try:
    (people
         .assign(body_mass_index = people["weight"] / (people["height"] / 100) ** 2)
         .assign(overweight = people["body_mass_index"] > 25)
    )
except KeyError as e:
    print("Key error:", e)
```

    Key error: 'body_mass_index'


But fear not, there is a simple solution. You can pass a function to the `assign()` method (typically a `lambda` function), and this function will be called with the `DataFrame` as a parameter:


```python
(people
     .assign(body_mass_index = lambda df: df["weight"] / (df["height"] / 100) ** 2)
     .assign(overweight = lambda df: df["body_mass_index"] > 25)
)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>height</th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
      <th>body_mass_index</th>
      <th>overweight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>68</td>
      <td>172</td>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>36</td>
      <td>True</td>
      <td>NaN</td>
      <td>22.985398</td>
      <td>False</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>83</td>
      <td>181</td>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>37</td>
      <td>True</td>
      <td>0.0</td>
      <td>25.335002</td>
      <td>True</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>112</td>
      <td>185</td>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>29</td>
      <td>False</td>
      <td>5.0</td>
      <td>32.724617</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Problem solved!

#### `drop` and `pop`


```python
arr = np.random.randint(10, 100, size=(4,8))
df = pd.DataFrame(data=arr,columns=["a", "b", "c", "d", "e", "f", "g", "h"])
df['a+b'] = df['a'] + df['b']
df['a-b'] = df['a'] * df['b']
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
      <th>g</th>
      <th>h</th>
      <th>a+b</th>
      <th>a-b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>60</td>
      <td>21</td>
      <td>41</td>
      <td>85</td>
      <td>27</td>
      <td>92</td>
      <td>49</td>
      <td>70</td>
      <td>600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>74</td>
      <td>90</td>
      <td>86</td>
      <td>87</td>
      <td>65</td>
      <td>103</td>
      <td>2580</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55</td>
      <td>62</td>
      <td>23</td>
      <td>58</td>
      <td>83</td>
      <td>80</td>
      <td>90</td>
      <td>72</td>
      <td>117</td>
      <td>3410</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>97</td>
      <td>11</td>
      <td>13</td>
      <td>91</td>
      <td>92</td>
      <td>49</td>
      <td>35</td>
      <td>144</td>
      <td>4559</td>
    </tr>
  </tbody>
</table>
</div>




```python
delC = df.pop('c')  # removes column c
del df["d"] # removes column d
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>e</th>
      <th>f</th>
      <th>g</th>
      <th>h</th>
      <th>a+b</th>
      <th>a-b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>60</td>
      <td>85</td>
      <td>27</td>
      <td>92</td>
      <td>49</td>
      <td>70</td>
      <td>600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>60</td>
      <td>90</td>
      <td>86</td>
      <td>87</td>
      <td>65</td>
      <td>103</td>
      <td>2580</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55</td>
      <td>62</td>
      <td>83</td>
      <td>80</td>
      <td>90</td>
      <td>72</td>
      <td>117</td>
      <td>3410</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>97</td>
      <td>91</td>
      <td>92</td>
      <td>49</td>
      <td>35</td>
      <td>144</td>
      <td>4559</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(columns=['e','f','a-b'])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>g</th>
      <th>h</th>
      <th>a+b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>60</td>
      <td>92</td>
      <td>49</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>60</td>
      <td>87</td>
      <td>65</td>
      <td>103</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55</td>
      <td>62</td>
      <td>90</td>
      <td>72</td>
      <td>117</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>97</td>
      <td>49</td>
      <td>35</td>
      <td>144</td>
    </tr>
  </tbody>
</table>
</div>




```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>e</th>
      <th>f</th>
      <th>g</th>
      <th>h</th>
      <th>a+b</th>
      <th>a-b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>60</td>
      <td>85</td>
      <td>27</td>
      <td>92</td>
      <td>49</td>
      <td>70</td>
      <td>600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>60</td>
      <td>90</td>
      <td>86</td>
      <td>87</td>
      <td>65</td>
      <td>103</td>
      <td>2580</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55</td>
      <td>62</td>
      <td>83</td>
      <td>80</td>
      <td>90</td>
      <td>72</td>
      <td>117</td>
      <td>3410</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>97</td>
      <td>91</td>
      <td>92</td>
      <td>49</td>
      <td>35</td>
      <td>144</td>
      <td>4559</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(columns=['e', 'f', 'a-b'], inplace=True)  # original df is modified

```


```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>g</th>
      <th>h</th>
      <th>a+b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>60</td>
      <td>92</td>
      <td>49</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>60</td>
      <td>87</td>
      <td>65</td>
      <td>103</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55</td>
      <td>62</td>
      <td>90</td>
      <td>72</td>
      <td>117</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>97</td>
      <td>49</td>
      <td>35</td>
      <td>144</td>
    </tr>
  </tbody>
</table>
</div>



### Handy Methods and Properties


```python
arr = np.random.randint(10, 100, size=(6, 4))
df = pd.DataFrame(data=arr)
df.columns = 'a b c d'.split()
df

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>73</td>
      <td>76</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>84</td>
      <td>25</td>
      <td>87</td>
      <td>59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>79</td>
      <td>36</td>
      <td>76</td>
      <td>69</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46</td>
      <td>88</td>
      <td>43</td>
      <td>41</td>
    </tr>
    <tr>
      <th>4</th>
      <td>95</td>
      <td>24</td>
      <td>15</td>
      <td>37</td>
    </tr>
    <tr>
      <th>5</th>
      <td>59</td>
      <td>49</td>
      <td>36</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



#### `shape` , `dtypes` , `info()`, `describe()`


```python
df.shape
```




    (6, 4)




```python
df.dtypes
```




    a    int32
    b    int32
    c    int32
    d    int32
    dtype: object




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6 entries, 0 to 5
    Data columns (total 4 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       6 non-null      int32
     1   b       6 non-null      int32
     2   c       6 non-null      int32
     3   d       6 non-null      int32
    dtypes: int32(4)
    memory usage: 224.0 bytes



```python
df.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>68.666667</td>
      <td>49.166667</td>
      <td>55.500000</td>
      <td>46.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>20.146133</td>
      <td>26.331856</td>
      <td>28.317839</td>
      <td>16.334014</td>
    </tr>
    <tr>
      <th>min</th>
      <td>46.000000</td>
      <td>24.000000</td>
      <td>15.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>51.500000</td>
      <td>27.750000</td>
      <td>37.750000</td>
      <td>38.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>69.000000</td>
      <td>42.500000</td>
      <td>59.500000</td>
      <td>44.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>82.750000</td>
      <td>67.000000</td>
      <td>76.000000</td>
      <td>56.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>95.000000</td>
      <td>88.000000</td>
      <td>87.000000</td>
      <td>69.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe().T
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>6.0</td>
      <td>68.666667</td>
      <td>20.146133</td>
      <td>46.0</td>
      <td>51.50</td>
      <td>69.0</td>
      <td>82.75</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>6.0</td>
      <td>49.166667</td>
      <td>26.331856</td>
      <td>24.0</td>
      <td>27.75</td>
      <td>42.5</td>
      <td>67.00</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>6.0</td>
      <td>55.500000</td>
      <td>28.317839</td>
      <td>15.0</td>
      <td>37.75</td>
      <td>59.5</td>
      <td>76.00</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>6.0</td>
      <td>46.000000</td>
      <td>16.334014</td>
      <td>23.0</td>
      <td>38.00</td>
      <td>44.0</td>
      <td>56.00</td>
      <td>69.0</td>
    </tr>
  </tbody>
</table>
</div>



#### `head` and `tail`

- `head`: prints the first 5 rows
- `tail`: prints the last 5 rows


```python
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80</td>
      <td>11</td>
      <td>76</td>
      <td>81</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51</td>
      <td>29</td>
      <td>24</td>
      <td>59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64</td>
      <td>64</td>
      <td>26</td>
      <td>62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>29</td>
      <td>78</td>
      <td>97</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>72</td>
      <td>77</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head(n=3)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80</td>
      <td>11</td>
      <td>76</td>
      <td>81</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51</td>
      <td>29</td>
      <td>24</td>
      <td>59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64</td>
      <td>64</td>
      <td>26</td>
      <td>62</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(n=2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>72</td>
      <td>77</td>
      <td>45</td>
    </tr>
    <tr>
      <th>5</th>
      <td>83</td>
      <td>55</td>
      <td>26</td>
      <td>71</td>
    </tr>
  </tbody>
</table>
</div>



#### `columns`


```python
df.columns
```




    Index(['a', 'b', 'c', 'd'], dtype='object')



#### `values` : returns a numpy array


```python
arr = df.values
arr
```




    array([[96, 71, 41, 34],
           [43, 92, 89, 79],
           [20, 28, 78, 42],
           [51, 82, 67, 30],
           [13, 27, 72, 79],
           [85, 12, 62, 14]])



#### `unique` and `nunique`

The Pandas Unique technique identifies the unique values of a Pandas Series.


```python
people_dict = {
    "country": pd.Series(['BD','IN','PAK','BD','BD','IN']),
	"name":pd.Series(['A','B','C','D','E','F'])
}
people = pd.DataFrame(people_dict)
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BD</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IN</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PAK</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BD</td>
      <td>D</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BD</td>
      <td>E</td>
    </tr>
    <tr>
      <th>5</th>
      <td>IN</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
</div>




```python
people.nunique()
```




    country    3
    name       6
    dtype: int64




```python
people['country'].nunique()

```




    3




```python
people['country'].unique()

```




    array(['BD', 'IN', 'PAK'], dtype=object)



#### `value_counts()`

count occupance of each unique element


```python
people['country'].value_counts()
```




    BD     3
    IN     2
    PAK    1
    Name: country, dtype: int64




```python
people['country'].value_counts()['BD']
```




    3



#### Sorting a `DataFrame`
You can sort a `DataFrame` by calling its `sort_index` method. By default it sorts the rows by their **index label**, in ascending order, but let's reverse the order:


```python
people_dict = {
    "country": pd.Series(['BD', 'IN', 'PAK', 'SL', 'US', 'IN']),
   	"name": pd.Series(['A', 'B', 'C', 'D', 'E', 'F']),
	   "cgpa":pd.Series([3.56, 4.00, 3.55, 3.86, 3.99, 3.89])
}
people = pd.DataFrame(people_dict)
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>name</th>
      <th>cgpa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BD</td>
      <td>A</td>
      <td>3.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IN</td>
      <td>B</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PAK</td>
      <td>C</td>
      <td>3.55</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SL</td>
      <td>D</td>
      <td>3.86</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>E</td>
      <td>3.99</td>
    </tr>
    <tr>
      <th>5</th>
      <td>IN</td>
      <td>F</td>
      <td>3.89</td>
    </tr>
  </tbody>
</table>
</div>




```python
people.sort_index(ascending=False).head(n=3)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>name</th>
      <th>cgpa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>IN</td>
      <td>F</td>
      <td>3.89</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>E</td>
      <td>3.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SL</td>
      <td>D</td>
      <td>3.86</td>
    </tr>
  </tbody>
</table>
</div>



Note that `sort_index` returned a sorted *copy* of the `DataFrame`. To modify `people` directly, we can set the `inplace` argument to `True`. Also, we can sort the columns instead of the rows by setting `axis=1`:


```python
people.sort_index(axis=1,ascending=False, inplace=True)
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>country</th>
      <th>cgpa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>BD</td>
      <td>3.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>IN</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>PAK</td>
      <td>3.55</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>SL</td>
      <td>3.86</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E</td>
      <td>US</td>
      <td>3.99</td>
    </tr>
    <tr>
      <th>5</th>
      <td>F</td>
      <td>IN</td>
      <td>3.89</td>
    </tr>
  </tbody>
</table>
</div>



To sort the `DataFrame` by the values instead of the labels, we can use `sort_values` and specify the column to sort by:


```python
people.sort_values(by=["name"], ascending=False,inplace=True)
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>country</th>
      <th>cgpa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>F</td>
      <td>IN</td>
      <td>3.89</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E</td>
      <td>US</td>
      <td>3.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>SL</td>
      <td>3.86</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>PAK</td>
      <td>3.55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>IN</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>BD</td>
      <td>3.56</td>
    </tr>
  </tbody>
</table>
</div>




```python
people.sort_values(by=["cgpa", "name"])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>country</th>
      <th>cgpa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>PAK</td>
      <td>3.55</td>
    </tr>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>BD</td>
      <td>3.56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>SL</td>
      <td>3.86</td>
    </tr>
    <tr>
      <th>5</th>
      <td>F</td>
      <td>IN</td>
      <td>3.89</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E</td>
      <td>US</td>
      <td>3.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>IN</td>
      <td>4.00</td>
    </tr>
  </tbody>
</table>
</div>



#### `apply`

Pandas.apply allow the users to pass a function and apply it on every single value of the Pandas series. It comes as a huge improvement for the pandas library as this function helps to segregate data according to the conditions required due to which it is efficiently used in data science and machine learning.


```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>74</td>
      <td>75</td>
      <td>90</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>95</td>
      <td>57</td>
      <td>33</td>
      <td>56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23</td>
      <td>27</td>
      <td>14</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64</td>
      <td>88</td>
      <td>45</td>
      <td>81</td>
    </tr>
    <tr>
      <th>4</th>
      <td>72</td>
      <td>17</td>
      <td>59</td>
      <td>65</td>
    </tr>
    <tr>
      <th>5</th>
      <td>87</td>
      <td>83</td>
      <td>98</td>
      <td>91</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.apply(lambda x: x+x)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>148</td>
      <td>150</td>
      <td>180</td>
      <td>126</td>
    </tr>
    <tr>
      <th>1</th>
      <td>190</td>
      <td>114</td>
      <td>66</td>
      <td>112</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46</td>
      <td>54</td>
      <td>28</td>
      <td>48</td>
    </tr>
    <tr>
      <th>3</th>
      <td>128</td>
      <td>176</td>
      <td>90</td>
      <td>162</td>
    </tr>
    <tr>
      <th>4</th>
      <td>144</td>
      <td>34</td>
      <td>118</td>
      <td>130</td>
    </tr>
    <tr>
      <th>5</th>
      <td>174</td>
      <td>166</td>
      <td>196</td>
      <td>182</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['a'].apply(lambda x:x*10)
```




    0    740
    1    950
    2    230
    3    640
    4    720
    5    870
    Name: a, dtype: int64



### Saving & loading files

Pandas can save `DataFrame`s to various backends, including file formats such as CSV, Excel, JSON, HTML and HDF5, or to a SQL database. Let's create a `DataFrame` to demonstrate this:


```python
my_df = pd.DataFrame(
    [["Biking", 68.5, 1985, np.nan], ["Dancing", 83.1, 1984, 3]],
    columns=["hobby","weight","birthyear","children"],
    index=["alice", "bob"]
)
my_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>weight</th>
      <th>birthyear</th>
      <th>children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>68.5</td>
      <td>1985</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>83.1</td>
      <td>1984</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Saving
Let's save it to CSV, HTML and JSON:


```python
my_df.to_csv("my_df.csv")
# my_df.to_csv("my_df.csv",index=False)
my_df.to_html("my_df.html")
my_df.to_json("my_df.json")
```

#### Loading
Now let's load our CSV file back into a `DataFrame`:


```python
my_df_loaded = pd.read_csv("my_df.csv")
my_df_loaded
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>hobby</th>
      <th>weight</th>
      <th>birthyear</th>
      <th>children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alice</td>
      <td>Biking</td>
      <td>68.5</td>
      <td>1985</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bob</td>
      <td>Dancing</td>
      <td>83.1</td>
      <td>1984</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
my_df_loaded = pd.read_csv("my_df.csv",index_col=0)
my_df_loaded

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>weight</th>
      <th>birthyear</th>
      <th>children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>68.5</td>
      <td>1985</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>83.1</td>
      <td>1984</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



As you might guess, there are similar `read_json`, `read_html`, `read_excel` functions as well.  We can also read data straight from the Internet. For example, let's load the top 1,000 U.S. cities from github:


```python
us_cities = None
try:
    csv_url = "https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv"
    us_cities = pd.read_csv(csv_url, index_col=0)
    us_cities = us_cities.head()
except IOError as e:
    print(e)
us_cities
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Population</th>
      <th>lat</th>
      <th>lon</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Marysville</th>
      <td>Washington</td>
      <td>63269</td>
      <td>48.051764</td>
      <td>-122.177082</td>
    </tr>
    <tr>
      <th>Perris</th>
      <td>California</td>
      <td>72326</td>
      <td>33.782519</td>
      <td>-117.228648</td>
    </tr>
    <tr>
      <th>Cleveland</th>
      <td>Ohio</td>
      <td>390113</td>
      <td>41.499320</td>
      <td>-81.694361</td>
    </tr>
    <tr>
      <th>Worcester</th>
      <td>Massachusetts</td>
      <td>182544</td>
      <td>42.262593</td>
      <td>-71.802293</td>
    </tr>
    <tr>
      <th>Columbia</th>
      <td>South Carolina</td>
      <td>133358</td>
      <td>34.000710</td>
      <td>-81.034814</td>
    </tr>
  </tbody>
</table>
</div>



### Operations on `DataFrame`s
Although `DataFrame`s do not try to mimick NumPy arrays, there are a few similarities. Let's create a `DataFrame` to demonstrate this:


```python
grades_array = np.array([[8,8,9],[10,9,9],[4, 8, 2], [9, 10, 10]])
grades = pd.DataFrame(grades_array,
	columns=["sep", "oct", "nov"],
	index=["alice","bob","charles","darwin"])
grades
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>8</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>10</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>4</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>9</td>
      <td>10</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



You can apply NumPy mathematical functions on a `DataFrame`: the function is applied to all values:


```python
np.sqrt(grades)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>2.828427</td>
      <td>2.828427</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>3.162278</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>2.000000</td>
      <td>2.828427</td>
      <td>1.414214</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>3.000000</td>
      <td>3.162278</td>
      <td>3.162278</td>
    </tr>
  </tbody>
</table>
</div>



Similarly, adding a single value to a `DataFrame` will add that value to all elements in the `DataFrame`. This is called *broadcasting*:


```python
grades + 1
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>9</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>11</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>5</td>
      <td>9</td>
      <td>3</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>10</td>
      <td>11</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



Of course, the same is true for all other binary operations, including arithmetic (`*`,`/`,`**`...) and conditional (`>`, `==`...) operations:


```python
grades >= 5
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Aggregation operations, such as computing the `max`, the `sum` or the `mean` of a `DataFrame`, apply to each column, and you get back a `Series` object:


```python
grades.mean()
```




    sep    7.75
    oct    8.75
    nov    7.50
    dtype: float64



The `all` method is also an aggregation operation: it checks whether all values are `True` or not. Let's see during which months all students got a grade greater than `5`:


```python
(grades > 5).all()
```




    sep    False
    oct     True
    nov    False
    dtype: bool



Most of these functions take an optional `axis` parameter which lets you specify along which axis of the `DataFrame` you want the operation executed. The default is `axis=0`, meaning that the operation is executed vertically (on each column). You can set `axis=1` to execute the operation horizontally (on each row). For example, let's find out which students had all grades greater than `5`:


```python
(grades > 5).all(axis = 1)
```




    alice       True
    bob         True
    charles    False
    darwin      True
    dtype: bool



The `any` method returns `True` if any value is True. Let's see who got at least one grade 10:


```python
(grades == 10).any(axis = 1)
```




    alice      False
    bob         True
    charles    False
    darwin      True
    dtype: bool



If you add a `Series` object to a `DataFrame` (or execute any other binary operation), pandas attempts to broadcast the operation to all *rows* in the `DataFrame`. This only works if the `Series` has the same size as the `DataFrame`s rows. For example, let's subtract the `mean` of the `DataFrame` (a `Series` object) from the `DataFrame`:


```python
grades - grades.mean()  # equivalent to: grades - [7.75, 8.75, 7.50]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>0.25</td>
      <td>-0.75</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>2.25</td>
      <td>0.25</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>-3.75</td>
      <td>-0.75</td>
      <td>-5.5</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>1.25</td>
      <td>1.25</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>



We subtracted `7.75` from all September grades, `8.75` from October grades and `7.50` from November grades. It is equivalent to subtracting this `DataFrame`:


```python
pd.DataFrame([[7.75, 8.75, 7.50]]*4, index=grades.index, columns=grades.columns)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>7.75</td>
      <td>8.75</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>7.75</td>
      <td>8.75</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>7.75</td>
      <td>8.75</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>7.75</td>
      <td>8.75</td>
      <td>7.5</td>
    </tr>
  </tbody>
</table>
</div>



If you want to subtract the global mean from every grade, here is one way to do it:


```python
grades - grades.values.mean() # subtracts the global mean (8.00) from all grades
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>-4.0</td>
      <td>0.0</td>
      <td>-6.0</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



### Aggregating with `groupby`

Similar to the SQL language, pandas allows grouping your data into groups to run calculations over each group.

First, let's add some extra data about each person so we can group them, and let's go back to the `final_grades` `DataFrame` so we can see how `NaN` values are handled:


```python
iris = pd.read_csv("iris.csv")
iris.head()

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.aggregate('min')
```




    sepal_length       4.3
    sepal_width        2.0
    petal_length       1.0
    petal_width        0.1
    species         setosa
    dtype: object




```python
iris.aggregate(['min','max','mean','median'])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000</td>
      <td>1.000000</td>
      <td>0.100000</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400</td>
      <td>6.900000</td>
      <td>2.500000</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.054</td>
      <td>3.758667</td>
      <td>1.198667</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>median</th>
      <td>5.800000</td>
      <td>3.000</td>
      <td>4.350000</td>
      <td>1.300000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
groupby = iris.groupby('species')
groupby
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001EC566AC820>




```python
groupby.min()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
    <tr>
      <th>species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>setosa</th>
      <td>4.3</td>
      <td>2.3</td>
      <td>1.0</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>4.9</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>4.9</td>
      <td>2.2</td>
      <td>4.5</td>
      <td>1.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
groupby.mean()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
    <tr>
      <th>species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>setosa</th>
      <td>5.006</td>
      <td>3.418</td>
      <td>1.464</td>
      <td>0.244</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>5.936</td>
      <td>2.770</td>
      <td>4.260</td>
      <td>1.326</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>6.588</td>
      <td>2.974</td>
      <td>5.552</td>
      <td>2.026</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris[iris['species'] == 'setosa']['sepal_length'].mean()
```




    5.005999999999999



### Handling Missing Data

- `dropna`
- `fillna`


```python
iris.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
nan_idx = np.random.randint(0,150,20)
iris['sepal_length'][nan_idx] = np.nan
```

    <ipython-input-82-8496a87f94dc>:2: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      iris['sepal_length'][nan_idx] = np.nan



```python
import warnings
warnings.filterwarnings('ignore')
```


```python
nan_idx = np.random.randint(0, 150, 20)
iris['sepal_length'][nan_idx] = np.nan
```


```python
iris['sepal_length'][:20]
```




    0     5.1
    1     4.9
    2     4.7
    3     4.6
    4     5.0
    5     5.4
    6     4.6
    7     5.0
    8     4.4
    9     4.9
    10    NaN
    11    NaN
    12    4.8
    13    4.3
    14    NaN
    15    5.7
    16    NaN
    17    NaN
    18    NaN
    19    5.1
    Name: sepal_length, dtype: float64




```python
nan_idx = np.random.randint(0, 150, 20)
iris['petal_width'][nan_idx] = np.nan
```


```python
iris.isna().sum()
```




    sepal_length    63
    sepal_width      0
    petal_length     0
    petal_width     20
    species          0
    dtype: int64




```python
iris.dropna()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>138</th>
      <td>6.0</td>
      <td>3.0</td>
      <td>4.8</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>141</th>
      <td>6.9</td>
      <td>3.1</td>
      <td>5.1</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>144</th>
      <td>6.7</td>
      <td>3.3</td>
      <td>5.7</td>
      <td>2.5</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 5 columns</p>
</div>




```python
iris['sepal_length'].fillna(value="FILLED")[:20]
```




    0        5.1
    1        4.9
    2        4.7
    3        4.6
    4        5.0
    5        5.4
    6        4.6
    7        5.0
    8        4.4
    9        4.9
    10    FILLED
    11    FILLED
    12       4.8
    13       4.3
    14    FILLED
    15       5.7
    16    FILLED
    17    FILLED
    18    FILLED
    19       5.1
    Name: sepal_length, dtype: object




```python
iris['sepal_length'] = iris['sepal_length'].fillna(value=round(iris['sepal_length'].mean(),1))
```


```python
iris['sepal_length'][:20]
```




    0     5.1
    1     4.9
    2     4.7
    3     4.6
    4     5.0
    5     5.4
    6     4.6
    7     5.0
    8     4.4
    9     4.9
    10    5.8
    11    5.8
    12    4.8
    13    4.3
    14    5.8
    15    5.7
    16    5.8
    17    5.8
    18    5.8
    19    5.1
    Name: sepal_length, dtype: float64


