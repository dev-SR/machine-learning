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
      - [dictionary of `pd.Series` or `List` objects:](#dictionary-of-pdseries-or-list-objects)
      - [`DataFrame(columns=[],index=[])` constructor](#dataframecolumnsindex-constructor)
    - [Indexing, Masking, Query](#indexing-masking-query)
      - [Extracting Columns](#extracting-columns)
      - [Extracting Rows](#extracting-rows)
        - [`loc()` - label location](#loc---label-location)
        - [`iloc()` - integer location](#iloc---integer-location)
      - [Extracting Rows with Columns using slice](#extracting-rows-with-columns-using-slice)
      - [Masking - Boolean Indexing](#masking---boolean-indexing)
      - [Querying a `DataFrame`](#querying-a-dataframe)
    - [Adding and Removing Row/Columns](#adding-and-removing-rowcolumns)
      - [Adding Column](#adding-column)
        - [direct assignment](#direct-assignment)
        - [`insert(position,column,value)`](#insertpositioncolumnvalue)
        - [`assign()`: Assigning new columns](#assign-assigning-new-columns)
      - [Adding Row](#adding-row)
      - [Removing Rows/Columns](#removing-rowscolumns)
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
      - [`isnull()` and `notnull()`](#isnull-and-notnull)
      - [`fillna`](#fillna)
      - [`dropna`](#dropna)
    - [Handling String Data - Converting Reality to Numbers](#handling-string-data---converting-reality-to-numbers)

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



#### dictionary of `pd.Series` or `List` objects:


```python
data = {
	'roll_no': [10,3,2,4],
	'sid':[111,112,113,114],
	'marks':[90,80,70,60]
}
df = pd.DataFrame(data)
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roll_no</th>
      <th>sid</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>111</td>
      <td>90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>112</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>113</td>
      <td>70</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>114</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>




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
      <td>31</td>
      <td>86</td>
      <td>98</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31</td>
      <td>13</td>
      <td>43</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>24</td>
      <td>81</td>
      <td>85</td>
    </tr>
    <tr>
      <th>3</th>
      <td>57</td>
      <td>77</td>
      <td>83</td>
      <td>99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>13</td>
      <td>97</td>
      <td>49</td>
    </tr>
    <tr>
      <th>5</th>
      <td>91</td>
      <td>87</td>
      <td>36</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['c']
```




    0    98
    1    43
    2    81
    3    83
    4    97
    5    36
    Name: c, dtype: int32




```python
df.c
```




    0    98
    1    43
    2    81
    3    83
    4    97
    5    36
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
      <td>86</td>
      <td>98</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>43</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24</td>
      <td>81</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77</td>
      <td>83</td>
      <td>57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>97</td>
      <td>22</td>
    </tr>
    <tr>
      <th>5</th>
      <td>87</td>
      <td>36</td>
      <td>91</td>
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
      <td>88</td>
      <td>61</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>q</th>
      <td>11</td>
      <td>68</td>
      <td>61</td>
      <td>92</td>
    </tr>
    <tr>
      <th>r</th>
      <td>91</td>
      <td>54</td>
      <td>20</td>
      <td>58</td>
    </tr>
    <tr>
      <th>s</th>
      <td>63</td>
      <td>63</td>
      <td>82</td>
      <td>41</td>
    </tr>
    <tr>
      <th>t</th>
      <td>82</td>
      <td>41</td>
      <td>76</td>
      <td>98</td>
    </tr>
    <tr>
      <th>u</th>
      <td>73</td>
      <td>52</td>
      <td>58</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



##### `loc()` - label location

The `loc` attribute lets you access rows instead of columns. The result is a `Series` object in which the `DataFrame`'s column names are mapped to row index labels:


```python
df.loc["p"]
```




    a    88
    b    61
    c    34
    d    34
    Name: p, dtype: int32



##### `iloc()` - integer location

You can also access rows by integer location using the `iloc` attribute:


```python
df.iloc[2]
```




    a    91
    b    54
    c    20
    d    58
    Name: r, dtype: int32



#### Extracting Rows with Columns using slice


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
      <td>60</td>
      <td>53</td>
      <td>17</td>
      <td>23</td>
    </tr>
    <tr>
      <th>q</th>
      <td>62</td>
      <td>68</td>
      <td>79</td>
      <td>49</td>
    </tr>
    <tr>
      <th>r</th>
      <td>52</td>
      <td>99</td>
      <td>19</td>
      <td>47</td>
    </tr>
    <tr>
      <th>s</th>
      <td>13</td>
      <td>51</td>
      <td>64</td>
      <td>31</td>
    </tr>
    <tr>
      <th>t</th>
      <td>98</td>
      <td>79</td>
      <td>80</td>
      <td>21</td>
    </tr>
    <tr>
      <th>u</th>
      <td>61</td>
      <td>41</td>
      <td>81</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>




```python
# df.iloc[1:3]
df.iloc[1:3,:]
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
      <td>62</td>
      <td>68</td>
      <td>79</td>
      <td>49</td>
    </tr>
    <tr>
      <th>r</th>
      <td>52</td>
      <td>99</td>
      <td>19</td>
      <td>47</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[1:3,0:2]
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
      <td>62</td>
      <td>68</td>
    </tr>
    <tr>
      <th>r</th>
      <td>52</td>
      <td>99</td>
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
      <td>62</td>
      <td>68</td>
    </tr>
    <tr>
      <th>r</th>
      <td>52</td>
      <td>99</td>
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
df['a'] <50 # d.a < 50
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



### Adding and Removing Row/Columns

####  Adding Column


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



##### direct assignment


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



##### `insert(position,column,value)`

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



##### `assign()`: Assigning new columns

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

#### Adding Row


```python
arr= np.random.randint(10, 100, size=(6,4))
df = pd.DataFrame(data=arr,columns=["a", "b", "c", "d"])
df.index = "p q r s t u".split()
```


```python
print(df.loc['p'])
print(df.iloc[0])
```

    a    81
    b    86
    c    65
    d    23
    Name: p, dtype: int32
    a    81
    b    86
    c    65
    d    23
    Name: p, dtype: int32



```python
df.loc['x'] = [1,2,3,4]
df.tail()
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
      <th>r</th>
      <td>79</td>
      <td>12</td>
      <td>25</td>
      <td>51</td>
    </tr>
    <tr>
      <th>s</th>
      <td>10</td>
      <td>99</td>
      <td>54</td>
      <td>73</td>
    </tr>
    <tr>
      <th>t</th>
      <td>98</td>
      <td>55</td>
      <td>14</td>
      <td>90</td>
    </tr>
    <tr>
      <th>u</th>
      <td>61</td>
      <td>62</td>
      <td>63</td>
      <td>55</td>
    </tr>
    <tr>
      <th>x</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



#### Removing Rows/Columns

- `drop` method:
  - `drop(columns,axis=1)`
  - `drop(index,axis=0)`
  - `drop(labels,axis=1)`
  - `drop(labels,axis=0)`
-  `pop`


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
      <td>53</td>
      <td>98</td>
      <td>47</td>
      <td>62</td>
      <td>69</td>
      <td>88</td>
      <td>48</td>
      <td>22</td>
      <td>151</td>
      <td>5194</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58</td>
      <td>11</td>
      <td>46</td>
      <td>71</td>
      <td>46</td>
      <td>26</td>
      <td>61</td>
      <td>49</td>
      <td>69</td>
      <td>638</td>
    </tr>
    <tr>
      <th>2</th>
      <td>85</td>
      <td>97</td>
      <td>64</td>
      <td>76</td>
      <td>38</td>
      <td>56</td>
      <td>67</td>
      <td>94</td>
      <td>182</td>
      <td>8245</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>51</td>
      <td>29</td>
      <td>23</td>
      <td>60</td>
      <td>36</td>
      <td>46</td>
      <td>28</td>
      <td>104</td>
      <td>2703</td>
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
      <td>53</td>
      <td>98</td>
      <td>69</td>
      <td>88</td>
      <td>48</td>
      <td>22</td>
      <td>151</td>
      <td>5194</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58</td>
      <td>11</td>
      <td>46</td>
      <td>26</td>
      <td>61</td>
      <td>49</td>
      <td>69</td>
      <td>638</td>
    </tr>
    <tr>
      <th>2</th>
      <td>85</td>
      <td>97</td>
      <td>38</td>
      <td>56</td>
      <td>67</td>
      <td>94</td>
      <td>182</td>
      <td>8245</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>51</td>
      <td>60</td>
      <td>36</td>
      <td>46</td>
      <td>28</td>
      <td>104</td>
      <td>2703</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(columns=['e','f','a-b']) # removes columns e, f, a-b
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
      <td>53</td>
      <td>98</td>
      <td>48</td>
      <td>22</td>
      <td>151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58</td>
      <td>11</td>
      <td>61</td>
      <td>49</td>
      <td>69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>85</td>
      <td>97</td>
      <td>67</td>
      <td>94</td>
      <td>182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>51</td>
      <td>46</td>
      <td>28</td>
      <td>104</td>
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
      <td>53</td>
      <td>98</td>
      <td>69</td>
      <td>88</td>
      <td>48</td>
      <td>22</td>
      <td>151</td>
      <td>5194</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58</td>
      <td>11</td>
      <td>46</td>
      <td>26</td>
      <td>61</td>
      <td>49</td>
      <td>69</td>
      <td>638</td>
    </tr>
    <tr>
      <th>2</th>
      <td>85</td>
      <td>97</td>
      <td>38</td>
      <td>56</td>
      <td>67</td>
      <td>94</td>
      <td>182</td>
      <td>8245</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>51</td>
      <td>60</td>
      <td>36</td>
      <td>46</td>
      <td>28</td>
      <td>104</td>
      <td>2703</td>
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
      <td>53</td>
      <td>98</td>
      <td>48</td>
      <td>22</td>
      <td>151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58</td>
      <td>11</td>
      <td>61</td>
      <td>49</td>
      <td>69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>85</td>
      <td>97</td>
      <td>67</td>
      <td>94</td>
      <td>182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>51</td>
      <td>46</td>
      <td>28</td>
      <td>104</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(1,inplace=True) # removes row 1
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
      <td>53</td>
      <td>98</td>
      <td>48</td>
      <td>22</td>
      <td>151</td>
    </tr>
    <tr>
      <th>2</th>
      <td>85</td>
      <td>97</td>
      <td>67</td>
      <td>94</td>
      <td>182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>51</td>
      <td>46</td>
      <td>28</td>
      <td>104</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index
```




    Int64Index([0, 2, 3], dtype='int64')




```python
df.drop(df.index[[0,2]])

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
      <th>2</th>
      <td>85</td>
      <td>97</td>
      <td>67</td>
      <td>94</td>
      <td>182</td>
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
      <td>20</td>
      <td>34</td>
      <td>45</td>
      <td>36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>37</td>
      <td>11</td>
      <td>65</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36</td>
      <td>17</td>
      <td>68</td>
      <td>48</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77</td>
      <td>69</td>
      <td>99</td>
      <td>46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>83</td>
      <td>28</td>
      <td>34</td>
      <td>67</td>
    </tr>
    <tr>
      <th>5</th>
      <td>46</td>
      <td>24</td>
      <td>28</td>
      <td>11</td>
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
      <td>47.333333</td>
      <td>34.833333</td>
      <td>47.500000</td>
      <td>45.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>27.097355</td>
      <td>18.192489</td>
      <td>31.538865</td>
      <td>20.637345</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>17.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.500000</td>
      <td>25.000000</td>
      <td>29.500000</td>
      <td>38.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>41.000000</td>
      <td>31.000000</td>
      <td>39.500000</td>
      <td>47.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>69.250000</td>
      <td>36.250000</td>
      <td>62.250000</td>
      <td>60.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>83.000000</td>
      <td>69.000000</td>
      <td>99.000000</td>
      <td>67.000000</td>
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
      <td>47.333333</td>
      <td>27.097355</td>
      <td>20.0</td>
      <td>25.5</td>
      <td>41.0</td>
      <td>69.25</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>6.0</td>
      <td>34.833333</td>
      <td>18.192489</td>
      <td>17.0</td>
      <td>25.0</td>
      <td>31.0</td>
      <td>36.25</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>6.0</td>
      <td>47.500000</td>
      <td>31.538865</td>
      <td>11.0</td>
      <td>29.5</td>
      <td>39.5</td>
      <td>62.25</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>6.0</td>
      <td>45.500000</td>
      <td>20.637345</td>
      <td>11.0</td>
      <td>38.5</td>
      <td>47.0</td>
      <td>60.75</td>
      <td>67.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['a'].describe()
```




    count     6.000000
    mean     47.333333
    std      27.097355
    min      20.000000
    25%      25.500000
    50%      41.000000
    75%      69.250000
    max      83.000000
    Name: a, dtype: float64



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



More in [Handling String Data - Converting Reality to Numbers](#handling-string-data---converting-reality-to-numbers)


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
data = {
	'roll_no': np.random.randint(1, 100, size=5),
	'ppr_id':np.random.randint(1000, 2000, size=5),
	'marks':np.random.randint(50,100,size=5)
}
df = pd.DataFrame(data)
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roll_no</th>
      <th>ppr_id</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>1690</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>1700</td>
      <td>87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>1364</td>
      <td>55</td>
    </tr>
    <tr>
      <th>3</th>
      <td>94</td>
      <td>1372</td>
      <td>90</td>
    </tr>
    <tr>
      <th>4</th>
      <td>44</td>
      <td>1291</td>
      <td>76</td>
    </tr>
  </tbody>
</table>
</div>




```python
nan_idx = np.random.randint(0,5,3)
df['marks'][nan_idx] = np.nan
# df['marks'][[1,4,3]] = np.nan
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roll_no</th>
      <th>ppr_id</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>1690</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>1700</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>1364</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>94</td>
      <td>1372</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>44</td>
      <td>1291</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### `isnull()` and `notnull()`


```python
df.isnull()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roll_no</th>
      <th>ppr_id</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    roll_no    0
    ppr_id     0
    marks      3
    dtype: int64




```python
df['marks'].isnull()
```




    0    False
    1     True
    2    False
    3     True
    4     True
    Name: marks, dtype: bool



#### `fillna`


```python
df.fillna("FILLED")
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roll_no</th>
      <th>ppr_id</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>1690</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>1700</td>
      <td>FILLED</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>1364</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>94</td>
      <td>1372</td>
      <td>FILLED</td>
    </tr>
    <tr>
      <th>4</th>
      <td>44</td>
      <td>1291</td>
      <td>FILLED</td>
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
      <th>roll_no</th>
      <th>ppr_id</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>1690</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>1700</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>1364</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>94</td>
      <td>1372</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>44</td>
      <td>1291</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.fillna("FILLED")
df.fillna("FILLED",inplace=True)
df

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roll_no</th>
      <th>ppr_id</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>1690</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>1700</td>
      <td>FILLED</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>1364</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>94</td>
      <td>1372</td>
      <td>FILLED</td>
    </tr>
    <tr>
      <th>4</th>
      <td>44</td>
      <td>1291</td>
      <td>FILLED</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = {
	'roll_no': np.random.randint(1, 100, size=5),
	'ppr_id': np.random.randint(1000, 2000, size=5),
	'marks': np.random.randint(50, 100, size=5)
}
df = pd.DataFrame(data)
nan_idx = np.random.randint(0, 5, 3)
df['marks'][nan_idx] = np.nan
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roll_no</th>
      <th>ppr_id</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60</td>
      <td>1764</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58</td>
      <td>1443</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23</td>
      <td>1603</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>1626</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
      <td>1656</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.marks.fillna(df.marks.mean(), inplace=True)
```


```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roll_no</th>
      <th>ppr_id</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60</td>
      <td>1764</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58</td>
      <td>1443</td>
      <td>74.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23</td>
      <td>1603</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>1626</td>
      <td>74.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
      <td>1656</td>
      <td>74.5</td>
    </tr>
  </tbody>
</table>
</div>



#### `dropna`


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
data = {
	'roll_no': np.random.randint(1, 100, size=5),
	'ppr_id': np.random.randint(1000, 2000, size=5),
	'marks': np.random.randint(50, 100, size=5)
}
df = pd.DataFrame(data)
df['marks'][[0, 2, 4]] = np.nan
df['roll_no'][[0, 2]] = np.nan
df

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roll_no</th>
      <th>ppr_id</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>1074</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>1867</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>1103</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>90.0</td>
      <td>1699</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70.0</td>
      <td>1372</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roll_no</th>
      <th>ppr_id</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>1867</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>90.0</td>
      <td>1699</td>
      <td>54.0</td>
    </tr>
  </tbody>
</table>
</div>



### Handling String Data - Converting Reality to Numbers


```python
people_dict = {
	# panda series from a list of A to D
	"name":pd.Series(['A','B','C','D']),
	"gender":pd.Series(['Female','Male','Female','Female'])
}
people = pd.DataFrame(people_dict)
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>




```python
def f(g):
	if g == 'Male':
		return 0
	else:
		return 1


people['sex'] = people.gender.apply(f)
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>gender</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>Female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>Male</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>Female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>Female</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


