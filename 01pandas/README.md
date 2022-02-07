# Pandas

- [Pandas](#pandas)
  - [Introduction](#introduction)
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
    - [From Numpy Array](#from-numpy-array)
    - [Using `dictionary` of `List`, `pd.Series`, `np.Array`:](#using-dictionary-of-list-pdseries-nparray)
    - [`DataFrame(columns=[],index=[])` constructor](#dataframecolumnsindex-constructor)
  - [Indexing, Masking, Query](#indexing-masking-query)
    - [🚀Extracting Columns - Native accessors: `df[col]`, `df[[col1,col2,..]]`](#extracting-columns---native-accessors-dfcol-dfcol1col2)
    - [🚀Index-based selection - `iloc[row_indexer,col_indexer]`](#index-based-selection---ilocrow_indexercol_indexer)
    - [🚀Label-based selection - `loc[row_indexer,col_indexer]`](#label-based-selection---locrow_indexercol_indexer)
      - [Choosing between loc and iloc](#choosing-between-loc-and-iloc)
    - [🚀🚀 Split Input and Output Features and convert to NumPy arrays](#-split-input-and-output-features-and-convert-to-numpy-arrays)
    - [🚀🚀Masking - Boolean Indexing](#masking---boolean-indexing)
      - [`isin`](#isin)
      - [`isnull`](#isnull)
    - [Querying a `DataFrame`](#querying-a-dataframe)
  - [Summary Functions and Maps](#summary-functions-and-maps)
    - [`shape` , `dtypes` , `info()`, `describe()`](#shape--dtypes--info-describe)
    - [`head` and `tail`](#head-and-tail)
    - [`columns`](#columns)
    - [`unique` and `nunique`](#unique-and-nunique)
    - [`value_counts()`](#value_counts)
    - [Maps](#maps)
  - [Edit Whole Row/Columns](#edit-whole-rowcolumns)
    - [Adding Column](#adding-column)
        - [direct assignment](#direct-assignment)
        - [`insert(position,column,value)`](#insertpositioncolumnvalue)
        - [`assign()`: Assigning new columns](#assign-assigning-new-columns)
    - [Adding Row](#adding-row)
    - [Combine Dataframes](#combine-dataframes)
        - [`concat()`](#concat)
      - [`join()`](#join)
    - [Removing Rows/Columns](#removing-rowscolumns)
      - [`drop()`](#drop)
      - [Conditional Drop](#conditional-drop)
    - [Renaming Columns](#renaming-columns)
    - [👉Shuffle a DataFrame rows](#shuffle-a-dataframe-rows)
      - [Using `pd.sample()`](#using-pdsample)
      - [Using `sklearn.utils.shuffle()`](#using-sklearnutilsshuffle)
  - [Data Types and Missing Values](#data-types-and-missing-values)
    - [`dtypes`, `astype()`](#dtypes-astype)
    - [Missing data](#missing-data)
      - [`isnull()` and `notnull()`](#isnull-and-notnull)
      - [`fillna`](#fillna)
      - [`dropna`](#dropna)
  - [Saving & loading files](#saving--loading-files)
    - [Saving](#saving)
    - [Loading](#loading)
    - [Minimize the size of Large DataSet](#minimize-the-size-of-large-dataset)
  - [Operations on `DataFrame`s](#operations-on-dataframes)
  - [Grouping and Sorting](#grouping-and-sorting)
    - [Groupwise analysis](#groupwise-analysis)
    - [Sorting](#sorting)
    - [More example:](#more-example)
  - [Categorical encoding](#categorical-encoding)
    - [Introduction](#introduction-1)
    - [Label Encoding](#label-encoding)
      - [Custom Map Function](#custom-map-function)
      - [Using Pandas.factorize()](#using-pandasfactorize)
        - [Reverse the process - Decoding](#reverse-the-process---decoding)
      - [Using  🌟`sklearn.LabelEncoder()`🌟](#using--sklearnlabelencoder)
        - [Decoding](#decoding)
    - [One-Hot-Encoding](#one-hot-encoding)
        - [Using `Pandas.get_dummies()`](#using-pandasget_dummies)
      - [Using 🌟`sklearn.OneHotEncoder()`🌟](#using-sklearnonehotencoder)

## Introduction

- library for Data Analysis and Manipulation

**Why Pandas?**

- provides ability to work with Tabular data
  - `Tabular Data` : data that is organized into tables having rows and cols


```python
"""
cd .\01pandas\
jupyter nbconvert --to markdown pandas.ipynb --output README.md

 """
import pandas as pd
import numpy as np
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

A `DataFrame` is a table. It contains an array of individual entries, each of which has a certain value. Each entry corresponds to a row (or record) and a column.

- A DataFrame object represents a 2d labelled array, with cell values, column names and row index labels
- You can see `DataFrame`s as dictionaries of `Series`.



<div align="center">
<img src="img/anatomy.png" alt="anatomy.jpg" width="1000px">
</div>

## Creating a `DataFrame`

### From Numpy Array


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
      <td>19</td>
      <td>51</td>
      <td>72</td>
      <td>11</td>
    </tr>
    <tr>
      <th>q</th>
      <td>92</td>
      <td>26</td>
      <td>88</td>
      <td>15</td>
    </tr>
    <tr>
      <th>r</th>
      <td>68</td>
      <td>10</td>
      <td>90</td>
      <td>14</td>
    </tr>
    <tr>
      <th>s</th>
      <td>46</td>
      <td>61</td>
      <td>37</td>
      <td>41</td>
    </tr>
    <tr>
      <th>t</th>
      <td>12</td>
      <td>78</td>
      <td>48</td>
      <td>93</td>
    </tr>
    <tr>
      <th>u</th>
      <td>29</td>
      <td>28</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.random.seed(5)
arr=np.random.randint(100, size=(5, 5))
df = pd.DataFrame(arr,
				columns=list("ABCDE"),
                index=["R" + str(i) for i in range(5)])
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R0</th>
      <td>99</td>
      <td>78</td>
      <td>61</td>
      <td>16</td>
      <td>73</td>
    </tr>
    <tr>
      <th>R1</th>
      <td>8</td>
      <td>62</td>
      <td>27</td>
      <td>30</td>
      <td>80</td>
    </tr>
    <tr>
      <th>R2</th>
      <td>7</td>
      <td>76</td>
      <td>15</td>
      <td>53</td>
      <td>80</td>
    </tr>
    <tr>
      <th>R3</th>
      <td>27</td>
      <td>44</td>
      <td>77</td>
      <td>75</td>
      <td>65</td>
    </tr>
    <tr>
      <th>R4</th>
      <td>47</td>
      <td>30</td>
      <td>84</td>
      <td>86</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



For more see [`DataFrame(columns=[],index=[])` constructor](#dataframecolumnsindex-constructor)


### Using `dictionary` of `List`, `pd.Series`, `np.Array`:

The syntax for declaring a new one is a `dictionary` whose `keys` are the `column` names (`col1`, `col2`, `col3` ..in this example), and whose **values are a `list` of entries**. This is the standard way of constructing a new DataFrame, and the one you are most likely to encounter.


```python
df = pd.DataFrame({
	'col1': [10, 3, 2, 4],
	'col2': [111, 112, 113, 114],
	'col3': [90, 80, 70, 60]
})
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



The dictionary-list constructor assigns values to the column labels, **but just uses an ascending count from 0 (0, 1, 2, 3, ...) for the row labels**. Sometimes this is OK, but oftentimes we will want to assign these labels ourselves.

The list of row labels used in a DataFrame is known as an `Index`. We can assign values to it by using an index parameter in our constructor:


```python
df = pd.DataFrame({
	'col1': [10, 3, 2, 4],
	'col2': [111, 112, 113, 114],
	'col3': [90, 80, 70, 60]
},index=["row1", "row2", "row3", "row4"])
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row1</th>
      <td>10</td>
      <td>111</td>
      <td>90</td>
    </tr>
    <tr>
      <th>row2</th>
      <td>3</td>
      <td>112</td>
      <td>80</td>
    </tr>
    <tr>
      <th>row3</th>
      <td>2</td>
      <td>113</td>
      <td>70</td>
    </tr>
    <tr>
      <th>row4</th>
      <td>4</td>
      <td>114</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_data = {
	"MarksA": np.random.randint(1,100,5),
	"MarksB": np.random.randint(50,100,5),
	"MarksC": np.random.randint(1,100,5)
}
df = pd.DataFrame(user_data)
df.head(n=3)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MarksA</th>
      <th>MarksB</th>
      <th>MarksC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>55</td>
      <td>97</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>92</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>97</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.DataFrame({
	"id": np.arange(10),
	'b': np.random.normal(size=10),
	"c": pd.Series(np.random.choice(["cat", 'dog', "hippo"], replace=True, size=10))
})
df.head()

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-0.736681</td>
      <td>dog</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.284158</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.213199</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-2.400537</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.112093</td>
      <td>hippo</td>
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



For more see [Indexing, Masking, Query](#indexing-masking-query)

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



### `DataFrame(columns=[],index=[])` constructor

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



## Indexing, Masking, Query

### 🚀Extracting Columns - Native accessors: `df[col]`, `df[[col1,col2,..]]`


```python
np.random.seed(10)
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
      <td>19</td>
      <td>25</td>
      <td>74</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99</td>
      <td>39</td>
      <td>18</td>
      <td>83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>50</td>
      <td>46</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>64</td>
      <td>98</td>
      <td>72</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43</td>
      <td>82</td>
      <td>88</td>
      <td>59</td>
    </tr>
    <tr>
      <th>5</th>
      <td>61</td>
      <td>64</td>
      <td>87</td>
      <td>79</td>
    </tr>
  </tbody>
</table>
</div>



In Python, we can access the property of an object by accessing it as an attribute. A `book` object, for example, might have a `title` property, which we can access by calling` book.title`. `Columns` in a pandas DataFrame work in much the same way.


```python
df.c
# If column name has spaces, this will not work
```




    0    74
    1    18
    2    46
    3    98
    4    88
    5    87
    Name: c, dtype: int32



If we have a Python dictionary, we can access its values using the indexing (`[]`) operator. We can do the same with `columns` in a DataFrame:


```python
df['c']
```




    0    74
    1    18
    2    46
    3    98
    4    88
    5    87
    Name: c, dtype: int32



Indexing operator `[]` does have the advantage that it can handle `column` names with **reserved characters** in them (e.g. if we had a `country providence` column, `reviews.country providence` wouldn't work).

Doesn't a pandas Series look kind of like a fancy dictionary? It pretty much is, so it's no surprise that, to drill down to a **single specific value**, we need only use the indexing operator `[]` once more:


```python
df['c'][0]
```




    74



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
      <td>25</td>
      <td>74</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39</td>
      <td>18</td>
      <td>99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50</td>
      <td>46</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64</td>
      <td>98</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>82</td>
      <td>88</td>
      <td>43</td>
    </tr>
    <tr>
      <th>5</th>
      <td>64</td>
      <td>87</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>



### 🚀Index-based selection - `iloc[row_indexer,col_indexer]`



```python
np.random.seed(10)
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
      <td>19</td>
      <td>25</td>
      <td>74</td>
      <td>38</td>
    </tr>
    <tr>
      <th>q</th>
      <td>99</td>
      <td>39</td>
      <td>18</td>
      <td>83</td>
    </tr>
    <tr>
      <th>r</th>
      <td>10</td>
      <td>50</td>
      <td>46</td>
      <td>26</td>
    </tr>
    <tr>
      <th>s</th>
      <td>21</td>
      <td>64</td>
      <td>98</td>
      <td>72</td>
    </tr>
    <tr>
      <th>t</th>
      <td>43</td>
      <td>82</td>
      <td>88</td>
      <td>59</td>
    </tr>
    <tr>
      <th>u</th>
      <td>61</td>
      <td>64</td>
      <td>87</td>
      <td>79</td>
    </tr>
  </tbody>
</table>
</div>



Pandas indexing works in one of two paradigms. The first is **index-based selection**: ***selecting data based on its numerical position in the data***. `iloc` follows this paradigm.


```python
first_row = df.iloc[0]
first_row
```




    a    19
    b    25
    c    74
    d    38
    Name: p, dtype: int32



Both `loc` and `iloc` are `row-first, column-second`. This is the opposite of what we do in native `Python`, which is `column-first, row-second`.

This means that it's marginally easier to retrieve `rows`, and marginally harder to get retrieve `columns`. To get a column with `iloc`, we can do the following:


```python
df.iloc[:, 0] # all rows, first column
```




    p    19
    q    99
    r    10
    s    21
    t    43
    u    61
    Name: a, dtype: int32



On its own, the `:` operator, which also comes from native Python, means `"everything"`. When combined with other selectors, however, it can be used to indicate a range of values. For example, to select the country column from just the first, second, and third row, we would do:

Or, to select just the second and third entries, we would do:


```python
df.iloc[1:3, 0] # second and third row, first column
```




    q    99
    r    10
    Name: a, dtype: int32



It's also possible to pass a list:


```python
df.iloc[[0, 1, 2], 0] # first three rows, first column
```




    p    19
    q    99
    r    10
    Name: a, dtype: int32




```python
df.iloc[:, 0:3] # all rows, first three columns
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p</th>
      <td>19</td>
      <td>25</td>
      <td>74</td>
    </tr>
    <tr>
      <th>q</th>
      <td>99</td>
      <td>39</td>
      <td>18</td>
    </tr>
    <tr>
      <th>r</th>
      <td>10</td>
      <td>50</td>
      <td>46</td>
    </tr>
    <tr>
      <th>s</th>
      <td>21</td>
      <td>64</td>
      <td>98</td>
    </tr>
    <tr>
      <th>t</th>
      <td>43</td>
      <td>82</td>
      <td>88</td>
    </tr>
    <tr>
      <th>u</th>
      <td>61</td>
      <td>64</td>
      <td>87</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[:2,:3] # first two rows, first three columns
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p</th>
      <td>19</td>
      <td>25</td>
      <td>74</td>
    </tr>
    <tr>
      <th>q</th>
      <td>99</td>
      <td>39</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[:2, [1,3]] # first two rows, second and fourth columns
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p</th>
      <td>25</td>
      <td>38</td>
    </tr>
    <tr>
      <th>q</th>
      <td>39</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[1,3]
```




    83




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
      <td>99</td>
      <td>39</td>
    </tr>
    <tr>
      <th>r</th>
      <td>10</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[1:3,[df.columns.get_loc(v) for v in ['a','b']]]
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
      <td>99</td>
      <td>39</td>
    </tr>
    <tr>
      <th>r</th>
      <td>10</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



### 🚀Label-based selection - `loc[row_indexer,col_indexer]`

The second paradigm for attribute selection is the one followed by the `loc` operator: **label-based selection**. In this paradigm, it's the **data index value**, **not its position**, which matters.


```python
x = df.loc["p"]
print(type(x))
x
```

    <class 'pandas.core.series.Series'>





    a    19
    b    25
    c    74
    d    38
    Name: p, dtype: int32



Accessing a single row with list of labels returns a `DataFrame` object:


```python
x1= df.loc[["p"]]
print(type(x1))
x1
```

    <class 'pandas.core.frame.DataFrame'>





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
      <td>19</td>
      <td>25</td>
      <td>74</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>



`iloc` is conceptually simpler than loc because it ignores the dataset's indices. When we use iloc we treat the dataset like a big matrix (a list of lists), one that we have to index into by position. `loc`, by contrast, uses the information in the indices to do its work. Since your dataset usually has meaningful indices, it's usually easier to do things using loc instead.


```python
df.loc[["p","u"]]
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
      <td>19</td>
      <td>25</td>
      <td>74</td>
      <td>38</td>
    </tr>
    <tr>
      <th>u</th>
      <td>61</td>
      <td>64</td>
      <td>87</td>
      <td>79</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc["p","a"]
```




    19




```python
df.loc[["p","u"],["a"]]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p</th>
      <td>19</td>
    </tr>
    <tr>
      <th>u</th>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>



#### Choosing between loc and iloc

When choosing or transitioning between `loc` and `iloc`, there is one "gotcha" worth keeping in mind, which is that the two methods use slightly different indexing schemes.

iloc uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. So 0:10 will select entries 0,...,9. loc, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10.

Why the change? Remember that `loc` can **index any stdlib type:** `strings`, for example. If we have a `DataFrame` with `index` values `Apples,...,Potatoes, ...`, and we want to select *"all the alphabetical fruit choices between Apples and Potatoes"*, then it's a lot more convenient to index `df.loc['Apples':'Potatoes']` than it is to index something like `df.loc['Apples', 'Potatoet']` (t coming after s in the alphabet).

This is particularly **confusing when the DataFrame index is a simple numerical list**, e.g. `0,...,1000`. In this case `df.iloc[0:1000]` will return `1000` entries, while `df.loc[0:1000]` return `1001` of them! To get `1000` elements using `loc`, you will need to go one lower and ask for `df.loc[0:999]`.

Otherwise, the semantics of using `loc` are the same as those for `iloc`.

cols = ['country', 'variety']
df = reviews.loc[:99, cols]

equivalent to:

cols_idx = [0, 11]
df = reviews.iloc[:100, cols_idx]

### 🚀🚀 Split Input and Output Features and convert to NumPy arrays



```python
data = pd.read_csv("weight-height-min.csv")
data.head()

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>73.847017</td>
      <td>241.893563</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>68.781904</td>
      <td>162.310473</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>74.110105</td>
      <td>212.740856</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>71.730978</td>
      <td>220.042470</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>69.881796</td>
      <td>206.349801</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = data['Height']
print("Type of X:", type(X))
print("Shape of X:", X.shape)

X = data[['Height']]
print("Type of X:", type(X))
print("Shape of X:", X.shape)

X = data['Height'].values
print("Type of X:", type(X))
print("Shape of X:", X.shape)

X = data['Height'].values.reshape(-1,1)
print("Type of X:", type(X))
print("Shape of X:", X.shape)

```

    Type of X: <class 'pandas.core.series.Series'>
    Shape of X: (100,)
    Type of X: <class 'pandas.core.frame.DataFrame'>
    Shape of X: (100, 1)
    Type of X: <class 'numpy.ndarray'>
    Shape of X: (100,)
    Type of X: <class 'numpy.ndarray'>
    Shape of X: (100, 1)



```python
X = data.iloc[:, 1].values
print("Shape of X:", X.shape)
X =  X.reshape(-1, 1)
y = data.iloc[:, 2].values
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("Type of X:", type(X))
print("Type of y:", type(y))

```

    Shape of X: (100,)
    Shape of X: (100, 1)
    Shape of y: (100,)
    Type of X: <class 'numpy.ndarray'>
    Type of y: <class 'numpy.ndarray'>


### 🚀🚀Masking - Boolean Indexing


```python
np.random.seed(5)
df = pd.DataFrame(np.random.randint(100, size=(5, 5)), columns = list("ABCDE"),
                  index = ["R" + str(i) for i in range(5)])
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R0</th>
      <td>99</td>
      <td>78</td>
      <td>61</td>
      <td>16</td>
      <td>73</td>
    </tr>
    <tr>
      <th>R1</th>
      <td>8</td>
      <td>62</td>
      <td>27</td>
      <td>30</td>
      <td>80</td>
    </tr>
    <tr>
      <th>R2</th>
      <td>7</td>
      <td>76</td>
      <td>15</td>
      <td>53</td>
      <td>80</td>
    </tr>
    <tr>
      <th>R3</th>
      <td>27</td>
      <td>44</td>
      <td>77</td>
      <td>75</td>
      <td>65</td>
    </tr>
    <tr>
      <th>R4</th>
      <td>47</td>
      <td>30</td>
      <td>84</td>
      <td>86</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df > 50
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R0</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>R1</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>R2</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>R3</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>R4</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Example Dataset: [wine-reviews-dataset](https://www.kaggle.com/zynicide/wine-reviews)


```python
reviews = pd.read_csv('winemag-data-130k-v2-mod.csv',index_col=0)
reviews.head(n=2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
  </tbody>
</table>
</div>



We can start by checking if each wine is Italian or not: `country == 'Italy'`:


```python
reviews.country == 'Italy'
```




    0      True
    1     False
    2     False
    3     False
    4     False
          ...
    95    False
    96    False
    97    False
    98     True
    99    False
    Name: country, Length: 100, dtype: bool



This operation produced a Series of `True/False` booleans based on the `country` of each `record`.

To select **All Rows** where `country == 'Italy'`:


```python
res = reviews[reviews.country == 'Italy']
res.head(n=2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Italy</td>
      <td>Here's a bright, informal red that opens with ...</td>
      <td>Belsito</td>
      <td>87</td>
      <td>16.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Vittoria</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Terre di Giurfo 2013 Belsito Frappato (Vittoria)</td>
      <td>Frappato</td>
      <td>Terre di Giurfo</td>
    </tr>
  </tbody>
</table>
</div>



This result can then be used inside of `loc` to select the relevant data:


```python
res = reviews.loc[reviews.country == 'Italy']
res.head(n=2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Italy</td>
      <td>Here's a bright, informal red that opens with ...</td>
      <td>Belsito</td>
      <td>87</td>
      <td>16.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Vittoria</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Terre di Giurfo 2013 Belsito Frappato (Vittoria)</td>
      <td>Frappato</td>
      <td>Terre di Giurfo</td>
    </tr>
  </tbody>
</table>
</div>




```python
mask = reviews.country == 'Italy'
cols = ['country', 'points', 'taster_name']
res = reviews.loc[mask, cols]
res.head(n=2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>points</th>
      <th>taster_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>87</td>
      <td>Kerin O’Keefe</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Italy</td>
      <td>87</td>
      <td>Kerin O’Keefe</td>
    </tr>
  </tbody>
</table>
</div>



Suppose we'll buy any wine that's made in `Italy` **or** which is rated above average. For this we use a `pipe` (`|`). For `and` -> `&`:


```python
res = reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90), cols]
res.head(n=2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>points</th>
      <th>taster_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>87</td>
      <td>Kerin O’Keefe</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Italy</td>
      <td>87</td>
      <td>Kerin O’Keefe</td>
    </tr>
  </tbody>
</table>
</div>



I'm an economical wine buyer. Which wine is the "best bargain"? Create a variable `bargain_wine` with the title of the wine with the highest points-to-price ratio in the dataset.


```python
bargain_idx = (reviews.points / reviews.price).idxmax()
print(bargain_idx)
bargain_wine = reviews.loc[bargain_idx, 'title']
bargain_wine
```

    42





    'Henry Fessy 2012 Nouveau  (Beaujolais)'



#### `isin`

`isin` is lets you select data whose value `"is in"`**a list of values**. For example, here's how we can use it to select wines only from `Italy` or `France`:




```python
res = reviews.loc[reviews.country.isin(['Italy', 'France']),cols]
res.head(n=3)

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>points</th>
      <th>taster_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>87</td>
      <td>Kerin O’Keefe</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Italy</td>
      <td>87</td>
      <td>Kerin O’Keefe</td>
    </tr>
    <tr>
      <th>7</th>
      <td>France</td>
      <td>87</td>
      <td>Roger Voss</td>
    </tr>
  </tbody>
</table>
</div>



Create a DataFrame `top_oceania_wines` containing all reviews with at least `95 points` (out of 100) for wines from `Italy` or `France`.


```python
top_oceania_wines = reviews.loc[
    (reviews.country.isin(['Italy', 'France']))
    & (reviews.points >= 80)
	,cols
]
top_oceania_wines.head(n=3)

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>points</th>
      <th>taster_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>87</td>
      <td>Kerin O’Keefe</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Italy</td>
      <td>87</td>
      <td>Kerin O’Keefe</td>
    </tr>
    <tr>
      <th>7</th>
      <td>France</td>
      <td>87</td>
      <td>Roger Voss</td>
    </tr>
  </tbody>
</table>
</div>



####  `isnull`

The second is `isnull` (and its companion `notnull`). These methods let you highlight values which are (or are not) e`mpty (`NaN`). For example, to filter out wines lacking a price tag in the dataset, here's what we would do:



```python
res =  reviews.loc[reviews.price.isnull(),['country','price']]
res.head(n=3)

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Italy</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>France</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
res.shape
```




    (8, 2)




```python
res =  reviews.loc[reviews.price.notnull(),['country','price']]
res.shape
```




    (92, 2)



### Querying a `DataFrame`

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



## Summary Functions and Maps

Example Dataset: [wine-reviews-dataset](https://www.kaggle.com/zynicide/wine-reviews)


```python
reviews = pd.read_csv('winemag-data-130k-v2-mod.csv', index_col=0)
reviews.head(n=2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>77</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
  </tbody>
</table>
</div>



### `shape` , `dtypes` , `info()`, `describe()`


```python
reviews.shape

```




    (100, 13)




```python
reviews.dtypes

```




    country                   object
    description               object
    designation               object
    points                     int64
    price                    float64
    province                  object
    region_1                  object
    region_2                  object
    taster_name               object
    taster_twitter_handle     object
    title                     object
    variety                   object
    winery                    object
    dtype: object




```python
reviews.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 100 entries, 0 to 99
    Data columns (total 13 columns):
     #   Column                 Non-Null Count  Dtype
    ---  ------                 --------------  -----
     0   country                100 non-null    object
     1   description            100 non-null    object
     2   designation            70 non-null     object
     3   points                 100 non-null    int64
     4   price                  92 non-null     float64
     5   province               100 non-null    object
     6   region_1               88 non-null     object
     7   region_2               37 non-null     object
     8   taster_name            82 non-null     object
     9   taster_twitter_handle  74 non-null     object
     10  title                  100 non-null    object
     11  variety                100 non-null    object
     12  winery                 100 non-null    object
    dtypes: float64(1), int64(1), object(11)
    memory usage: 10.9+ KB



```python
reviews.describe()

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>points</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000000</td>
      <td>92.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>86.430000</td>
      <td>26.271739</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.794616</td>
      <td>18.320170</td>
    </tr>
    <tr>
      <th>min</th>
      <td>85.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>86.000000</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>86.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>87.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.000000</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews.describe().T

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
      <th>points</th>
      <td>100.0</td>
      <td>86.430000</td>
      <td>0.794616</td>
      <td>85.0</td>
      <td>86.0</td>
      <td>86.0</td>
      <td>87.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>price</th>
      <td>92.0</td>
      <td>26.271739</td>
      <td>18.320170</td>
      <td>9.0</td>
      <td>14.0</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>



This method generates a high-level summary of the attributes of the given `column`. It is type-aware, meaning that its output changes based on the data type of the input. The output above only makes sense for numerical data; for string data here's what we get:


```python
reviews.taster_name.describe()
# reviews['taster_name'].describe()
```




    count             82
    unique            12
    top       Roger Voss
    freq              16
    Name: taster_name, dtype: object



### `head` and `tail`

- `head`: prints the first 5 rows
- `tail`: prints the last 5 rows


```python
reviews.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews.head(2)
reviews.head(n=2)

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews.tail(n=2)

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>98</th>
      <td>Italy</td>
      <td>Forest floor, menthol, espresso, cranberry and...</td>
      <td>Dono Riserva</td>
      <td>88</td>
      <td>30.0</td>
      <td>Tuscany</td>
      <td>Morellino di Scansano</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Serpaia di Endrizzi 2010 Dono Riserva  (Morell...</td>
      <td>Sangiovese</td>
      <td>Serpaia di Endrizzi</td>
    </tr>
    <tr>
      <th>99</th>
      <td>US</td>
      <td>This blends 20% each of all five red-Bordeaux ...</td>
      <td>Intreccio Library Selection</td>
      <td>88</td>
      <td>75.0</td>
      <td>California</td>
      <td>Napa Valley</td>
      <td>Napa</td>
      <td>Virginie Boone</td>
      <td>@vboone</td>
      <td>Soquel Vineyards 2013 Intreccio Library Select...</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Soquel Vineyards</td>
    </tr>
  </tbody>
</table>
</div>



### `columns`


```python
reviews.columns

```




    Index(['country', 'description', 'designation', 'points', 'price', 'province',
           'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'title',
           'variety', 'winery'],
          dtype='object')



### `unique` and `nunique`

The Pandas Unique technique identifies the unique values of a Pandas Series.

Count Not of Unique Values


```python
reviews.nunique()

```




    country                   10
    description              100
    designation               65
    points                     4
    price                     34
    province                  30
    region_1                  56
    region_2                  10
    taster_name               12
    taster_twitter_handle     10
    title                    100
    variety                   44
    winery                    91
    dtype: int64




```python
# reviews['country'].nunique()
reviews.country.nunique()
```




    10




```python
reviews['country'].unique()
```




    array(['Italy', 'Portugal', 'US', 'Spain', 'France', 'Germany',
           'Argentina', 'Chile', 'Australia', 'Austria'], dtype=object)



### `value_counts()`

count occupance of each unique element


```python
reviews['country'].value_counts()
```




    US           43
    Italy        24
    France       14
    Chile         5
    Germany       4
    Spain         3
    Australia     2
    Portugal      2
    Argentina     2
    Austria       1
    Name: country, dtype: int64




```python
reviews['country'].value_counts()['US']
```




    43



### Maps

A `map` is a term, borrowed from mathematics, for a function that takes one set of values and "maps" them to another set of values.

In data science we often have a need for **creating new representations from existing data**, *or* for **transforming data from one format to another**.

`Maps` are what handle this work, making them extremely important for getting your work done! There are two mapping methods that you will use often- `map()` and `apply()`.

map() is the first, and slightly simpler one. For example, suppose that we wanted to remean the scores the wines received to 0. We can do this as follows:



```python
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)
```




    0     0.57
    1     0.57
    2     0.57
    3     0.57
    4     0.57
          ...
    95    1.57
    96    1.57
    97    1.57
    98    1.57
    99    1.57
    Name: points, Length: 100, dtype: float64



The function you pass to `map()` should **expect** a single value from the `Series` (a point value, in the above example), and **return** a **transformed version of that value**. `map()` **returns** a new Series where all the values have been transformed by your function.


```python
res = reviews.head(3)
res[['country', 'points']]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>87</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>87</td>
    </tr>
  </tbody>
</table>
</div>



`apply() ` is the equivalent method if we want to transform a whole DataFrame by calling a custom method on each row.


```python
def remean_points(row):
    row.points = row.points - review_points_mean
    return row


res = reviews.apply(remean_points, axis='columns')
res = res.head(3)
res[['country', 'points']]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>0.57</td>
    </tr>
  </tbody>
</table>
</div>



If we had called `reviews.apply()` with `axis='index'`, then instead of passing a function to transform each row, we would need to give a function to transform each column.

Pandas will also understand what to do if we perform these operations between Series of equal length. For example, an easy way of combining country and region information in the dataset would be to do the following:


```python
reviews.country + " - " + reviews.region_1
```




    0                      Italy - Etna
    1                               NaN
    2            US - Willamette Valley
    3          US - Lake Michigan Shore
    4            US - Willamette Valley
                      ...
    95                France - Juliénas
    96                  France - Régnié
    97                US - Finger Lakes
    98    Italy - Morellino di Scansano
    99                 US - Napa Valley
    Length: 100, dtype: object




```python
reviews.price - reviews.price.mean()
```




    0           NaN
    1    -11.271739
    2    -12.271739
    3    -13.271739
    4     38.728261
            ...
    95    -6.271739
    96    -8.271739
    97    -6.271739
    98     3.728261
    99    48.728261
    Name: price, Length: 100, dtype: float64



These operators are faster than `map()` or `apply()` because they uses speed ups built into pandas. All of the standard Python operators (`>, <, ==`, and so on) work in this manner.

However, they are not as flexible as `map()` or `apply()`, which can do more advanced things, like applying conditional logic, which cannot be done with addition and subtraction alone.

Example:

There are only so many words you can use when describing a bottle of wine. Is a wine more likely to be "tropical" or "fruity"? Create a Series `descriptor_counts` counting how many times each of these two words appears in the `description` column in the dataset. (For simplicity, let's ignore the capitalized versions of these words.)


```python
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])
descriptor_counts

```




    tropical    4
    fruity      8
    dtype: int64



We'd like to host these wine reviews on our website, but a rating system ranging from 80 to 100 points is too hard to understand - we'd like to translate them into simple star ratings. A score of 95 or higher counts as 3 stars, a score of at least 85 but less than 95 is 2 stars. Any other score is 1 star.

Also, the Canadian Vintners Association bought a lot of ads on the site, so any wines from Canada should automatically get 3 stars, regardless of points.

Create a series `star_ratings` with the number of stars corresponding to each review in the dataset.


```python
def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 90:
        return 3
    elif row.points >= 80:
        return 2
    else:
        return 1


star_ratings = reviews.apply(stars, axis='columns')
star_ratings

```




    0     1
    1     2
    2     2
    3     2
    4     3
         ..
    95    2
    96    2
    97    2
    98    2
    99    2
    Length: 100, dtype: int64



## Edit Whole Row/Columns

###  Adding Column


```python
df = pd.DataFrame({
    "a": [1, 2, 3, 4],
    "b": ["Bob", "Alice", "Bob", "Alice"],
})
df
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
      <th>0</th>
      <td>1</td>
      <td>Bob</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Alice</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Bob</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
    </tr>
  </tbody>
</table>
</div>



##### direct assignment


```python
df['c'] = [1, 2, 3, 4]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bob</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Alice</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Bob</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['d'] = 1
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
      <td>1</td>
      <td>Bob</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Alice</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Bob</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['d'] = range(0, len(df))
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bob</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Alice</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Bob</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
      <td>4</td>
      <td>3</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['e'] = df['a'] + df['c']
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bob</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Alice</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Bob</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
      <td>4</td>
      <td>3</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



In Column "B", Find values, where value="Bob" and replace with "0"


```python
find_Bob_in_b = df['b']=='Bob'
find_Bob_in_b
```




    0     True
    1    False
    2     True
    3    False
    Name: b, dtype: bool




```python
df.loc[find_Bob_in_b,'b'] = 'FOUND'
# Not df.loc[find_Bob_in_b] = 'FOUND' => it will replace all Row
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
      <td>1</td>
      <td>FOUND</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Alice</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>FOUND</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Store the result in a new column


```python
df.loc[find_Bob_in_b,'f'] = 'FOUND'
# Not df.loc[find_Bob_in_b] = 'FOUND' => it will replace all Row
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
      <th>f</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>FOUND</td>
      <td>1</td>
      <td>1</td>
      <td>FOUND</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Alice</td>
      <td>2</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>FOUND</td>
      <td>3</td>
      <td>1</td>
      <td>FOUND</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
      <td>4</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



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

### Adding Row


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



### Combine Dataframes

[https://www.datacamp.com/community/tutorials/joining-dataframes-pandas](https://www.datacamp.com/community/tutorials/joining-dataframes-pandas)

##### `concat()`


```python
df1 = pd.DataFrame({'id': ['A01', 'A02', 'A03', 'A04'],
                    'Name': ['ABC', 'PQR', 'DEF', 'GHI']})
df2 = pd.DataFrame({'id': ['B05', 'B06', 'B07', 'B08'],
                    'Name': ['XYZ', 'TUV', 'MNO', 'JKL']})
frames = [df1, df2]
pd.concat(frames)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A01</td>
      <td>ABC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A02</td>
      <td>PQR</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A03</td>
      <td>DEF</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A04</td>
      <td>GHI</td>
    </tr>
    <tr>
      <th>0</th>
      <td>B05</td>
      <td>XYZ</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B06</td>
      <td>TUV</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B07</td>
      <td>MNO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B08</td>
      <td>JKL</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat(frames,ignore_index = True)

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A01</td>
      <td>ABC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A02</td>
      <td>PQR</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A03</td>
      <td>DEF</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A04</td>
      <td>GHI</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B05</td>
      <td>XYZ</td>
    </tr>
    <tr>
      <th>5</th>
      <td>B06</td>
      <td>TUV</td>
    </tr>
    <tr>
      <th>6</th>
      <td>B07</td>
      <td>MNO</td>
    </tr>
    <tr>
      <th>7</th>
      <td>B08</td>
      <td>JKL</td>
    </tr>
  </tbody>
</table>
</div>




```python
result = pd.concat(frames,axis=1)
result
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Name</th>
      <th>id</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A01</td>
      <td>ABC</td>
      <td>B05</td>
      <td>XYZ</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A02</td>
      <td>PQR</td>
      <td>B06</td>
      <td>TUV</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A03</td>
      <td>DEF</td>
      <td>B07</td>
      <td>MNO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A04</td>
      <td>GHI</td>
      <td>B08</td>
      <td>JKL</td>
    </tr>
  </tbody>
</table>
</div>



#### `join()`


```python
data1 = {
    "name": ["Sally", "Mary", "John"],
    "age": [50, 40, 30]
}
data2 = {
    "qualified": [True, False, False]
}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df1.join(df2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>qualified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sally</td>
      <td>50</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mary</td>
      <td>40</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>John</td>
      <td>30</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Removing Rows/Columns

#### `drop()`

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



`drop` creates a new copy for you with the required changes. To modify the original Dataframe use `inplace=True` options.


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



#### Conditional Drop


```python
df = pd.read_csv("spam_small.csv")
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['len'] = df['v2'].apply(len)
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
      <th>len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df["len"] <100].index
```




    Int64Index([1, 3, 4, 6], dtype='int64')




```python
# df = df.drop(index=df[df["len"] <100].index)
df = df.drop(df[df["len"] <100].index)
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
      <th>len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
    </tr>
    <tr>
      <th>5</th>
      <td>spam</td>
      <td>FreeMsg Hey there darling it's been 3 week's n...</td>
      <td>147</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ham</td>
      <td>As per your request 'Melle Melle (Oru Minnamin...</td>
      <td>160</td>
    </tr>
    <tr>
      <th>8</th>
      <td>spam</td>
      <td>WINNER!! As a valued network customer you have...</td>
      <td>157</td>
    </tr>
  </tbody>
</table>
</div>



### Renaming Columns


```python
data = pd.read_csv('spam.csv')
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
```


```python
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.rename(columns={'v1': 'label', 'v2': 'messages'}, inplace=True)
```


```python
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>messages</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>



### 👉Shuffle a DataFrame rows

#### Using `pd.sample()`

The first option you have for shuffling pandas DataFrames is the `panads.DataFrame.sample` method that **returns a random sample of items**. In this method you can specify either the exact number or the fraction of records that you wish to sample. Since we want to shuffle the whole DataFrame, we are going to use `frac=1 `so that all records are returned.



```python
original = pd.DataFrame({
    'colA': [10, 20, 30, 40, 50],
    'colB': ['a', 'b', 'c', 'd', 'e'],
    'colC': [True, False, False, True, False],
    'colD': [0.5, 1.2, 2.4, 3.3, 5.5],
})
original

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>colA</th>
      <th>colB</th>
      <th>colC</th>
      <th>colD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>a</td>
      <td>True</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>b</td>
      <td>False</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>c</td>
      <td>False</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>d</td>
      <td>True</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>e</td>
      <td>False</td>
      <td>5.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
a = original.sample(frac=1)
a
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>colA</th>
      <th>colB</th>
      <th>colC</th>
      <th>colD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>c</td>
      <td>False</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>b</td>
      <td>False</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>e</td>
      <td>False</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>a</td>
      <td>True</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>d</td>
      <td>True</td>
      <td>3.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
b = original.sample(frac=1, random_state=42).reset_index(drop=True)
b

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>colA</th>
      <th>colB</th>
      <th>colC</th>
      <th>colD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>b</td>
      <td>False</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>e</td>
      <td>False</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>c</td>
      <td>False</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>a</td>
      <td>True</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>d</td>
      <td>True</td>
      <td>3.3</td>
    </tr>
  </tbody>
</table>
</div>



- `frac=1` means all rows of a dataframe
- `random_state=42` means keeping same order in each execution
- `reset_index(drop=True)` means reinitialize index for randomized dataframe

#### Using `sklearn.utils.shuffle()`


```python
from sklearn.utils import shuffle
c = shuffle(original, random_state=42)
c
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>colA</th>
      <th>colB</th>
      <th>colC</th>
      <th>colD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>b</td>
      <td>False</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>e</td>
      <td>False</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>c</td>
      <td>False</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>a</td>
      <td>True</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>d</td>
      <td>True</td>
      <td>3.3</td>
    </tr>
  </tbody>
</table>
</div>



## Data Types and Missing Values


```python
reviews = pd.read_csv('winemag-data-130k-v2-mod.csv', index_col=0)
reviews.head(n=2)

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>77</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
  </tbody>
</table>
</div>



### `dtypes`, `astype()`

The data type for a column in a DataFrame or a Series is known as the dtype.


```python
reviews.dtypes

```




    country                   object
    description               object
    designation               object
    points                     int64
    price                    float64
    province                  object
    region_1                  object
    region_2                  object
    taster_name               object
    taster_twitter_handle     object
    title                     object
    variety                   object
    winery                    object
    dtype: object



You can use the dtype property to grab the type of a specific column. For instance, we can get the dtype of the price column in the reviews DataFrame:




```python
reviews.price.dtype
```




    dtype('float64')



It's possible to convert a column of one type into another wherever such a conversion makes sense by using the `astype()` function. For example, we may transform the points column from its existing int64 data type into a float64 data type:




```python
reviews.points.astype('float64')
```




    0     77.0
    1     87.0
    2     87.0
    3     87.0
    4     90.0
          ...
    95    88.0
    96    88.0
    97    88.0
    98    88.0
    99    88.0
    Name: points, Length: 100, dtype: float64



### Missing data


Entries missing values are given the value `NaN`, short for `"Not a Number"`. For technical reasons these NaN values are always of the float64 dtype.


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


Pandas provides some methods specific to missing data. To select NaN entries you can use `pd.isnull()` (or its companion `pd.notnull()`). This is meant to be used thusly:



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



How many reviews in the dataset are missing a price?


```python
missing_price_reviews = reviews[reviews.price.isnull()]
n_missing_prices = len(missing_price_reviews)
# Cute alternative solution: if we sum a boolean series, True is treated as 1 and False as 0
n_missing_prices = reviews.price.isnull().sum()
# or equivalently:
n_missing_prices = pd.isnull(reviews.price).sum()
n_missing_prices
```




    8




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



What are the most common wine-producing regions? Create a Series counting the number of times each value occurs in the `region_1` field. This field is often missing data, so replace missing values with `Unknown`. Sort in descending order.  Your output should look something like this:


```python
reviews_per_region = reviews.region_1.fillna(
    'Unknown').value_counts().sort_values(ascending=False)
reviews_per_region
```




    Unknown                       12
    Sicilia                       11
    Napa Valley                    7
    Columbia Valley (WA)           5
    Virginia                       3
    Alsace                         3
    Willamette Valley              3
    Alexander Valley               2
    Etna                           2
    Paso Robles                    2
    Terre Siciliane                2
    Champagne                      2
    Sonoma Coast                   2
    Aglianico del Vulture          1
    Sonoma County                  1
    Puglia                         1
    Ancient Lakes                  1
    Chablis                        1
    Central Coast                  1
    Vin de France                  1
    Ribera del Duero               1
    Lake County                    1
    Vernaccia di San Gimignano     1
    Dry Creek Valley               1
    Beaujolais-Villages            1
    Navarra                        1
    McLaren Vale                   1
    South Australia                1
    Clarksburg                     1
    Brouilly                       1
    California                     1
    Knights Valley                 1
    Howell Mountain                1
    Eola-Amity Hills               1
    Toscana                        1
    Santa Ynez Valley              1
    Lake Michigan Shore            1
    Cafayate                       1
    McMinnville                    1
    North Coast                    1
    Sonoma Valley                  1
    Monticello                     1
    Mendoza                        1
    Juliénas                       1
    Régnié                         1
    Beaujolais                     1
    Cerasuolo di Vittoria          1
    Vittoria                       1
    Morellino di Scansano          1
    Romagna                        1
    Monica di Sardegna             1
    Oregon                         1
    Rías Baixas                    1
    Finger Lakes                   1
    Bordeaux Blanc                 1
    Mâcon-Milly Lamartine          1
    Calistoga                      1
    Name: region_1, dtype: int64



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



## Saving & loading files

Pandas can save `DataFrame`s to various backends, including file formats such as CSV, Excel, JSON, HTML and HDF5, or to a SQL database. Let's create a `DataFrame` to demonstrate this:


```python
df = pd.DataFrame({
	"id":np.arange(10),
	'b':np.random.normal(size=10),
	"c":pd.Series(np.random.choice(["cat",'dog',"hippo"],replace=True,size=10))
})
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.212280</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.492354</td>
      <td>hippo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1.667453</td>
      <td>dog</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-1.904760</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.520301</td>
      <td>hippo</td>
    </tr>
  </tbody>
</table>
</div>



### Saving
Let's save it to CSV, HTML and JSON:


```python
df.to_csv("my_df.csv")
df.to_csv("my_df_index_false.csv", index=False)
df.to_html("my_df.html")
df.to_json("my_df.json")

```

### Loading


```python
import os
print(os.getcwd())
print(os.listdir())
```

    d:\CSE\Others\ML-py\01pandas
    ['img', 'iris.csv', 'my_df.csv', 'my_df.html', 'my_df.json', 'my_df_index_false.csv', 'pandas.ipynb', 'README.md']


Now let's load our CSV file back into a `DataFrame`:

- Loading from file saved without `index=False`


```python
my_df_loaded = pd.read_csv("my_df.csv")
my_df_loaded.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1.106266</td>
      <td>hippo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>-1.612778</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>-0.264879</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>-0.213137</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>-0.184308</td>
      <td>hippo</td>
    </tr>
  </tbody>
</table>
</div>



- Loading from file saved with `index=False`


```python
my_df_loaded_index_false = pd.read_csv("my_df_index_false.csv")
my_df_loaded_index_false.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.106266</td>
      <td>hippo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-1.612778</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-0.264879</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-0.213137</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.184308</td>
      <td>hippo</td>
    </tr>
  </tbody>
</table>
</div>



- Loading from file saved without `index=False`, without `Unnamed: 0` column

The `pd.read_csv()` function is well-endowed, with over 30 optional parameters you can specify. For example, you can see in this dataset that the CSV file has a built-in index, which pandas did not pick up on automatically. To make pandas use that column for the index (instead of creating a new one from scratch), we can specify an `index_col`.



```python
my_df_loaded = pd.read_csv("my_df.csv",index_col=0)
my_df_loaded.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.106266</td>
      <td>hippo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-1.612778</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-0.264879</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-0.213137</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.184308</td>
      <td>hippo</td>
    </tr>
  </tbody>
</table>
</div>



- Or Dropping "Unnamed: 0" Column


```python
my_df_loaded = pd.read_csv("my_df.csv")
my_df_loaded = my_df_loaded.drop(columns=['Unnamed: 0'])
my_df_loaded.head()

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.106266</td>
      <td>hippo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-1.612778</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-0.264879</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-0.213137</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.184308</td>
      <td>hippo</td>
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



### Minimize the size of Large DataSet

[wine-reviews-dataset](https://www.kaggle.com/zynicide/wine-reviews)


```python
data = pd.read_csv('winemag-data-130k-v2.csv')
print(f"Pre Shape : {data.shape}")
# read only first 100 rows
data = pd.read_csv('winemag-data-130k-v2.csv', nrows=100, index_col=0)
data.head(n=2)
```

    Pre Shape : (129971, 14)





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Save the dataframe to a csv file
data.to_csv("winemag-data-130k-v2-mod.csv")
```


```python

new_data = pd.read_csv('winemag-data-130k-v2-mod.csv', index_col=0)
print(f"Post Shape: {new_data.shape}")
new_data.head(n=2)
```

    Post Shape: (100, 13)





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
  </tbody>
</table>
</div>



## Operations on `DataFrame`s

Although `DataFrame`s do not try to mimick NumPy arrays, there are a few similarities. Let's create a `DataFrame` to demonstrate this:


```python
data = {
	'roll_no': np.random.randint(1, 100, size=5),
	'ppr_id': np.random.randint(1000, 2000, size=5),
	'marks': np.random.randint(50, 100, size=5)
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
      <td>83</td>
      <td>1930</td>
      <td>83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>1954</td>
      <td>57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>92</td>
      <td>1638</td>
      <td>67</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92</td>
      <td>1126</td>
      <td>91</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56</td>
      <td>1369</td>
      <td>96</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['marks'].sum()
```




    394




```python
df['marks'].mean()
```




    78.8




```python
df['marks'].cumsum()
```




    0     83
    1    140
    2    207
    3    298
    4    394
    Name: marks, dtype: int32




```python
df['roll_no'].count()
```




    5




```python
df['marks'].min()
```




    57




```python
df['marks'].max()
```




    96




```python
df['marks'].var()
```




    269.2




```python
df['marks'].std()
```




    16.407315441594946




```python
df.corr()
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
      <th>roll_no</th>
      <td>1.000000</td>
      <td>-0.352082</td>
      <td>0.25746</td>
    </tr>
    <tr>
      <th>ppr_id</th>
      <td>-0.352082</td>
      <td>1.000000</td>
      <td>-0.70311</td>
    </tr>
    <tr>
      <th>marks</th>
      <td>0.257460</td>
      <td>-0.703110</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>



Similarly, adding a single value to a `DataFrame` will add that value to all elements in the `DataFrame`. This is called *broadcasting*:


```python
df + 1
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
      <td>84</td>
      <td>1931</td>
      <td>84</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46</td>
      <td>1955</td>
      <td>58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>93</td>
      <td>1639</td>
      <td>68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>93</td>
      <td>1127</td>
      <td>92</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>1370</td>
      <td>97</td>
    </tr>
  </tbody>
</table>
</div>



Of course, the same is true for all other binary operations, including arithmetic (`*`,`/`,`**`...) and conditional (`>`, `==`...) operations:


```python
df >= 500
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
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Aggregation operations, such as computing the `max`, the `sum` or the `mean` of a `DataFrame`, apply to each column, and you get back a `Series` object:


```python
df.mean()
```




    roll_no      73.6
    ppr_id     1603.4
    marks        78.8
    dtype: float64



The `all` method is also an aggregation operation: it checks whether all values are `True` or not. Let's see during which months all students got a grade greater than `5`:


```python
(df > 50).all()
```




    roll_no    False
    ppr_id      True
    marks       True
    dtype: bool



Most of these functions take an optional `axis` parameter which lets you specify along which axis of the `DataFrame` you want the operation executed. The default is `axis=0`, meaning that the operation is executed vertically (on each column). You can set `axis=1` to execute the operation horizontally (on each row). For example, let's find out which students had all grades greater than `5`:


```python
(df > 50).all(axis = 1)
```




    0     True
    1    False
    2     True
    3     True
    4     True
    dtype: bool



The `any` method returns `True` if any value is True. Let's see who got at least one grade 10:


```python
(df == 92).any(axis = 1)
```




    0    False
    1    False
    2     True
    3     True
    4    False
    dtype: bool



If you add a `Series` object to a `DataFrame` (or execute any other binary operation), pandas attempts to broadcast the operation to all *rows* in the `DataFrame`. This only works if the `Series` has the same size as the `DataFrame`s rows. For example, let's subtract the `mean` of the `DataFrame` (a `Series` object) from the `DataFrame`:


```python
df - df.mean()
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
      <td>9.4</td>
      <td>326.6</td>
      <td>4.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-28.6</td>
      <td>350.6</td>
      <td>-21.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.4</td>
      <td>34.6</td>
      <td>-11.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18.4</td>
      <td>-477.4</td>
      <td>12.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-17.6</td>
      <td>-234.4</td>
      <td>17.2</td>
    </tr>
  </tbody>
</table>
</div>



If you want to subtract the global mean from every grade, here is one way to do it:


```python
df - df.values.mean() # subtracts the global mean
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
      <td>-502.266667</td>
      <td>1344.733333</td>
      <td>-502.266667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-540.266667</td>
      <td>1368.733333</td>
      <td>-528.266667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-493.266667</td>
      <td>1052.733333</td>
      <td>-518.266667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-493.266667</td>
      <td>540.733333</td>
      <td>-494.266667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-529.266667</td>
      <td>783.733333</td>
      <td>-489.266667</td>
    </tr>
  </tbody>
</table>
</div>



## Grouping and Sorting


### Groupwise analysis


One function we've been using heavily thus far is the `value_counts()` function. We can replicate what `value_counts()` does by doing the following:


```python
reviews.groupby('points').points.count()
```




    points
    50     1
    66     1
    70     2
    77     1
    80     1
    85     9
    86    49
    87    23
    88    11
    90     1
    99     1
    Name: points, dtype: int64



`groupby()` created a group of reviews which allotted the same point values to the given wines. Then, for each of these groups, we grabbed the `points() `column and counted how many times it appeared. `value_counts()` is just a shortcut to this `groupby()` operation.



We can use any of the**summary functions** we've used before with this data. For example, to get the cheapest wine in each point value category, we can do the following:


```python
reviews.groupby('points').price.min()
```




    points
    50    28.0
    66    13.0
    70    14.0
    77     NaN
    80    30.0
    85    10.0
    86     9.0
    87    12.0
    88    12.0
    90    65.0
    99     NaN
    Name: price, dtype: float64



You can think of each group we generate as being a slice of our DataFrame containing only data with values that match. This DataFrame is accessible to us directly using the `apply()` method, and we can then manipulate the data in any way we see fit. For example, here's one way of selecting the name of the first wine reviewed from each winery in the dataset:


```python
reviews.groupby('winery').apply(lambda df: df.title.iloc[0])
```




    winery
    Acrobat                                             Acrobat 2013 Pinot Noir (Oregon)
    Adega Cooperativa do Cartaxo       Adega Cooperativa do Cartaxo 2014 Bridão Touri...
    Aresti                             Aresti 2014 Special Release Reserva Carmenère ...
    Baglio di Pianetto                 Baglio di Pianetto 2007 Ficiligno White (Sicilia)
    Basel Cellars                      Basel Cellars 2013 Inspired Red (Columbia Vall...
                                                             ...
    Vignerons de Bel Air                Vignerons de Bel Air 2011 Eté Indien  (Brouilly)
    Vignerons des Terres Secrètes      Vignerons des Terres Secrètes 2015  Mâcon-Mill...
    Viticultori Associati Canicatti    Viticultori Associati Canicatti 2008 Scialo Re...
    Yalumba                            Yalumba 2016 Made With Organic Grapes Chardonn...
    Z'IVO                               Z'IVO 2015 Rosé of Pinot Noir (Eola-Amity Hills)
    Length: 91, dtype: object



For even more fine-grained control, you can also group by more than one column. For an example, here's how we would pick out the best wine by country and province:


```python
reviews.groupby(['country', 'province']).apply(
    lambda df: df.loc[df.points.idxmax()]).head(3)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
    <tr>
      <th>country</th>
      <th>province</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Argentina</th>
      <th>Mendoza Province</th>
      <td>Argentina</td>
      <td>Raw black-cherry aromas are direct and simple ...</td>
      <td>Winemaker Selection</td>
      <td>87</td>
      <td>13.0</td>
      <td>Mendoza Province</td>
      <td>Mendoza</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Gaucho Andino 2011 Winemaker Selection Malbec ...</td>
      <td>Malbec</td>
      <td>Gaucho Andino</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>Argentina</td>
      <td>Baked plum, molasses, balsamic vinegar and che...</td>
      <td>Felix</td>
      <td>87</td>
      <td>30.0</td>
      <td>Other</td>
      <td>Cafayate</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Felix Lavaque 2010 Felix Malbec (Cafayate)</td>
      <td>Malbec</td>
      <td>Felix Lavaque</td>
    </tr>
    <tr>
      <th>Australia</th>
      <th>South Australia</th>
      <td>Australia</td>
      <td>This medium-bodied Chardonnay features aromas ...</td>
      <td>Made With Organic Grapes</td>
      <td>86</td>
      <td>18.0</td>
      <td>South Australia</td>
      <td>South Australia</td>
      <td>NaN</td>
      <td>Joe Czerwinski</td>
      <td>@JoeCz</td>
      <td>Yalumba 2016 Made With Organic Grapes Chardonn...</td>
      <td>Chardonnay</td>
      <td>Yalumba</td>
    </tr>
  </tbody>
</table>
</div>



Another `groupby()` method worth mentioning is `agg()`, which lets you run a bunch of different functions on your DataFrame simultaneously. For example, we can generate a simple statistical summary of the dataset as follows:


```python
reviews.groupby(['country']).price.agg([len, min, max])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>len</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Argentina</th>
      <td>2.0</td>
      <td>13.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>2.0</td>
      <td>18.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>1.0</td>
      <td>12.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>Chile</th>
      <td>5.0</td>
      <td>9.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>France</th>
      <td>14.0</td>
      <td>9.0</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>4.0</td>
      <td>9.0</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>24.0</td>
      <td>10.0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>Portugal</th>
      <td>2.0</td>
      <td>15.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>Spain</th>
      <td>3.0</td>
      <td>15.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>US</th>
      <td>43.0</td>
      <td>12.0</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>



### Sorting

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



### More example:


```python
reviews_written = reviews.groupby('taster_twitter_handle').size()
reviews_written
```




    taster_twitter_handle
    @AnneInVino          1
    @JoeCz               2
    @gordone_cellars     1
    @kerinokeefe        13
    @mattkettmann        3
    @paulgwine           6
    @vboone             16
    @vossroger          16
    @wawinereport        6
    @wineschach         10
    dtype: int64



What is the **best** wine I can buy for a given amount of money? Create a `Series` whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review. Sort the values by price, ascending (so that `4.0` dollars is at the top and `3300.0` dollars is at the bottom).


```python
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()
best_rating_per_price
```




    price
    9.0      86
    10.0     86
    11.0     86
    12.0     88
    13.0     87
    14.0     87
    15.0     87
    16.0     87
    17.0     87
    18.0     88
    19.0     88
    20.0     88
    21.0     86
    22.0     88
    23.0     88
    24.0     87
    25.0     86
    26.0     86
    27.0     87
    28.0     50
    29.0     86
    30.0     88
    32.0     87
    34.0     87
    35.0     87
    40.0     86
    46.0     86
    50.0     86
    55.0     88
    58.0     86
    65.0     90
    69.0     87
    75.0     88
    100.0    86
    Name: points, dtype: int64



What are the minimum and maximum prices for each `variety` of wine? Create a `DataFrame` whose index is the `variety` category from the dataset and whose values are the `min` and `max` values thereof.


```python
price_extremes = reviews.groupby('variety').price.agg([min, max])
price_extremes[:3]

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>variety</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aglianico</th>
      <td>32.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>Albariño</th>
      <td>16.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Bordeaux-style Red Blend</th>
      <td>46.0</td>
      <td>75.0</td>
    </tr>
  </tbody>
</table>
</div>



Create a `Series` whose index is reviewers and whose values is the average review score given out by that reviewer. Hint: you will need the `taster_name` and `points` columns.


```python
reviewer_mean_ratings=reviews.groupby('taster_name').points.mean()
reviewer_mean_ratings
```




    taster_name
    Alexander Peartree    87.000000
    Anna Lee C. Iijima    86.800000
    Anne Krebiehl MW      88.000000
    Jim Gordon            86.000000
    Joe Czerwinski        86.000000
    Kerin O’Keefe         86.000000
    Matt Kettmann         86.666667
    Michael Schachner     80.800000
    Paul Gregutt          87.000000
    Roger Voss            86.812500
    Sean P. Sullivan      86.333333
    Virginie Boone        86.625000
    Name: points, dtype: float64




```python
reviewer_mean_ratings.describe()
```




    count    12.000000
    mean     86.169792
    std       1.782245
    min      80.800000
    25%      86.000000
    50%      86.645833
    75%      86.859375
    max      88.000000
    Name: points, dtype: float64



What combination of countries and varieties are most common? Create a `Series` whose index is a `MultiIndex`of `{country, variety}` pairs. For example, a pinot noir produced in the US should map to `{"US", "Pinot Noir"}`. Sort the values in the `Series` in descending order based on wine count.


```python
country_variety_counts = reviews.groupby(
    ['country', 'variety']).size().sort_values(ascending=False)
country_variety_counts

```




    country    variety
    US         Pinot Noir                    6
               Red Blend                     5
               Cabernet Sauvignon            5
    France     Gamay                         5
    Italy      Red Blend                     4
    US         Sauvignon Blanc               4
               Chardonnay                    4
    Italy      White Blend                   4
    US         Riesling                      3
    Italy      Nero d'Avola                  3
    US         Bordeaux-style Red Blend      3
    Germany    Riesling                      3
    Argentina  Malbec                        2
    US         Meritage                      2
    Italy      Sangiovese                    2
    France     Gewürztraminer                2
               Chardonnay                    2
               Champagne Blend               2
    US         Merlot                        2
               Pinot Gris                    2
               Chenin Blanc                  1
               Cabernet Franc                1
               Petite Sirah                  1
               Albariño                      1
    Spain      Tempranillo-Merlot            1
               Tempranillo Blend             1
               Albariño                      1
    Portugal   Touriga Nacional              1
    US         Viognier                      1
    Portugal   Portuguese Red                1
    US         Malbec                        1
    Italy      Primitivo                     1
               Vernaccia                     1
    France     Pinot Gris                    1
    Australia  Rosé                          1
    Austria    Grüner Veltliner              1
    Chile      Carmenère                     1
               Merlot                        1
               Petit Verdot                  1
               Pinot Noir                    1
               Viognier-Chardonnay           1
    France     Bordeaux-style White Blend    1
               Petit Manseng                 1
    Germany    Gewürztraminer                1
    Italy      Rosato                        1
               Aglianico                     1
               Cabernet Sauvignon            1
               Catarratto                    1
               Frappato                      1
               Grillo                        1
               Inzolia                       1
               Monica                        1
               Nerello Mascalese             1
    Australia  Chardonnay                    1
    US         Zinfandel                     1
    dtype: int64



## Categorical encoding

### Introduction

In many Machine-learning or Data Science activities, the**data set might contain text or categorical values** (basically non-numerical values). For example, color feature having values like red, orange, blue, white etc. Meal plan having values like breakfast, lunch, snacks, dinner, tea etc. Few algorithms such as CATBOAST, decision-trees can handle categorical values very well but **most of the algorithms expect numerical values** to achieve state-of-the-art results.

Over your learning curve in AI and Machine Learning, one thing you would notice that **most of the algorithms work better with numerical inputs**. *Therefore, the main challenge faced by an analyst is to convert text/categorical data into numerical data and still make an algorithm/model to make sense out of it*. **Neural networks, which is a base of deep-learning, expects input values to be numerical.**

There are many ways to convert categorical values into numerical values. Each approach has its own trade-offs and impact on the feature set. Hereby, I would focus on 2 main methods:` One-Hot-Encoding` and `Label-Encoder`.

<div align="center">
<img src="img/len.jpg" alt="encoding.jpg" width="800px">
</div>


[More](https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd)

### Label Encoding

#### Custom Map Function


```python
people_dict = {
	"gender":pd.Series(['Female','Male','Female','Female',"Other"])
}
people = pd.DataFrame(people_dict)
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Other</td>
    </tr>
  </tbody>
</table>
</div>




```python
def f(g):
	if g == 'Male':
		return 0
	elif g == "Female":
		return 1
	else:
		return 2

people['label'] = people.gender.apply(f)
people
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Other</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



#### Using Pandas.factorize()

`pandas.factorize()` method helps to get the numeric representation of an array by identifying distinct values. It returns a tuple of two arrays, the first containing **numeric representation of the values in the original array** and the second containing **unique values**.

**Parameters**:
- `values` : 1D sequence.
- `sort` : [bool, Default is False] Sort uniques and shuffle labels.
- `na_sentinel` : [ int, default -1] Missing Values to mark ‘not found’.

**Return**: Numeric representation of array


```python
arr= ['b', 'd', 'd', 'c', 'a', 'c', 'a', 'b']
labels, uniques = pd.factorize(arr)

print("Original Array:\n",arr )
print("Numeric Representation : \n", labels)
print("Unique Values : \n", uniques)

```

    Original Array:
     ['b', 'd', 'd', 'c', 'a', 'c', 'a', 'b']
    Numeric Representation :
     [0 1 1 2 3 2 3 0]
    Unique Values :
     ['b' 'd' 'c' 'a']



```python
arr= ['b', 'd', 'd', 'c', 'a', 'c', 'a', 'b']
labels, uniques = pd.factorize(arr,sort=True)

print("Original Array:\n",arr )
print("Numeric Representation : \n", labels)
print("Unique Values : \n", uniques)

```

    Original Array:
     ['b', 'd', 'd', 'c', 'a', 'c', 'a', 'b']
    Numeric Representation :
     [1 3 3 2 0 2 0 1]
    Unique Values :
     ['a' 'b' 'c' 'd']



```python
values = {
	"class":pd.Series(['A','B','C','D','A','A','C','B']),
}
data = pd.DataFrame(values)
```


```python
q =data["class"].unique()
q
```




    array(['A', 'B', 'C', 'D'], dtype=object)




```python
data["class"].factorize()
```




    (array([0, 1, 2, 3, 0, 0, 2, 1], dtype=int64),
     Index(['A', 'B', 'C', 'D'], dtype='object'))




```python
Y,label = data["class"].factorize()
data["Y"] = Y
data
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>B</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



##### Reverse the process - Decoding

Sometimes, you might want to convert the numeric representation back to the original values. For example, if the predicted class is 1, you might want to know what the original class was.


```python
Y,label = data["class"].factorize()
Y,label
```




    (array([0, 1, 2, 3, 0, 0, 2, 1], dtype=int64),
     Index(['A', 'B', 'C', 'D'], dtype='object'))




```python
predicted_class = 1
```


```python
label[predicted_class]
```




    'B'



#### Using  🌟`sklearn.LabelEncoder()`🌟



```python
from sklearn.preprocessing import LabelEncoder
# creating initial dataframe
gender_types = ['male','female','other']
gender_df = pd.DataFrame(gender_types, columns=['gender'])
gender_df

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>other</td>
    </tr>
  </tbody>
</table>
</div>




```python
# creating instance of labelencoder
labelencoder = LabelEncoder()
```


```python
# Assigning numerical values and storing in another column
gender_df['label'] = labelencoder.fit_transform( gender_df['gender'])
gender_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>other</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



##### Decoding


```python
labelencoder.inverse_transform([0])
```




    array(['female'], dtype=object)




```python
labelencoder.inverse_transform([0,0,1,1,1,2])
```




    array(['female', 'female', 'male', 'male', 'male', 'other'], dtype=object)



### One-Hot-Encoding

Though label encoding is straight but it has the disadvantage that the numeric values can be misinterpreted by algorithms as having some sort of hierarchy/order in them. This ordering issue is addressed in another common alternative approach called ‘One-Hot Encoding’. In this strategy, each category value is converted into a new column and assigned a 1 or 0 (notation for true/false) value to the column.

##### Using `Pandas.get_dummies()`


```python
gender_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>other</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.get_dummies(gender_df.gender)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>female</th>
      <th>male</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.get_dummies(gender_df.gender,prefix='Sex')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>Sex_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Attaching to the DataFrame:


```python
dummies = pd.get_dummies(gender_df.gender,prefix='Sex')
gender_df = pd.concat([gender_df, dummies], axis=1)
gender_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>label</th>
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>Sex_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>other</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Multiple Columns can be transform at a time : i.e df = pd.get_dummies(df, columns=['Sex', 'Embarked'])


[More](https://towardsdatascience.com/what-is-one-hot-encoding-and-how-to-use-pandas-get-dummies-function-922eb9bd4970)

#### Using 🌟`sklearn.OneHotEncoder()`🌟


```python
gender_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>other</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(gender_df[['label']]).toarray())
enc_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# merge with main df gender_df on key values
gender_df = gender_df.join(enc_df)
gender_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>label</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>other</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


