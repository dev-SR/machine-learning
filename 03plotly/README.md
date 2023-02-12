# Plotly.py

- [Plotly.py](#plotlypy)
  - [Scatter Plot](#scatter-plot)
  - [Save to Pdf/PNG..](#save-to-pdfpng)
  - [🔥 Example](#-example)
    - [Joint PDFs + Area Under Curve](#joint-pdfs--area-under-curve)


```python
"""
cd .\03plotly\
jupyter nbconvert --to markdown plotly.ipynb --output README.md

 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('png')

# plotly
import plotly.express as px
import plotly.graph_objs as go

```

## Scatter Plot


```python
# Create a sample DataFrame
df = px.data.tips()
df.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a 2D scatter plot
fig = px.scatter(df, x='total_bill', y='tip')
fig.show()
```




```python
fig = px.scatter(df, x='total_bill', y='tip' , color='smoker',size='size')
fig.show()
```



If you want to see `day` information while hovering then ->


```python
fig = px.scatter(df, x='total_bill', y='tip' , color='smoker',size='size', hover_data=['day'])
fig.show()
```




```python
# Create a 3D scatter plot
fig = px.scatter_3d(df, x='total_bill', y='tip', z='size', color='smoker', size='size', size_max=15,
                   title='3D Scatter Plot of Total Bill, Tip, and Size')
fig.update_layout(width=800, height=800)
fig.show()
```



## Save to Pdf/PNG..


```python
# !pip install -U kaleido
```


```python
fig = px.scatter(df, x='total_bill', y='tip' , color='smoker',size='size')
fig.write_image('scatter_plot.pdf', format='pdf', width=1000, height=500, scale=3.33)
# scale ~ (width / dpi) =  (1000 / 300) = 3.33

```

## 🔥 Example

### Joint PDFs + Area Under Curve


```python
from scipy.integrate import quad
from scipy.stats import multivariate_normal
# def pdf(data, mean: float, variance: float):
#   # A normal continuous random variable.
#   s1 = 1 / (np.sqrt(2 * np.pi * variance))
#   s2 = np.exp(-(np.square(data - mean) / (2 * variance)))
#   return s1 * s2
np.random.seed(0)
PC1 = np.random.normal(1, 1, 100)
PC2 = np.random.normal(0, 1, 100)
text1 = "PC1"
text2 = "PC2"
x_start, x_end = -10, 10
x = np.linspace(x_start, x_end, 1000)
pdf1 = multivariate_normal.pdf(x, mean=PC1.mean(), cov=PC1.var())
pdf2 = multivariate_normal.pdf(x, mean=PC2.mean(), cov=PC2.var())
AUC = np.min(np.vstack((pdf1, pdf2)), axis=0)

# Overlap Percentage Calculation
overlap, _ = quad(lambda x: np.minimum(
multivariate_normal.pdf(x, mean=PC1.mean(), cov=PC1.var()), multivariate_normal.pdf(x, mean=PC2.mean(), cov=PC2.var())), -np.inf, np.inf)
percent_overlap = 100 * overlap

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=pdf1, name="pdf1"))
fig.add_trace(go.Scatter(x=x, y=pdf2, name="pdf2"))
fig.add_trace(go.Scatter(x=x, y=AUC, fill='tozeroy',fillcolor='rgba(0,100,110,0.5)', name=f"Overlap: {round(percent_overlap,2)}%"))

fig.add_annotation(x=PC1.mean(), y=np.max(pdf1), text=text1,arrowhead=2,showarrow=True)
fig.add_annotation(x=PC2.mean(), y=np.max(pdf2), xref='x', yref='y', text=text2, ax=1, ay=-50, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='black', showarrow=True)

fig.update_layout(width=1000, height=500)
fig.show()
```




```python

```
