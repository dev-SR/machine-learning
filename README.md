# Machine Learning

- [Machine Learning](#machine-learning)
	- [Jupyter Notebook](#jupyter-notebook)
		- [Opening Jupyter Notebook](#opening-jupyter-notebook)
	- [Convert `ipynb` files into html, markdown, pdf and other format files](#convert-ipynb-files-into-html-markdown-pdf-and-other-format-files)
	- [`tqdm` examples:](#tqdm-examples)

## Jupyter Notebook

### Opening Jupyter Notebook

To open `Jupyter Notebook`, we firstly have to `activate` our environment and use command: `jupyter notebook`

```bash
jupyter notebook
```
<div align="center"><img src="img/venv_2.jpg" alt="dfs" width="800px"></div>

## Convert `ipynb` files into html, markdown, pdf and other format files

```bash
# ipython nbconvert --to FORMAT notebook.ipynb

jupyter nbconvert --to html test.ipynb
jupyter nbconvert --to markdown test.ipynb --output README.md
jupyter nbconvert --to pdf test.ipynb

# Note: VsCode jupyter notebooks has built in converter
```

[https://ipython.org/ipython-doc/3/notebook/nbconvert.html](https://ipython.org/ipython-doc/3/notebook/nbconvert.html)

[https://www.programmersought.com/article/95748768264/](https://www.programmersought.com/article/95748768264/)

## `tqdm` examples:

Nested Progress Bar with TQDM in Python:

a nested progress bar using TQDM in Python, providing an intuitive visualization of progress for both the outer and inner loops.


```python
from tqdm.notebook import tqdm
from time import sleep
for i in tqdm(range(3), desc="Outer Progress", position=0):
    for j in tqdm(range(5), desc="Inner Progress", position=1):
        sleep(0.1)
```

Controlled bar:

```python
with tqdm(total=1000) as pbar:
    for i in range(100):
        sleep(0.1)
        pbar.update(10)
        # 10*100 = 1000 (total)
```

Mod descriptions:

```python
import time
for i in tqdm(range(100), desc="Progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
    time.sleep(0.1)
```

