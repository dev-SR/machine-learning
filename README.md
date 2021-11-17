# Machine Learning

- [Machine Learning](#machine-learning)
	- [Virtual Environment](#virtual-environment)
		- [Activating Virtual Environment Using Conda](#activating-virtual-environment-using-conda)
		- [Installing Jupyter Notebook inside Conda environment](#installing-jupyter-notebook-inside-conda-environment)
		- [Opening Jupyter Notebook](#opening-jupyter-notebook)
	- [Convert `ipynb` files into html, markdown, pdf and other format files](#convert-ipynb-files-into-html-markdown-pdf-and-other-format-files)

## Virtual Environment

<!-- <div align="center" ><img src="../img/venv-1.jpg" alt="venv 1" width="700px"></div> -->

### Activating Virtual Environment Using Conda

```bash
conda create ./env --prefix numpy pandas matplotlib scikit-learn
```

This will create a virtual environment with the following Data Science tools numpy, pandas, matplotlib
and scikit learn.

After creating the VM, `conda` will show a message to activate the environment.

```bash
conda activate D:\ML\project\env
```

### Installing Jupyter Notebook inside Conda environment

Once we are inside `Conda` environment, we use the following command to install `Jupyter Notebook`

```bash
conda install jupyter
```

### Opening Jupyter Notebook

To open `Jupyter Notebook`, we firstly have to `activate` our environment and use command: `jupyter notebook`

```bash
conda activate D:\ML\project\env
jupyter notebook
```

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
