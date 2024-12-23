# Notebooks in Colab

We describe here how you can run in [Google Colab](https://colab.research.google.com/) the notebooks we provide for this tutorial and therefore reproduce the  experiments we present.

*Google Colab, short for Collaboratory, is a free, cloud-based platform by Google that allows users to write and execute Python code in a Jupyter Notebook environment. It’s especially popular for machine learning and data science tasks because it provides access to powerful hardware like GPUs and TPUs at no cost.*

From Colab, choose `File`, `Import`, `Github`, and indicate the following repository `https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook`.

```{figure} ./images/colab_1.png
---
name: colab_1
---
Importing github files in Google Colab
```



Within the imported notebook, change `do_deploy` to` True`.\
If you then run the notebook, it will automatically
1. **install/import** all necessary packages,
2. **git clone** the code of this tutorial in your local (temporary) Colab space
3. **download/unzip** the necessary datasets (audio in .hdf5 and annotations in .pyjama) in your local (temporary) Colab space

```{figure} ./images/colab_2.png
---
name: colab_2
---
Deploying package and dataset within Google Colab
```




### Actions:

We show that
- import a notebook on colab
