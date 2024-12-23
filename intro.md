# Introduction

## Organisation of the book

The <mark>first part</mark> of the book, **"Tasks"**, describes a subset of typical audio-based MIR tasks.\
To facilitate the reading of the book, we follow a similar structure to describe each of the audio-based MIR tasks we consider.
We describe in turn:
- the **goal** of the task
- the performance measures used to **evaluate** the task
- the popular **datasets** used for the task \
*(Datasets can be used to train system or evaluate the performances of a system.)*
- how we can solve the task using **deep learning**. \
*(This part refers to bricks that are described individually in the second part of the book.*)


The <mark>second part</mark> of the book, **"Deep Learning Bricks"**, describes each brick individually.\
We have chosen to separate the description of the bricks from the tasks in which they can be used in order to emphasise the fact that the same brick can be used for several tasks.

```{figure} ./images/main_concept1.png
---
name: concept1
---
Overall description of tasks in terms of goal/evaluation/datasets/model
```





## Simplifying the development

To make our life easier and **facilitate the reading of the code of the notebooks** we will rely on the following elements.
- for **datasets** (audio and annotations): [.hdf5](https://docs.h5py.org/) (for audio) and [.pyjama](https://github.com/geoffroypeeters/pyjama) (for annotations), described below
- for **deep learning**: [pytorch](https://pytorch.org/) (a python library for deep learning)
  - for the dataset/dataloader
  - for the models, the losses, the optimizers
- for **training**: [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/) (a library added to pytorch that makes it easier/faster to train and deploy models)




### Evaluation metrics

In the notebooks, we will rely most of the time on
- [`mir_eval`](https://github.com/craffel/mir_eval) which provides most MIR specific evaluation metrics,
- [`scikit-learn`](https://scikit-learn.org/) which provides the standard machine-learning evaluation metrics.



### In summary

We summarize the various elements of code to be written below.

```{figure} ./images/main_concept2.png
---
name: concept2
---
Code to be written
```
