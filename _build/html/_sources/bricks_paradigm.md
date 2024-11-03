# Paradigms

![top](/images/top.png)

We denote by `paradigm` the overall problem that is used to train a neural network: *such as supervised, metric-learning, self-supervised, adversarial, encoder-decoder, ...*



(lab_supervised)=
## Supervised

Supervised learning is the most standard paradigm in machine learning, hence in deep learning, in which <mark>one has access to both input data $X$ and the corresponding ground-truth $y$</mark>.

The goal is then to define a function $f$ (a specific neural network architecture) and optimize its parameters $\theta$ such that $\hat{\mathbf{y}}=f_{\theta}(\mathbf{x})$ best approximates $\mathbf{y}$.
This is done by defining a loss $\mathcal{L}$ associated to the approximation of $\mathbf{y}$ by $\hat{\mathbf{y}}$.
Such a loss can be
- binary cross entropy (for binary classification problems, i.e. $y \in \{0,1\}$, or multi-label problems i.e. $\mathbf{y} \in \{0,1\}^C$,
- categorical cross entropy (for multi-class problem, i.e. $y \in \{0,\ldots, C-1\}$)
- mean square error (for regression problems, i.e. $y \in \mathbb{R}$)

Since we do not have access to the distribution $p(\mathbf{x},\mathbf{y})$ but only to samples of it $\mathbf{x}^{(i)}$, $\mathbf{y}^{(i)} \sim p(\mathbf{x},\mathbf{y})$, we **empirically minimize the loss/risk** for a set of training examples $(\mathbf{x}^{(i)}$, $\mathbf{y}^{(i)})$:

$$\theta^* = \arg\min_{\theta} \sum_{i=0}^{I-1} \mathcal{L}(f_{\theta}(\mathbf{x}^{(i)}), \mathbf{y}^{(i)})$$

This minimization is usually done using one type of Steepest Gradient Descent (SDG, Momentum, AdaGrad, AdaDelta, ADAM) and using various cardinality for $I$ (stochastic, mini-batch, batch GD).


## Self-supervised





(lab_metric_learning)=
## Metric Learning

Metric learning is a type of machine learning technique focused on <mark>learning a distance function or similarity measure between data points</mark>.
The goal is to <mark>map input data into a space where</mark>
- similar examples are close together and
- dissimilar examples are far apart, based on a certain metric (e.g., Euclidean distance).

There exist several type of supervision to achieve this
- **Class** labels: $(\mathbf{x},\mathbf{y})$
- **Pairwise** similarity/dissimilarity: $(\mathbf{x}^{(1)},\mathbf{x}^{(2)},\pm)$
- **Relative** comparisons (triplet): $(\mathbf{x}^{(1)}, \mathbf{x}^{(2)},\mathbf{x}^{(3)}) \Rightarrow d(\mathbf{x}^{(1)},\mathbf{x}^{(2)}) < d(\mathbf{x}^{(1)},\mathbf{x}^{(3)})$

There exist many algorithm to train such a model such as
- <mark>Contrastive Loss</mark> {cite}`DBLP:conf/cvpr/HadsellCL06` in which we optimize in turns (but not jointly) the model to minimize a distance for similar pairs and maximize (up to a margin) it for dissimilar pairs
- <mark>Triplet Loss</mark> {cite}`DBLP:journals/corr/HofferA14` see below

The triplet loss can be extended to the multiple-loss with close relationship with Contrastive Learning (InfoNCE, NT-Xent losses).

Fore more details, see the very good tutorial
- ["Metric Learning for Music Information Retrieval" by Brian McFee, Jongpil Lee and Juhan Nam](https://github.com/bmcfee/ismir2020-metric-learning)

(lab_triplet)=
### Triplet Loss

The goal is to train a network $f_{\theta}$ such that the resulting projections satisfy the following triplet constraint:

$$d( f_{\theta}(\mathbf{x}_A), f_{\theta}(\mathbf{x}_P) ) + \alpha < d(f_{\theta}(\mathbf{x}_A), f_{\theta}(\mathbf{x}_N))$$

$$d_{AP} + \alpha < d_{AN} $$

where
- $\mathbf{x}_A$ is an anchor,
- $\mathbf{x}_P$ a positive we consider close to $\mathbf{x}_A$,
- $\mathbf{x}_N$ a negative we consider distant from $\mathbf{x}_A$
- $\alpha$ a margin.

In other words, we want $d_{AP}$ to be smaller by a margin $\alpha$ than $d_{AN}$

We solve this by minimizing the following loss $\mathcal{L} = \max(d_{AP} + \alpha - d_{AN},0)$.
It is usual to L2-normalized the output of $f_{\theta}$ (which then lives in the unit-hypersphere) to facilitate the setting of the $\alpha$ value.

![triplet-loss](/images/brick_triplet.png)\
**Figure**
*Triplet Loss, bringing A and P closer and A and N further appart; image source: [Link](https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905)*


```python
loss = F.relu(dist_pos + (param_d.margin - dist_neg))
```


(lab_TripletMining)=
### Triplet Mining

Triplet mining is the process of <mark>selecting the triplets</mark> for training using the triplet loss.
The goal is to ensure the model learns effectively by choosing the right combination of examples.

Given the choice of an Anchor and a Positive, we denotes by

- **Easy negatives**: N that are already far from A.
	- do not provide much useful information (since the model already distinguishes them well).
- **Hard negatives**: N that are very close to A  (even closer than the P).
	- these are difficult for the model to separate
	- might lead to instability or overfitting.
- **Semi-hard negatives**: N that are farther from A than the P but still relatively close.
	- provides valuable information  (because they are challenging without being as problematic as hard negatives).

![triplet-mining](/images/brick_tripletmining.png)\
**Figure**
*Triplet mining: easy, hard, semi-hard; image source: [Link](https://www.researchgate.net/figure/Online-Triplet-Mining-strategies-For-an-anchor-blue-A-and-a-positive-green-P-sample_fig6_364057028)*

We also distinguish between
- **Offline mining**: triplets are <mark>selected prior to training</mark>
	- may not adapt to the evolving model during training.
- **Online mining**: triplets are <mark>selected during training (from the current mini-batch)</mark> using the already learned projection $f_{\theta}$
	- allows selecting the most informative triplets



## Adversarial

## Encoder-Decoder

## Diffusion
