## Paradigms

(lab_supervised)=
### Supervised

Supervised learning is the most standard paradigm in machine learning, hence in deep learning, in which one has access to both input data $X$ and the corresponding ground-truth $y$.
The goal is then to define a function $f$ (a specific neural network architecture) and optimize its parameters $\theta$ such that $\hat{y}=f_{\theta}(X)$ best approximates $y$.
This is done by defining a loss $\mathcal{L}$ associated to the approximation of $y$ by $\hat{y}$.
Such a loss can be
- binary cross entropy (for binary classification problems, i.e. $y \in \{0,1\}$, or multi-label problems i.e. $y \in \{0,1\}^C$,
- categorical cross entropy (for multi-class problem, i.e. $y \in \{0,\ldots, C-1\}$)
- mean square error (for regression problems, i.e. $y \in \mathbb{R}$)

Since we do not have access to the distribution $p(X,y)$ but only to samples of it $X^{(i)}$, $y^{(i)} \sim p(X,y)$, we **empirically minimize the loss/risk** for a set of training examples $(X^{(i)}$, $y^{(i)})$:
$\theta^* = \arg\min_{\theta} \sum_{i=0}^{I-1} \mathcal{L}(f_{\theta}(X^{(i)}), y^{(i)})$.

This minimization is usually done using one type of Steepest Gradient Descent (SDG, Momentum, AdaGrad, AdaDelta, ADAM) and using various cardinality for $I$ (stochastic, mini-batch, batch GD).

The function $f$ is defined by writing a class inherited from the `nn.Module`
The choice of $I$ is defined in the `torch.utils.data.DataLoader`.
In `TorchLightning`, the loss, model and updates method are defined as follow:


```python
class AutoTaggingLigthing(pl.LightningModule):
    def __init__(self, in_model):
        super().__init__()
        self.model = in_model
        elf.loss = nn.CrossEntropyLoss() # --- multiclass
    def training_step(self, batch, batch_idx):
        hat_y = self.model(batch['X'])
        loss = self.loss(hat_y, batch['y'])
        return loss
    def validation_step(self, batch, batch_idx):
        hat_y = self.model(batch['X'])
        loss = self.loss(hat_y, batch['y'])
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 0.001)
        return optimizer
```

### Self-supervised

(lab_metric_learning)=
### Metric Learning

Metric learning is a type of machine learning technique focused on learning a distance function or similarity measure between data points. The goal is to map input data into a space where similar examples are close together and dissimilar examples are far apart, based on a certain metric (e.g., Euclidean distance).

There exist several type of supervision to achieve this
- Class labels: $(x,y)$
- Pairwise similarity/dissimilarity: $(x^{(1)},x^{(2)},\pm)$
- Relative comparisons (triplet): $(x^{(1)}, x^{(2)},x^{(3)}) \Rightarrow d(x^{(1)},x^{(2)}) < d(x^{(1)},x^{(3)})$

There exist many algorithm to train such a model such as
- Contrastive Loss {cite}`DBLP:conf/cvpr/HadsellCL06` in which we optimize in turns (but not jontly) the model to minize a distance for similar pairs and maximize (up to a margin) it for dissimilar pairs
- Triplet Loss {cite}`DBLP:journals/corr/HofferA14` see below

The triplet loss can be extended to the multiple-loss with close relationship with Contrastive Learning (InfoNCE, NT-Xent losses).



(lab_triplet)=
#### Triplet Loss

The goal is to train a network $f_{theta}$ such that the resulting projections satisfy the following triplet constraint:
$d( f_{\theta}(A), f_{\theta}(P) ) + \alpha < d(f_{\theta}(A), f_{\theta}(N)) $
where $A$ is an anchor, $P$ a positive we consider close to $A$, $N$ a negative we consider distance from $A$ and $\alpha$ a margin.
In other words, we want $d_{AN}$ to be larger than $d_{AP}$ by a margin $\alpha$

We solve this by minimizing the following loss
$\mathcal{L} = \max(d_{AP} + \alpha - d_{AN},0)$ where $\alpha$ is a margin parameters.
It is usual to L2-normalized the output of $f_{\theta}$ (which then lives in the unit-hypersphere) to facilite the setting the $\alpha$.

![triplet-loss](/images/brick_triplet.png)

*image source: https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905*


```python
loss = F.relu(dist_pos + (param_d.margin - dist_neg))
```


#### Triplet Mining

Triplet mining is the process of selecting triplets of examples (Anchor, Positive, Negative) for training in triplet loss.
The goal is to ensure the model learns effectively by choosing the right combination of examples.

Given the choice of an Anchor and a Positive, we denotes by

- **Easy negatives**: Negatives (different class) that are already far from the anchor. These don't provide much useful information since the model already distinguishes them well.
- **Semi-hard negatives**: Negatives that are farther from the anchor than the positive but still relatively close. These provide valuable learning opportunities because they are challenging without being as problematic as hard negatives.
- **Hard negatives**: Negatives that are very close to the anchor in the feature space (even closer than the positive). These are difficult for the model to separate and can help the model learn but might also lead to instability or overfitting.

![tripletmining](/images/brick_tripletmining.png)

*image source: https://www.researchgate.net/figure/Online-Triplet-Mining-strategies-For-an-anchor-blue-A-and-a-positive-green-P-sample_fig6_364057028*

We also distinguish between
- **Offline mining**: triplets are chosen prior to training and may not adapt to the evolving model during training.
- **Online mining**: triplets are selected from the current mini-batch using the already learned projection $f_{\theta}$, this allows selecting the most informative triplets



### Adversarial

### Encoder-Decoder

### Diffusion
