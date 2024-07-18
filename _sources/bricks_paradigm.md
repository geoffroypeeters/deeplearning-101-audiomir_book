## Paradigms

### Supervised

### Self-supervised

### Metric Learning

(lab_triplet)=
#### Triplet Loss

train the network $f_{theta}$ such that the resulting projections satisfy the following constraint:

$d( f_{\theta}(A), f_{\theta}(P) ) + \alpha < d(f_{\theta}(A), f_{\theta}(N)) $

where $A$ is an anchor, $P$ a positive we consider close to $A$, $N$ a negative we consider distance from $A$ and $\alpha$ a margin.

In other words, we want $d_{AN}$ to be larger than $d_{AP}$ by a margin $\alpha$
![triplet-loss](/images/brick_triplet.png)

We solve this by minimizing the following loss
$\mathcal{L} = \max(d_{AP} + \alpha - d_{AN},0)$

`loss = F.relu(dist_pos + (param_d.margin - dist_neg)) `

#### Triplet Mining

![tripletmining](/images/brick_tripletmining.png)


### Adversarial

### Encoder-Decoder

### Diffusion
