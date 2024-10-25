# Architectures

(lab_unet)=
## U-Net

The U-Net was proposed in {cite}`DBLP:conf/miccai/RonnebergerFB15` in the framework of biomedical image segmentation and made popular in MIR by {cite}`DBLP:conf/ismir/JanssonHMBKW17` for singing voice separation.

The U-Net is an auto-encoder with skip-connections.
The encoder (left part) downsample the spatial dimensions and increase the depth, while the decoder (right part) upsample the spatial dimensions and decrease the depth.
Skip connections are added between equivalent layers of the encoder and decoder: the 256 channels level of the encoder is concatenated with the 256 level of the decoder to form a 512 tensor.

![brick_unet](/images/brick_unet.png)

*image source: {cite}`DBLP:conf/miccai/RonnebergerFB15`*

The goal of the skip-connections are two-folds:
- to bring back details of the original images to the decoder (the bottleneck being to compressed to represent detailed information)
- to facilitate the backpropagation of the gradient.

The upsampling part can be done either
- using Transposed Convolution (hence a well-known checkerboard artefact may appears)
- using Interpolation followed by Normal convolution


## Many to One: reducing the time dimensions

They are many different ways to reduce map a temporel sequence of embeddings $\{X_1, \ldots X_{T_x}\}$(Many) to a single embedding $X$ (One).

Such a mechanism can be necessary in order to map the temporel embedding provided by the last layer of a network to a single ground-truth (such as in auto-tagging, where the whole track is from a given genre, or in Acoustic Scene Classification).

The most simple way to achieve this is to use the Mean/Average value (Average Pooling) or Maximum value (Max Pooling) of the $X_t$ over time (as done for example in {cite}`Dieleman2014Spotify`).

### Attention weighting

Another possibility is to compute a weighted sum of the values $X_t$ where the weights $a_t$ are attention parameters:
$X = \sum_{t=0}^{T_x-1} a_t X_t$

In {cite}`DBLP:conf/ismir/GururaniSL19`, it is proposed to compute these weights $a_t$ either
- by computing a new projection of the $X_t$ and then normalizing them:
	$a_t = \frac{\sigma(v^T h(X_t))}{\sum_{\tau} \sigma(v^T h(X_{\tau}))}$
- doing the same after splitting $X_t$ in two (along the channel dimensions): the first part being used to compute "values", the second to compute "weights"

![brick_attention_instrument](/images/brick_attention_instrument.png)

*image source: {cite}`DBLP:conf/ismir/GururaniSL19`*

```python
class nnSoftmaxWeight(nn.Module):
    """
    Perform attention weighing based on softmax with channel splitting
    Code from https://github.com/furkanyesiler/move
    """
    def __init__(self, nb_channel):
        super().__init__()
        self.nb_channel = nb_channel
    def forward(self, X):
        weights = torch.nn.functional.softmax(X[:, int(self.nb_channel/2):], dim=3)
        X = torch.sum(X[:, :int(self.nb_channel/2)] * weights, dim=3, keepdim=True)
        return X
```

(lab_AutoPoolWeightSplit)=
### Auto-Pool
The above attention mechanism can by combined with the auto-pool operators proposed by {cite}`DBLP:journals/taslp/McFeeSB18`.

The auto-pool operators is defined as $a_t = \frac{\exp(\alpha X_t)}{\sum_{\tau} \exp(\alpha X_{\tau})}$

It uses a parameter $\alpha$ which allows to range from $\alpha=0$ (unweighted, a.k.a. average pooling), $\alpha=1$ (softmax weighted mean), $\alpha=\infty$: (a.k.a. max pooling).
The $\alpha$ parameters is a trainable parameters (optimized using SGD).

![brick_autopool](/images/brick_autopool.png)

*image source: {cite}`DBLP:journals/taslp/McFeeSB18`*

```python
# Code: https://github.com/furkanyesiler/move
def f_autopool_weights(data, autopool_param):
    """
    Calculating the autopool weights for a given tensor
    :param data: tensor for calculating the softmax weights with autopool
    :return: softmax weights with autopool

    see https://arxiv.org/pdf/1804.10070
    alpha=0: unweighted mean
    alpha=1: softmax
    alpha=inf: max-pooling
    """
    # --- x: (batch, 256, 1, T)
    x = data * autopool_param
    # --- max_values: (batch, 256, 1, 1)
    max_values = torch.max(x, dim=3, keepdim=True).values
    # --- softmax (batch, 256, 1, T)
    softmax = torch.exp(x - max_values)
    # --- weights (batch, 256, 1, T)
    weights = softmax / torch.sum(softmax, dim=3, keepdim=True)
    return weights
```

### Using models

It is also possible to use a **RNN/LSTM in Many-to-One configuration** (only the last hidden state $X_{T_x}$ is mapped to an output $\hat{y}$).

Finally it is possible to add an extra CLASS token to a Transformer architecture.

![brick_pooling](/images/brick_pooling_P.png)



It should be noted that the term "Attention" encapsulates a large set of different paradigms.
- In the **encode-decoder architecture** {cite}`DBLP:journals/corr/BahdanauCB14` it is used during decoding to define the correct context $c(\tau)$ to be used to generate the hidden state $s(\tau)$. For this it compares $s(\tau-1)$ to all the hidden state of the encoder $a(t)$.
- In the **transformer architecture** {cite}`DBLP:conf/nips/VaswaniSPUJGKP17` it is used to compute a self-attention. For this, the $x(t)$ are mapped (using matrix projections) to query $q(t)$, key $k(t)$ and values $v(t)$. A given $q(\tau)$ is then compared to all $k(t)$ to compute attention weights $a(t,\tau)$ which are used in the weighting sum of the $v(t)$:
$e(\tau) = \sum_t a(t,\tau) v(t)$.


## Recurrent Architectures

(lab_rnn)=
### RNN

**Recurrent Neural Networks (RNNs)** are a type of neural network designed to work with <mark>sequential data</mark> (e.g., time series, text, etc.).
They "remember" information from previous inputs by using hidden states, which allows them to model dependencies across time steps.

Their generic formulation for inputs $x^{<t>}$ over time is:

$$a^{<t>} = tanh (W_{aa} a^{<t-1>} + W_{ax} x^{<t>}+ b_a)$$

where $a^{<t>}$ is the hidden state of the RNN at time $t$.

A <mark>bi-directional-RNN</mark>, read the data in both directions (left-to-right and right-to-left).
The goal is to make $a^{<t>}$ both dependent on $a^{<t-1>}$ and $a^{<t+1>}$.

Two configurations are often used with RNNs:
- <mark>Many-to-many</mark>: RNN can be used to model the evaluation over time of features (such as done in the past with Kalman filters or HMM).
They are often used to represent a Language model.
- <mark>Many-to-one</mark>: One can also use the last hidden state of a RNN $a^{<T_x>}$ where $T_x$ is the length of the input sequence, to sum up the content of the input sequence (see picture below).

![brick_rnn](/images/brick_rnn.png)

```python
torch.nn.RNN(input_size, hidden_size, num_layers=1, bidirectional=False)
```

*image source: [Link](https://www.researchgate.net/figure/The-four-types-of-recurrent-neural-network-architectures-a-univariate-many-to-one_fig3_317192370)*

### LSTM
**Long Short-Term Memory (LSTM)**  are a specialized type of RNN designed to handle long-term dependencies more effectively.
LSTM use a more complex architecture with a memory $c_t$ over time $t$, a hidden value $h_t$ and a set of gates (input gate, forget gate, and output gate) to control the flow of information
between the input $x_t$, the previous hidden state $h_{t-1}$ and memory $c_{t-1}$ and their new values.
This allows them to retain relevant information over longer sequences while "forgetting" irrelevant information.

As RNN, two configurations are often used with LSTMs:
- <mark>Many-to-many</mark>
- <mark>Many-to-one</mark>

![brick_rnn](/images/brick_lstm.png)

*image source: [Link](https://mlarchive.com/deep-learning/understanding-long-short-term-memory-networks/)*

```python
torch.nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=False)
```

### Transformer/ Self-Attention
