## Projections

### 2D-Conv

### 1D-Conv

### Dilated-Conv

### Dephtwise Separable Convolution

```python
class depthwise_conv(nn.Module):

    def __init__(self, nin, kernels_per_layer):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)

    def forward(self, x):
        out = self.depthwise(x)
        return out

class pointwise_conv(nn.Module):

    def __init__(self, nin, nout):
        super(pointwise_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.pointwise(x)
        return out

class depthwise_separable_conv(nn.Module):

    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

model = depthwise_separable_conv(16, 1, 32)
X = torch.randn(2,16,23,23)
model(X).size()
```

### WaveNet

WaveNet was proposed in {cite}`DBLP:conf/ssw/OordDZSVGKSK16`

https://github.com/facebookresearch/music-translation/blob/main/src/wavenet.py


### Temporal Convolution Network (TCN)

LEAF (Learnable Audio Front-End) was proposed in {cite}`DBLP:journals/corr/abs-1803-01271`

![tcn](/images/brick_tcn.png)

https://github.com/paul-krug/pytorch-tcn

https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks
