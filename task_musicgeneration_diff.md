# Generation with Diffusion Models

<!-- Diffusion models offer another approach to musical audio generation. They transform random noise into meaningful continuous audio representations. -->

In the [notebook](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Generation_Diffusion.ipynb) we implement a diffusion model that adopts a **Rectified Flow** method with **ODE-based sampling**. 
This approach combines elements from both **denoising diffusion probabilistic models (DDPMs)** and **normalizing flows**, resulting in a continuous-time framework for generative modeling.

---

## **Overview of the Method**

1. **Noise Addition via Linear Interpolation**: The model adds noise to the data by linearly interpolating between the clean data and pure noise, controlled by a time variable $t$.

2. **Training Objective**: The model is trained to predict the residual $v = y - \text{noise}$ from the noisy samples and the time $t$. This residual guides the denoising process.

3. **ODE-Based Sampling**: During inference, the model uses an ODE solver to integrate over time from $t = 1$ (pure noise) to $t = 0$ (clean data), effectively reversing the noise addition.

---

## **Detailed Breakdown**

### **1. Diffusion Model Architecture**

The `DiffusionUnet` class defines a [classic U-Net architecture](lab_unet) with time conditioning. 
The time embedding (`self.time_mlp`) is a multi-layer perceptron (MLP) that embeds the time variable $t$ into a higher-dimensional space to condition the model on the diffusion time step.
It is added to the deepest layer, allowing the model to adjust its predictions based on the amount of noise present.

   ```python
   def forward(self, x, t):
       # ...
       
       # Bottleneck
       h = self.bottleneck(h)

       # Time embedding
       t_emb = self.time_mlp(t.unsqueeze(-1))  # [batch_size, channels]
       t_emb = t_emb.unsqueeze(-1)             # [batch_size, channels, 1]
       h = h + t_emb                           # Broadcast addition
       # ... 
   ```

### **2. Noise Addition and Training Objective**

The `RectifiedFlows` class handles the noise addition and defines the training loss:

- **Noise Addition (`add_noise` method)**:

  ```python
  def add_noise(self, x, noise, times):
      return (1. - times) * x + times * noise
  ```

  `add_noise` performs a linear interpolation between clean data `x` and random noise `noise` based on the time variable `times`.

- **Time Variable Sampling**:

  ```python
  times = torch.nn.functional.sigmoid(
      torch.randn(y.shape[0]) * self.P_std + self.P_mean
  )
  ```

  The sigmoid non-linearity ensures `times` lies between 0 and 1.

- **Training Objective (`forward` method)**:

  ```python
  def forward(self, model, y, sigma=None, return_loss=True, **model_kwargs):
      # ...
      noises = torch.randn_like(y)
      v = y - noises
      noisy_samples = self.add_noise(y, noises, times)
      fv = model(noisy_samples, times, **model_kwargs)
      loss = mse(v, fv)
      # ...
  ```
The model calculates the residual by subtracting the noise from the data, expressed as $v = y - \text{noise}$. 
It then predicts $fv$, an approximation of the residual $v$, based on the noisy samples and the corresponding time step. 
The loss function used is the Mean Squared Error (MSE) between the true residual $v$ and the predicted residual $fv$, which is given by $\text{Loss} = \| v - \text{fv} \|^2$. 
Predicting the residual instead of the noise or the data directly is a characteristic of the **Rectified Flow** method, while in **Denoising Diffusion Probabilistic Models (DDPMs)**, the model typically predicts the noise. 

### **3. Inference and Sampling Process (`inference` function)**

During inference, the model generates new samples by solving an ODE:

```python
def inference(rectified_flows, net, latents_shape, num_steps):
    # Initialize with pure noise
    current_sample = torch.randn(latents_shape)
    times = torch.ones(latents_shape[0])
    # Integrate over time
    for i in range(num_steps):
        v = net(current_sample, times)
        current_sample = current_sample + step_size * v
        times = times - step_size
    return current_sample / sigma_data
```

- **Initialization**:

  - **`current_sample`**: Starts as pure noise.
  
  - **`times`**: Begins at $t = 1$, representing the highest noise level.

- **Integration Loop**:

  - **Time Step (`step_size`)**: Calculated as $\Delta t = \frac{1}{\text{num\_steps}}$.

  - **Euler's Method**: Updates the sample by moving in the direction of `v` predicted by the model:

    $
    \text{current\_sample} = \text{current\_sample} + \Delta t \cdot v
    $

  - **Time Update**:

    $
    t = t - \Delta t
    $

- **Result**: After integrating from $t = 1$ to $t = 0$, the `current_sample` approximates a sample from the data distribution.

### **4. Method: Rectified Flow with ODE-Based Sampling**

- **Continuous-Time Framework**: The model operates in continuous time, treating the diffusion process as an ODE.

- **Rectified Flow Concept**:

  - **Unification**: Bridges the gap between normalizing flows (which transform data through invertible mappings) and diffusion models (which add and remove noise).

  - **Deterministic Sampling**: Unlike stochastic diffusion processes, this method uses deterministic integration, which can be more efficient and stable.

- **Model Training**:

  - **Predicting Residuals**: Instead of predicting the noise or the data directly, the model predicts the residual $v = y - \text{noise}$, guiding the denoising process.

- **Advantages**:

  - **Efficiency**: ODE-based sampling can require fewer steps compared to traditional diffusion models.

  - **Stability**: Deterministic sampling paths reduce variance in the generated samples.

---

## **Conclusion**

The code adopts a **Rectified Flow** method that:

- **Trains the model to predict residuals** between the data and noise.

- **Uses ODE-based deterministic sampling** to generate new data samples.

- **Incorporates a time-conditioned U-Net architecture** to handle varying noise levels.


