# Source Separation

- Get the slides in [Here](https://docs.google.com/presentation/d/1wx7UlnwGMKhnByI1FPjTR5Vtj8WY4y8rvRUtfyCOpDk/edit?usp=sharing)

- Test the NOTEBOOK in [Here](https://colab.research.google.com/github/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Source_Separation.ipynb)

## Goal of the Task

Music source separation is the process of isolating individual musical sources, such as vocals, drums, bass, and other instruments, from a mixed music track. This task is distinct from general source separation due to the unique challenges presented by musical compositions. For example, in a band setting, isolating a single instrument, like the bass guitar, allows musicians to practice along, or extracting vocals enables karaoke applications.
Formally:

$$
y(t) = \displaystyle\sum_{i=1}^{N} x_i(t).
$$

where  $y(t)$ is composed of $N$ sources $x_n(t)$, for $n=1...N$.

The underlying challenge in music source separation is that musical signals are highly correlated, meaning multiple sources, like instruments, often change in harmony or in response to rhythmic patterns in the track. Moreover, music recordings undergo complex, non-linear processing during mixing, where reverb, filters, and other effects alter each instrument's natural characteristics. As a result, music source separation is often an underdetermined problem, where the number of sources exceeds the available observed mixture channels, making it mathematically complex to isolate each element independently.

This tutorial will outline the main characteristics of music source separation, providing a foundation for further exploration of open-source tools and datasets. For an in-depth review of available resources and methods, see the["Open Source Tools & Data for Music Source Separation"](https://source-separation.github.io/tutorial/landing.html).

Additionally, we introduce the concept of conditional learning, an approach in which input $x$ d is processed differently based on an external context $z$. This enables a single model to adapt its behavior dynamically, allowing the separation process to respond flexibly to diverse condition.

## Popular Datasets

- **MUSDB**: The MUSDB dataset includes 100 tracks for training and 50 for testing, each with four stems: drums, bass, vocals, and others. Covering approximately 10 hours of music across multiple genres, including 46 tracks from MedleyDB, it provides a foundational dataset for experiments in music source separation.

- **MedleyDB**: MedleyDB offers 179 full-length tracks, each about 3 to 5 minutes long, in a multitrack format with up to 17 individual stems. Spanning approximately 12 hours, the dataset includes a wide range of genres such as Singer/Songwriter, Classical, Rock, World/Folk, Fusion, Jazz, Pop, Musical Theatre, and Rap, making it valuable for tasks requiring genre diversity and high-quality source separation.

- **MoisesDB**: MoisesDB contains 240 tracks with 11 unique stems, including Bass, Bowed Strings, Drums, Guitar, Other, Other Keys, Other Plucked, Percussion, Piano, Vocals, and Wind instruments. Comprising approximately 14 hours of music from 47 artists across twelve genres, MoisesDB is especially useful for complex separation tasks requiring a detailed breakdown of diverse instrument types.

## How is the Task Evaluated?

Three commonly used metrics for evaluating music source separation are Source-to-Distortion Ratio (SDR), Source-to-Interference Ratio (SIR), and Source-to-Artifact Ratio (SAR). These metrics assess the quality of a system's output by analyzing how well it isolates each target source from undesired elements.

Given an estimate of a source $\hat{s}_i$, it can be decomposed as follows:

$\hat{s}_i = s_{\text{target}} + e_{\text{interf}} + e_{\text{noise}} + e_{\text{artif}}$

where:

- $s_{\text{target}}$ is the true source component,
- $e_{\text{interf}}$ represents interference from other sources,
- $e_{\text{noise}}$ is the noise, and
- $e_{\text{artif}}$ accounts for artifacts introduced by the separation system

Using these components, we can calculate the three evaluation metrics:

- **Source-to-Artifact Ratio (SAR)**: This metric quantifies the level of unwanted artifacts in the estimated source relative to the true source. A high SAR value indicates fewer artifacts. It represents the algorithmic artifacts of the process.

$$\text{SAR} := 10 \log_{10} \left( \frac{\| s_{\text{target}} + e_{\text{interf}} + e_{\text{noise}} \|^2}{ \| e_{\text{artif}} \|^2} \right)
$$

- **Source-to-Interference Ratio (SIR)**: SIR measures the amount of interference from other sources in the estimate. This metric is helpful for understanding the extent of “bleed” or “leakage” from other instruments. It represents the interference in the isolation from other sources.

$$ \text{SIR} := 10 \log_{10} \left( \frac{\| s_{\text{target}} \|^2}{ \| e_{\text{interf}} \|^2} \right)
 $$

- **Source-to-Distortion Ratio (SDR)**: SDR provides an overall measure of the estimate's quality by comparing the true source to the combined distortions (interference, noise, and artifacts). Higher SDR values indicate a better overall quality of separation, and it is often reported as the primary performance measure. It represents the overall performance of the separation.

$$ \text{SDR} := 10 \log_{10} \left( \frac{\| s_{\text{target}} \|^2}{ \| e_{\text{interf}} + e_{\text{noise}} + e_{\text{artif}} \|^2} \right)
 $$

All three metrics are calculated in decibels (dB), with higher values indicating better performance.
For instance, if SDR is 1 dB better, then the distortion is 1 dB less (target is constant) 3dB means the “distortions” are two times more quiet. They require access to ground truth isolated sources and are computed over short, windowed segments of the signal for finer temporal accuracy.

### Discussion

Coming shortly

## Models

Coming shortly

## Losses

Coming shortly

## Conditioning

Coming shortly
