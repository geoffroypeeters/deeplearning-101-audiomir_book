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
