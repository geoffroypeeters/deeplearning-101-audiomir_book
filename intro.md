# Deep Learning 101 for Audio-based MIR

```{tableofcontents}
```


This is a web book written for a tutorial session of the [25th International Society for Music Information Retrieval Conference](https://ismir2024.ismir.net/), Nov 10-14, 2024 in San Francisco, CA, USA. The [ISMIR conference](https://ismir.net/) is the world’s leading research forum on processing, searching, organising and accessing music-related data.

## Scope

*Audio-based MIR (MIR based on the processing of audio signals) covers a broad range of tasks, including analysis (pitch, chord, beats, tagging), similarity/cover identification, and processing/generation of samples or music fragments. A wide range of techniques can be employed for solving each of these tasks, spanning from conventional signal processing and machine learning algorithms to the whole zoo of deep learning techniques.*

*This tutorial aims to review the various elements of this deep learning zoo commonly applied in Audio-based MIR tasks. We review typical audio front-ends (such as waveform, Log-Mel-Spectrogram, HCQT, SincNet, LEAF, quantization using VQ-VAE, RVQ), as well as projections (including 1D-Conv, 2D-Conv, Dilated-Conv, TCN, WaveNet, RNN, Transformer, Conformer, U-Net, VAE), and examine the various training paradigms (such as supervised, self-supervised, metric-learning, adversarial, encoder-decoder, diffusion).*

*Rather than providing an exhaustive list of all of these elements, we illustrate their use within a subset of (commonly studied) Audio-based MIR tasks such as multi-pitch/chord-estimation, cover-detection, auto-tagging, source separation, music-translation or music generation. This subset of Audio-based MIR tasks is designed to encompass a wide range of deep learning elements.*

*For each tack we address a) the goal of the tasks, b) how it is evaluated, c) provide some popular datasets to train a system, and d) explain (using slides and pytorch code) how we can solve it using deep learning.*

*The objective is to provide a 101 lecture (introductory lecture) on deep learning techniques for Audio-based MIR. It does not aim at being exhaustive in terms of Audio-based MIR tasks neither on deep learning techniques but to provide an overview for newcomers to Audio-Based MIR on how to solve the most common tasks using deep learning. It will provide a portfolio of codes (Colab notebooks and Jupyter book) to help newcomers achieve the various Audio-based MIR Tasks.*

## About the authors

[Geoffroy Peeters](https://perso.telecom-paristech.fr/gpeeters/) is a full professor in the Image-Data-Signal (IDS) department of [Télécom Paris](https://www.telecom-paris.fr/). Before that (from 2001 to 2018), he was Senior Researcher at IRCAM, leading research related to Music Information Retrieval. He received his Ph.D. in signal processing for speech processing in 2001 and his Habilitation (HDR) in Music Information Retrieval in 2013 from the University Paris VI. His research topics concern signal processing and machine learning (including deep learning) for audio processing, with a strong focus on music. He has participated in many national or European projects, published numerous articles and several patents in these areas, and co-authored the ISO MPEG-7 audio standard. He has been co-general-chair of the DAFx-2011 and ISMIR-2018 conferences, member and president of the ISMIR society, and is the current AASP review chair for ICASSP.

[Gabriel Meseguer-Brocal](https://github.com/gabolsgabs) is a research scientist at [Deezer](https://research.deezer.com/) with over two years of experience at the company. Before joining Deezer, he completed postdoctoral research at Centre National de la Recherche Scientifique (CNRS) in France. In 2020, he earned his Ph.D. in Computer Science, Telecommunications, and Electronics with a focus on the Sciences \& Technologies of Music and Sound at IRCAM. His research interests include signal processing and deep learning techniques for music processing, with a focus on areas such as source separation, dataset creation, multi-tagging, self-supervised learning, and multimodal analysis.

[Alain Riou](https://github.com/aRI0U) is a PhD student working on self-supervised learning of musical representations at Télécom-Paris and Sony CSL Paris, under the supervision of Stefan Lattner, Gaëtan Hadjeres and Geoffroy Peeters.
Before that, he obtained a master degree in mathematics for machine learning at Ecole Normale Supérieure de Cachan (2020) and another one in signal processing and computer science applied to music at IRCAM (2021).
His main research interests are related to deep representation learning, with a strong focus on self-supervised methods for music information retrieval and controllable music generation.
His work "PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective" received the Best Paper Award at ISMIR 2023.

[Stefan Lattner](https://csl.sony.fr/member/stefan-lattner-phd/) serves as a researcher leader at the music team at [Sony CSL](https://csl.sony.fr) Paris, where he focuses on generative AI for music production, music information retrieval, and computational music perception. He earned his PhD in 2019 from Johannes Kepler University (JKU) in Linz, Austria, following his research at the Austrian Research Institute for Artificial Intelligence in Vienna and the Institute of Computational Perception Linz. His studies centered on the modeling of musical structure, encompassing transformation learning and computational relative pitch perception.
His current interests include human-computer interaction in music creation, live staging, and information theory in music.
He specializes in generative sequence models, computational short-term memories, (self-supervised) representation learning and musical audio generation. In 2019, Lattner received the best paper award at ISMIR for his work, “Learning Complex Basis Functions for Invariant Representations of Audio.”

blablabla{cite}`holdgraf_evidence_2014`


## Citing this book¶

```
@book{musicclassification:book,
    Author = {Peeters, Geoffroy and Meseguer-Brocal, Gabriel and Riou, Alain and Lattner, Stefan},
    Month = Nov.,
    Publisher = {https://music-classification.github.io/tutorial},
    Title = {Deep Learning 101 for Audio-based MIR},
    Year = 2024,
    Url = {https://music-classification.github.io/tutorial},
    doi = {10.5281/zenodo.5703779}
}
```

```{bibliography}
```
