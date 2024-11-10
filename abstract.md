# Abstract

**Context.** Audio-based MIR (MIR based on the processing of audio signals) covers a broad range of tasks, including
- music audio <mark>analysis</mark> (pitch/chord, beats, tagging), retrieval (similarity, cover, fingerprint),
- music audio <mark>processing</mark> (source separation, music translation)
- music audio <mark>generation</mark> (of samples or whole tracks).

A wide range of techniques can be employed for solving each of these tasks, spanning
- from conventional <mark>signal processing</mark> and <mark>machine learning</mark> algorithms
- to the whole zoo of <mark>deep learning techniques</mark>.





**Objective.** This tutorial aims to review the various elements of this deep learning zoo which are commonly applied in Audio-based MIR tasks.
We review typical
- <mark>inputs</mark>: such as [waveform](lab_waveform), [Log-Mel-Spectrogram](lab_lms), [CQT](lab_cqt), [HCQT](lab_hcqt), [Chroma](lab_chroma)
- <mark>front-ends</mark>: such as [Dilated-Conv](lab_dilated), [TCN](lab_tcn), [SincNet](lab_sincnet)
- <mark>projections</mark>: such as [1D-Conv](lab_conv1d), [2D-Conv](lab_conv2d), [U-Net](lab_unet), [RNN](lab_rnn), [LSTM](lab_lstm), [Transformer](lab_transformer)
- <mark>bottleneck</mark>: AE, VAE quantization using VQ-VAE, RVQ
- <mark>training paradigms</mark>: such as [supervised](lab_supervised), [unsupervised](lab_usl), [encoder-decoder](lab_encoder_decoder), [self-supervised](lab_ssl), [metric-learning](lab_metric_learning), adversarial, denoising/latent diffusion





**Method.** Rather than providing an exhaustive list of all of these elements, we illustrate their use within a subset of (commonly studied) <mark>Audio-based MIR tasks</mark> such as
- <mark>analysis</mark>: [multi-pitch](lab_multi_pitch), [cover-detection](lab_cover_detection), [auto-tagging](lab_auto_tagging),
- <mark>processing</mark>: [source separation](lab_source_separation)
- <mark>generation</mark>: [auto-regressive/LLM](lab_ex_autoregressive), [diffusion](lab_ex_diffusion)

This subset of Audio-based MIR tasks is designed to encompass a wide range of deep learning elements.





*The objective is to provide a 101 lecture (introductory lecture) on deep learning techniques for Audio-based MIR. It does not aim at being exhaustive in terms of Audio-based MIR tasks neither on deep learning techniques but to provide an overview for newcomers to Audio-Based MIR on how to solve the most common tasks using deep learning. It will provide a portfolio of codes (Colab notebooks and Jupyter book) to help newcomers achieve the various Audio-based MIR Tasks.*


*This tutorial can be considered  as a follow-up of the tutorial ["Deep Learning for MIR" ](https://github.com/keunwoochoi/dl4mir) by Alexander Schindler, Thomas Lidy and Sebastian Böck, held at ISMIR-2018.*
