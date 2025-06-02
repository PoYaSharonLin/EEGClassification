# EEGClassification

This project built upon the CNN-based model forked by the original author. The goal of this project is to dive into how different artifact removal methods affect the ICA labeling result and the CNN mdoel accuracy. 

This is a class project as part of 11320ISA557300 - Brain Computer Interfaces: Fundamentals and Application @ National Tsing Hua University.  

- [EEGClassification](#eeg-classification)
  * [1. Introduction](#introduction)
  * [2. Data Description](#files-in-the-repository)
  * [3. ICA & ICA Labeling Table](#introduction)
  * [4. Code](#code)
  * [5. Model Framework](#proposed-cnn-model)
  * [6. Validation & Visualization](#validation)
  * [7. Usage](#usage)
  * [7. Presentation](#presentation)
  * [8. References](#references)


## 1. Introduction
The MNIST database ( Modified National Institute of Standards and Technology database ) is a large database of handwritten digits (0-9) that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning.


MindBigData (or the Brain MNIST) aims to provide a comprehensive and updated dataset of brain signals related to a diverse set of human activities so it can inspire the use of machine learning algorithms as a benchmark of 'decoding' performance from raw brain activities into its corresponding (labels) mental (or physical) tasks.

![path](https://github.com/poyasharonlin/EEGClassification/blob/updated-CNN/images/path.PNG)


## 2. Dataset Description

### Dataset Aquisition

[The Visual "MNIST" of Brain Digits](https://www.mindbigdata.com/opendb/visualmnist.html) is an open dataset that provided by David Vivancos and Felix Cuesta. Among all the different versions, We are using the [MindBigData2022_MNIST_EP (3GB)](https://huggingface.co/datasets/DavidVivancos/MindBigData2022_MNIST_EP/viewer/default/train). 

----
### Experiment Paradigm & Data Collection
The experiments is an Event Related Paradigm (ERP) where the subjectee is exposed to visual stimuli (digits) and then followed by a cognitive processing (thinking about the digit).

The each trial of the brain signal is captured within 4 seconds, with 2 seconds of exposure to visual stimuli and 2 seconds of black screen to ponder over the digit. The channels used in the experiment are "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", and "AF4". 

![intro](https://github.com/poyasharonlin/EEGClassification/blob/updated-CNN/images/intro.PNG)

----
### Dataset Size 
Sampling rate: 128 Hz

Data Points: 128 * 2(secs) = 256 

Total Sample Size: 64,302

| Name         | Percentage | Size  |
|----------------------|------|-----------|
| Training Set | 75% | 48,226 |
| Validation Set | 15% | 9,646 |
| Testing Set| 10% | 6,430 |

----

### Hardware & Software
#### Hardware
The author used a customer cap 64 for collecting the data and with a Morlet wavelet transform scalogram PNG image. The distribution of the channels follows the standard 10-20 system and is displayed as below. 

![10-20](https://github.com/poyasharonlin/EEGClassification/blob/updated-CNN/images/10-20.PNG) 

#### Software
All the python packages required are mentioned in the `pip install` seciton. Please note that we have a version constraints on the following packages to avoid version conflict. 

```python
!pip install numpy==2.0.0
!pip install scipy>=1.14
```

----


## 3. ICA & ICA Labeling Table
| Pre-processing                                  |   |   | Numbers of ICs classified by ICLabel         |   |   |   |   |   |   |
|-------------------------------------------------|---|---|-----------------------------------------------|---|---|---|---|---|---|
| EEG (? Channels & ? Datasets)                   | bandpass filter | ASR | Brain | Muscle | Eye | Heart | Line Noise | Channel Noise | Other |
| raw                                             |    |    |   4    |     0   |   0  |   1    |      1      |      1        |    7   | 
| filtered                                        | v  |    |       |        |     |       |            |              |       |
| ASR-corrected                                   | v  | v  |       |        |     |       |            |              |       |



## 4. Code
|File name         | Purpsoe |
|----------------------|------|
|`EEGClassification.ipynb`| main file - Jupiter Notebook format|
|`utils.py`| utils functions for loading and pre-processing the data|
|`images`| Images used for preview in README.md file|
|`model_raw.pth`| trained CNN without filtering hyperparameters using pytorch|



## 5. Model Framework
The network is composed of five convolution blocks and fully connected layers. Each convolution block consists of a convolution layer, a batch normalization, and an exponential linear unit, as shown in the following figure. 
An illustration of the proposed network is shown below:

![alt text](https://github.com/NitzanShitrit/EEGClassification/blob/main/images/cnn.PNG)

C1 and C2 blocks were designed to extract the spectral representation of the EEG input, as it performs convolution across the time dimension, capturing features from each EEG channel independently from the others.

C3 block was designed for performing spatial filtering, as it performs convolutions across the channel dimension. The objective of this layer is to learn the weights of all channels at each time sample.

C4 and C5 blocks are capturing the temporal patterns in each extracted feature maps.  
Dropout has been used in deep neural network training as a regularisation technique to reduce the network tendency to overfit during the training process. 
Therefore, we used it after convolution blocks C3 and C5, with the dropout set to 0.5.


## 6. Validation & Visualization


In the validation 


### Validation 


### Visualization
In the experiment, there are 3 waves that we would like to examine, alpha, beta, and theta. Alpha waves is significant sign of eye blinking. Beta is primary for capturing cognitive activity(recognizing the digits), whereas theta waves records memory and imagery-related activity (imagine the digits). 

1. EEG signal 
For raw data, we expect the EEG signal to be relatively noisy and contains some spikes compared to that of the one after bandpass filtering and ASR. 
![EEG](https://github.com/poyasharonlin/EEGClassification/blob/updated-CNN/images/EEG.PNG) 
![FFT](https://github.com/poyasharonlin/EEGClassification/blob/updated-CNN/images/FFT.PNG)


2. Alpha, Beta, Theta SNR 
![Alpha-SNR](https://github.com/poyasharonlin/EEGClassification/blob/updated-CNN/images/Alpha_SNR.PNG)
![Beta-SNR](https://github.com/poyasharonlin/EEGClassification/blob/updated-CNN/images/Beta_SNR.PNG)
![Theta-SNR](https://github.com/poyasharonlin/EEGClassification/blob/updated-CNN/images/Theta_SNR.PNG)


3. Scalp Plot 
\

![Alpha-scalp](https://github.com/poyasharonlin/EEGClassification/blob/updated-CNN/images/Alpha_scalp.PNG)

![Beta-scalp](https://github.com/poyasharonlin/EEGClassification/blob/updated-CNN/images/Beta_scalp.PNG)

![Theta-scalp](https://github.com/poyasharonlin/EEGClassification/blob/updated-CNN/images/Theta_scalp.PNG)
----
### Usage 
1. The findings can be an support of the importance of performing artifact removal on EEG data. 
2. Provide a CNN modern version that can be built upon
3. Provide EEG, FFT, Scalp plot for visualization. 



## 7. Presentation
Streamlit App (TBD)



## 8. References
*	Mindbigdata dataset. http://www.mindbigdata.com/ (2018)
*	Henry, J. Craig. "Electroencephalography: basic principles, clinical applications, and related fields." Neurology 67.11 (2006): 2092-2092. 
*	Bird, Jordan J., et al. "A deep evolutionary approach to bioinspired classifier optimisation for brain-machine interaction." Complexity 2019 (2019).
*	Jolly, Baani Leen Kaur, et al. "Universal EEG encoder for learning diverse intelligent tasks." 2019 IEEE Fifth International Conference on Multimedia Big Data (BigMM). IEEE, 2019. 
*	Khok, Hong Jing, Victor Teck Chang Koh, and Cuntai Guan. "Deep Multi-Task Learning for SSVEP Detection and Visual Response Mapping." 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC). IEEE, 2020.
*	Bozal Chaves, Alberto. Personalized image classification from EEG signals using Deep Learning. BS thesis. Universitat Politècnica de Catalunya, 2017.
*	Kwon, Yea-Hoon, Sae-Byuk Shin, and Shin-Dug Kim. "Electroencephalography based fusion two-dimensional (2D)-convolution neural networks (CNN) model for emotion recognition system." Sensors 18.5 (2018): 1383.
*	Ha, Kwon-Woo, and Jin-Woo Jeong. "Motor imagery EEG classification using capsule networks." Sensors 19.13 (2019): 2854.
*	Aznan, Nik Khadijah Nik, et al. "Simulating brain signals: Creating synthetic eeg data via neural-based generative models for improved ssvep classification." 2019 International Joint Conference on Neural Networks (IJCNN). IEEE, 2019.
*	Manor, Ran, and Amir B. Geva. "Convolutional neural network for multi-category rapid serial visual presentation BCI." Frontiers in computational neuroscience 9 (2015): 146.
