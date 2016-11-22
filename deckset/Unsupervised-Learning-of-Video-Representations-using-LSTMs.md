footer: © Junho Cho, 2016
slidenumbers: true


# [fit] Unsupervised Learning of
# [fit] Video Representations using LSTMs

#### **University of Toronoto**
#### *Nitish Srivastava*  
#### *Elman Mansimov*
#### *Ruslan Salakhutdinov*

## ICML2015

---

![fit](http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000001.gif)

---

# [fit] Long-term Future Prediction results from this paper

![inline](http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000000.gif)  ![](http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000002.gif)

![inline](http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000003.gif) ![](http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000004.gif) ![](http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000005.gif)

![inline](https://camo.githubusercontent.com/095e967778362ac84862cdf737623044d2175527/687474703a2f2f692e67697068792e636f6d2f7854373758527058676d6a4d7a527a7853672e676966) ![inline](https://camo.githubusercontent.com/9eab37838fb0346dc50a9fc847d8ecf6b4707592/687474703a2f2f692e67697068792e636f6d2f785437375935776270516b305363665865452e676966)

---

# RNN

![fit](https://www.dropbox.com/s/3ieij2zt6i8fksq/%202015-11-15%2012.43.33_zpsbcdry6o9.png?raw=1)

---

# LSTM

![fit](https://www.dropbox.com/s/ky741fihwdmyf9b/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2022.59.54.png?raw=1)

---

# Autoencoder

![fit](https://www.dropbox.com/s/lfe99d4foq9akmn/autoencoder_schema.jpg?raw=1)

---

# Seq2Seq

![fit](https://www.dropbox.com/s/29kxwcr74675dem/687474703a2f2f6936342e74696e797069632e636f6d2f333031333674652e706e67.png?raw=1)

^- Conditioning the decoder can be used to add stochasticity to the model.
- During training de- coder is conditioned on the true frame, at test time is conditioned on the generated frame.

---

# Introduction

- Videos are very high dimensional entity compared to single image.
- Very difficult to credit assignment and learn long range structure

![right](http://i.giphy.com/Af59T5klmASNG.gif)


---
# Introduction

- To keep low dimension
	- unless collect more labelled data
	- or do feature engineering like flow features

- Needs unsupervised learning to find represent structure in videos.

---

# [fit] Lets' utilize temporal structure of videos as
# [fit] a supervisory signal for unsupervised learning.

^Understanding temporal sequences is important for solving many video related problems.
Learning good representations of videos requires a model that captures both spatial and temporal information presented in videos.




---

# [fit] Use the LSTM Encoder-Decoder[^1]
# [fit] to learn good video representations.

![fit](https://camo.githubusercontent.com/599a879884c2cebb9132c5991f626f26d5557732/68747470733a2f2f342e62702e626c6f6773706f742e636f6d2f2d61417253306c31706a48512f566a6a3731704b416145492f41414141414141414178452f4e767931465362445f56732f733634302f325446737461746963677261706869635f616c742d30312e706e67)

[^1]:I.Sutskever, O. Vinyals, and Q. Le. Sequence to sequence learning with neural networks. In NIPS, 2014.

---

# [fit] In this paper, the author proposed three models based on LSTM

- Autoencoder Model
- Future Predictor Model
- Composite Model

---

# 1) LSTM Autoencoder Model:

![right fit ](https://www.dropbox.com/s/v82521taskchdhz/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2022.50.40.png?raw=1)

- This model is composed of two parts, the encoder and the decoder.
- In Autoencoder the target sequence is the same as the input sequence but in reverse order to make optimization easier for model.


---
# AutuEncoder: Intuition

- The reconstruction work requires the network to **capture information about the appearance of objects and the background**

- This is exactly the information that we would like **the representation to contain**.

^Why this should learn good features?
Encoder output : representation of the input video. Should retain information of video.
Decoder : should reconstruct back the input from the representation



---

# 2) LSTM Future Predictor Model:

![right fit](https://www.dropbox.com/s/0e19wte94ghehp6/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2022.53.34.png?raw=1)

 - In Future Predictor, the Decoder LSTM predicts the sequence of frames in video that come just after the input sequence.
 - It also varies with conditional/unconditional versions.


---


# Conditional Decoder

- allows decoder to model multiple modes to target distribution
	- without it, averaging multiple modes in low-level input space
- Frames have strong short-term correlation
	- let decoder easy to pick up correlation
	- unconditional decoder is forced to look for information deep inside the encoder.

![right fit](https://www.dropbox.com/s/y12cxlymccd4kv2/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2001.20.24.png?raw=1)

---
# Future Predctor: Intuition

In order to **predict the next few frames correctly**,

the model needs information about **which objects are present and how they are moving so that the motion can be extrapolated.**

---


# 3) A Composite Model:

![right fit](https://www.dropbox.com/s/ztp167oqbpxfja6/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2022.55.50.png?raw=1)

- The Composite Model combines together the best parts of Autoencoder and Future Predictor to form a more powerful model.
- These two modules share a same encoder, which encodes input sequences into a feature vector and copy them to different decoders.

---

# Overcome each models:

High capacity **autoencoder**
- would memorize the input very well.
- But not useful to predicting Future

**Future predictor** tend to store information only  last few frames
- since last frames are the important
- will forget large part of the

# [fit] Thus remember the *past* and predict the **future**

---

# Composite model: Intuition

Learns representations that contain
 **not only static appearance** of objects & background,
 **but also the dynamic informations** like moving objects and their moving pattern.

---

# Goal of Experiments


- Qualitative understanding what the LSTM learns to do
- Better performance on supervised task with **initializing networks with the weights found by unsupervised learning**, *especially with very few training examples*.
- Compare the three proposed models and conditional variants
- Compare with state-of-the-art action recognition benchmarks.

---

# Results on Moving MNIST

Each MNIST video is 20 frames long and consists of 2 digits moving inside a 64 x 64 patch.

![inline 150%](http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000002.gif)

Occlusions and dynamics of bouncing off the walls

Encoder takes 10 frames as input.
Decoder reconstruct the 10 frames and predict next 10 frames

---

Input: 64 x 64 = 4096

Each LSTM has 2048 units. We used logistic output units with a cross entropy loss function.

![inline](https://www.dropbox.com/s/ztp167oqbpxfja6/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2022.55.50.png?raw=1)

---

![fit](https://www.dropbox.com/s/jshhe0e9jjg8jeo/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2002.13.18.png?raw=1)

---

Three models

![original  fit](https://www.dropbox.com/s/v82521taskchdhz/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2022.50.40.png?raw=1)![ original fit](https://www.dropbox.com/s/0e19wte94ghehp6/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2022.53.34.png?raw=1)![original fit](https://www.dropbox.com/s/ztp167oqbpxfja6/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2022.55.50.png?raw=1)


---


# [fit] Two layer + Conditional (sharper)
# [fit] > Two layer > Single Layer

![inline fit](https://www.dropbox.com/s/jshhe0e9jjg8jeo/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2002.13.18.png?raw=1)

---

# Results on Video Patches

Each video is a randomly cropped 32 x 32 patch from UCF-101 dataset.

16 input frames, reconstruct and predict 13 future frames

---

![fit](https://www.dropbox.com/s/qxlpgdoekpjdp60/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2002.27.22.png?raw=1)

---

### Reconstructions and predictions are both blurry
### Bigger model sharpens reconstruction

---

# Comparison of variants

Measure on future prediciton

![inline](https://www.dropbox.com/s/evd0vl7e062rtnw/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2003.56.04.png?raw=1)

![fit right](https://www.dropbox.com/s/ztp167oqbpxfja6/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2022.55.50.png?raw=1)

---
# Generalization over time

Would it generalize after 10 frames?
one hidden layer unconditioned Composite model

Trained for 10 future frames
Test for 100 more frames

![right fit](http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000006.gif)

---

# Shwos Periodic Pattern of activity
 in 200 randomly chosen LSTM units in Future predictor

![fit inline](https://www.dropbox.com/s/1a55w1niqkafl2m/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2002.40.08.png?raw=1)

Able to generate persistent motion for longer time

---

Periodic behavior is not trivial
Also tried Random init future predictor

![fit inline](https://www.dropbox.com/s/3q3s7dxi1yk0pek/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2002.46.59.png?raw=1)

Cell state quickly converges and outputs blur completely

---

# Out-of-domain

![inline](https://www.dropbox.com/s/6a1y2pq36jgfu55/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2002.51.09.png?raw=1)

Try on One and Tree moving digits.
Drawback of modeling input.

---

# [fit] Visualizing Features

## [fit] 4 sets of weights that connect Input frame to Encoder LSTM

## [fit] Top-200 features ordered by L2 norm

![](http://colah.github.io/images/post-covers/lstm.png)

---

![fit](https://www.dropbox.com/s/31pawwpeym6eexh/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2005.34.11.png?raw=1)
![fit](https://www.dropbox.com/s/61x8h5vbho355ua/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2005.34.29.png?raw=1)

---

## high frequency strips might encode direction and velocity of motion

![original fit](https://www.dropbox.com/s/l5twigjcwhfrd9m/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2002.59.54.png?raw=1)




---

# output Features from Decoder

more blob and shorter strip might encode ??

![inline](https://www.dropbox.com/s/e7mb8gcia5dom6m/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2003.11.15.png?raw=1)

---

# Apply on Action recognition

## Features learned by unsupervised learning are good initialization?

![ left](https://media.giphy.com/media/3oriNYQX2lC6dfW2Ji/source.gif)

---

# Dataset - unsupervised

Sports-1M dataset (Karpathy et al., 2014),
Did not use label

Collected 300 hour randomly sampling 10 sec

Extracted feature with `VGG fc6 4096-dim`

Trained **two layer Composite model with 2048 hidden units**
- **Autoencode** 16 input seq, **predict** next 13 frames


---

# Dataset - supervised tasks

![fit right](http://www.iri.upc.edu/people/shusain/gifs/ucf101/img491.gif)

**UCF-101 (Soomro et al., 2012)**
- 13,320 videos
- Average length of 6.2 sec
- 101 different action categories.

**HMDB-51 (Kuehne et al., 2011)**
 - 5100 videos
 - Mean length of the videos is 3.2 seconds.
 - 51 different action categories.


---


# LSTM classifier

- The output from each LSTM goes into a softmax classifier
- Predict action at each time step and then gets averaged to make final prediction.


- **Baseline**: with randomly initialized
- **Composite model**: weights from unsupervised learning

![fit right](https://www.dropbox.com/s/o4rq12q3vetrtab/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2023.22.23.png?raw=1)

---

# Composite LSTM Model

- Pretrained on subset of Sports-1M [^2] dataset.
- Extracted **fc6 4096 features** (of image frames and optical flow[^4]) by frames through VGG ConvNet[^3] trained on ImageNet.
- Trained on sequences of **fc6 4096 features**.


[^2]: A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and L. Fei-Fei. Large-scale video classification with convolutional neural networks. In CVPR, 2014.

[^4]: K. Simonyan and A. Zisserman. Two-stream convolutional networks for action recognition in videos. In Advances in Neural Information Processing Systems, 2014.

[^3]: K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.


---



# Baseline and Composite model

UCF-101 and HMDB-51 Detailed Results

Pretraining gives more improvement when using smaller number of labelled examples in the dataset.

![inline](https://www.dropbox.com/s/3yfwkalqr31x02p/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2023.25.35.png?raw=1)![inline](https://www.dropbox.com/s/2fyiz21wwi338xv/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2023.18.26.png?raw=1)

---

# Results of Different Encoder-Decoder Models

![inline](https://www.dropbox.com/s/gm8w6t78c0x9xi0/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2023.26.55.png?raw=1)

---

# Compare with other

![inline  fit](https://www.dropbox.com/s/1hj0ower5dqyum4/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2023.25.05.png?raw=1)

---

# Conclusion

- Proposed models based on LSTMs that can learn good video representations.
- Composite Model with conditionality works best
- Able to persistently generate motion over long time.
- Improvement on supervised tasks with unsupervised learning initializing

---

# [fit] Thank you

---

## Appendix : LISP, CON, CAR, CDR

![inline](https://www.dropbox.com/s/bhqvnvyvh586u7j/img007.gif?raw=1)

---


# ConvNet feature

RGB percepts from `VGG fc6 4096`

Flow percepts from Temporal stream Convnet `fc6 4096`

![inline](https://www.dropbox.com/s/90g2b108qsph6bo/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-21%2002.01.28.png?raw=1)

---

 The model needs to know about motion (which direction and how fast things are moving) from the input. This requires precise information about **location (thin strips)** and **velocity (high frequency strips)**. But when it is generating the output, the model wants to hedge its bets so that it does not suffer **a huge loss for predicting things sharply at the wrong place**. This could explain why the **output features have somewhat bigger blobs**. The relative **shortness of the strips in the output features** can be explained by the fact that in the inputs, it does not hurt to have a longer feature than what is needed to detect a location because information is coarse-coded through multiple features. But in the output, the model may not want to put down a feature that is bigger than any digit because other units will have to conspire to correct for it.
---

# Learned Featrues

Features learned by the Encoder, Decoder of Future Predictor, and of Autoencoder respectively.

![inline](https://www.dropbox.com/s/9phlr16s1yle07a/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202016-11-20%2023.27.54.png?raw=1)

---
## Syntax highlighting

It’s _**sooo easy**_ to show `codeSamples();` in your presentations. Deckset applies syntax highlighting and scales the type size so it always looks great.

---

### Hello World!

```javascript
function myFunction(){
	alert(“Hello World!”)
};
```

### **loooong** lines are scaled down

```objectivec
UIView *someView = [[UIView alloc] init];
NSString *string = @"some string that is really, really, really really long, and has many, many, many words";
```
