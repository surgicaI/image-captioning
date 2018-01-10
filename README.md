# Image Captioning

### Author:
- [Simranjyot Singh Gill](https://github.com/surgicaI)

## Abstract
In this paper we present various models that generates natural language descriptions of images and their regions.
Our approach uses dense neural networks for this task. We use combination of convolution neural networks over the 
image regions to extract features from the image and then used Recurrent neural networks to generate text from 
these features. We also incorporate the attention mechanism while generating Natural language sequence from these 
features. This project leverages COCO dataset of images and their corresponding captions to learn about the inter-modal 
correspondences between and visual data and its natural language description. This paper then present results on unseen 
test set and also compare the results for different configurations of convolution neural networks(VGG and Resnet), 
using LSTM or GRU for recurrent neural network and results with and without the attention mechanism.  
```
Keywords:RNN, GRU, LSTM, RESNET, VGG, Attention, Embeddings
```

## 1. Introduction

Automatically captioning the image is a fundamental problem in artificial
intelligence. Image captioning task can have numerous applications. Most basic
application can be to generate captions such that the images which accurately
matches the search query can be easily searched by the search engines over the
Internet. Other such application can be to generate captions so that similar
images can be paired together in a huge corpus eg: all images of flowers.
Similarly many other applications can be thought for the task of image
captioning. Humans are very efficient at this task and can provide highly
detailed description of an image with just one quick glance. However this task
has been proven to be very hard for artificial systems before the advent of deep
learning models. The majority of previous work in visual recognition has focused
on labeling images with a fixed set of visual categories and this closed
vocabulary assumption proves very restrictive when compared to the enormous
amount of rich descriptions that a human can compose. Deep learning models
perform very well at this task and in this project we follow deep learning
approach to tackle the problem of automatic image captioning. Convolution neural
networks[1] belong to a class of neural networks which
are used to extract features from an image. Very dense combination of
convolution neural networks is used in mostly every state of the art model for
solving the problems in the field of computer vision. Recurrent neural networks
belong to another class of neural networks which aside from having a feed
forward network also maintain previous states[2]. Thus
recurrent neural networks are widely used in problems related to natural
languages such as machine translation because the output can also depend on any
of the previous states. As the problem of image captioning connects the fields
of computer vision and natural language processing we use combination of
convolution neural networks for extracting features from the image and then pass
these features to a recurrent neural network to generate natural language
description of the image. In this project we experiment with different
configurations of convolution neural networks such as ResNet and VGG. We also
experiment with different Recurrent neural networks namely LSTMs and GRUs. We
also incorporate the attention mechanism such that the neural network will focus
on a particular region in the image while generating description related to that
region of the image. We then present the results and compare the results for
above mentioned configurations. 

## 2. Related Work

As mentioned above there have been several attempts at solving the problem of
automatic image captioning before deep learning models were popularized. Some
examples are Midge - Mitchell et al.[3] and BabyTalk[4]. In their paper Mitchell
et al. leveraged syntactically informed word co-occurrence statistics such that
their generator filters and constrains the noisy detections output from a vision
system to generate syntactic trees that detail what the computer vision system
sees. In BabyTalk they first smooths the output of computer vision-based
detection and recognition algorithms with statistics mined from large pools of
visually descriptive text to determine the best content words to use to describe
an image. Then they choose words to construct natural language sentences based
on the predicted content and general statistics from natural language. Many of
the recent papers Kiros et al.[5], Vinyals et al.[6] on image captioning uses
Generative approach instead of retrieval approach. In generative approach the
caption is generated from the entire vocabulary but in retrieval approach best
words are picked among a certain restricted set (they restricted their
vocabulary size to approx 9567 words).Even unseen captions can be generated with
End- to-end models. Few papers followed End-to-end approach instead of pipeline
approach, where in end-to-end[6] the solution is encapsulated in one neural net.
In this model they are using a pipeline approach where they first used a vgg net
to extract the image descriptions and then used a language model to generate the
sequences. The advantage of an End-to-end model is that the solution is embedded
in one neural net and can be fully trainable using Stochastic Gradient Descent.


## 3. Dataset(s)

We use Microsoft COCO: common objects in context dataset[7] for training
the neural network model. The dataset contains images of complex everyday
scenes containing common objects in their natural context. The dataset also
contains five captions for each image. We divide the dataset between training
and test set such that there are around 82k images in training set and around
40k images in test set. Prime motivation for using this dataset is that its been
experimentally verified that models trained on COCO dataset perform better
when evaluated on everyday scenes as compared to those trained with prior
datasets. The convolution neural network in our deep neural model expects all
the images to be of same size thus images in the dataset have been resized and
cropped to 224x224 pixels and are normalized before passing to the convolution
network.

## 4. Architecture

The architecture of the basic model[8] without attention is shown in fig:1. As a
first step we gathered the set of words that are in vocabulary. Any word whose
count is greater than a threshold of 4 in caption sequences is added to the
vocabulary set. Each word is also given an index which will be used to represent
a word before passing to the embedding layer. To get word embedding for any
word we can pass the index of the word to the embedding layer in the model.
The weights of embedding layer are also trained with the model. We get the
word embeddings from the embedding layer which are passed to the LSTM.

![](https://github.com/surgicaI/image-captioning/blob/master/images/architecture.png)
```
Fig. 1. Architecture of deep neural network
```

As can be seen from the figure, the image of dimension 3x224x224 is passed
as an input to the convolution neural network. The images are initially resized
to 256x256 and then cropped to 224x224. We use random horizontal flip and
random crop for data augmentation. The convolution layers are taken from pre-
trained ResNet model. Once the image is passed through the convolution layers
we get a feature vector of size 2048. This feature vector is then passed through
a linear layer and the output is a vector of size 256. This vector is then provided
as an input to the LSTM whose hidden state and cell state are initialized to 0.
For all the next time steps LSTM will maintain its cell state and hidden state
and at each time step next token in the image caption sequence is provided as
input to the LSTM. At each time step a token out of vocabulary is predicted.

### 4.1 Attention Mechanism

As it can be seen in the above model the feature map from convolution neural
network, after getting passed through a linear layer, is used only as an input to
LSTM at zeroth time step, and is not used as an input in any of the remaining
time-steps. Hence we can see that the context vector which is used by the decoder
to unfold entire caption sequence is of size 256. Sometimes this small size context
vector may not be able to summarize the information of the entire image and
thus we need attention mechanism[9] which lets the decoder learn to focus over a
specific region of the input image while generating description of that region[10].

![](https://github.com/surgicaI/image-captioning/blob/master/images/architecture-attention.png)
```
Fig. 2. Architecture of deep neural network with attention mechanism
```
In this architecture, as can be seen in fig:3, starting from the first time-step
we calculate attention weights[11]. To calculate attention weights we concatenate
word embeddings of the current token with the current hidden state of the
LSTM. This concatenated vector is then passed through a linear layer and to
get the attention weights softmax function is applied to the output. We want our
neural network to learn to focus of certain region in the image while generating
the description of that image region. Thus these attention weights are then
multiplied(element wise) with cnn feature map weights to get attended weights.
After applying attention to the feature map of the image we use these attended
weights in the prediction of next token in image’s natural language description.
The attended weights are then concatenated with the word embeddings and
passed through a linear layer such that the output vector is of same size as
word embedding. This vector is then passed as input to the LSTM. Similar to
the previous architecture here also all the cell states and hidden states will be
maintained by the LSTM for each time step.  
During the test phase, the encoder part is almost same as the training phase.
The only difference is that batchnorm layer uses moving average and variance
instead of mini-batch statistics. This is implemented by changing the state of
encoder to evaluate from training. There is a significant difference for the decoder
part between the training phase and the test phase. During the test phase, as the
LSTM decoder can’t see the image description we pass the previously generated
word as the next input. The maximum allowed length of generated sequence is
20 words. We ignore all the remaining tokens once we encounter end of speech
tag<end>in the generated sequence.  
  
![](https://github.com/surgicaI/image-captioning/blob/master/images/attention.png)
```
Fig. 3.Attention
```
## 5. Experiments and Results
We experiment with different convolution neural networks namely ResNet and
VGG. We also experiment with different RNNs namely LSTM and GRU. We
have referenced chen et al. [12] for calculating BLEU score and CIDEr scores.
Fig: 5 shows calculated BLEU score and CIDEr score on validation set. As can
be seen from the results, BLUE score with attention model is slightly higher as
compared to other models. The configuration of ResNet for convolution neural
network and LSTM for RNN gives best results among all the other configura-
tions, thus I have extended this configuration to implement attention model.
Fig: 4 shows the decrease in training loss with training time for this model with
and without attention. Both the plots are quite overlapping and training loss
for attention model after convergence is almost same as compared to the model
without attention.

![](https://github.com/surgicaI/image-captioning/blob/master/images/loss.png)
```
Fig. 4.Training loss vs training time
```  
  
![](https://github.com/surgicaI/image-captioning/blob/master/images/results.png)
```
Fig. 5.Results
```
## 6 Conclusion and Results Interpretation

We get best results while using ResNet as convolution network as compared to
VGG. This might be due to the reason that ResNet has extra skip connections
that pass important features from earlier layers to later layers. For the recur-
rent neural network, results are almost same for LSTM and GRU with LSTM
giving slightly better results. The advantage of GRU over LSTM is that it has
two gates where LSTM has three gates, thus a GRU is faster to train and the
results are comparable for GRU and LSTM. We also experiment with various
optimizers such as SGD, Adam and RMSprop. The best results were obtained
using RMSprop as optimizer. We present few examples of generated captions on
the unseen test dataset with and without the attention mechanism. It can be
clearly seen from these examples that captions generated by the model with at-
tention mechanism describes the image scene in a much better way as compared
to the captions generated without the attention mechanism. We also present few
examples where both the models predicted caption incorrectly. Although the
generated captions for these images are incorrect but they describe very closely
related scenes. Our model might have seen these features in those closely related
images while training.

![](https://github.com/surgicaI/image-captioning/blob/master/images/fig-1.png)
```
Caption with attention mechanism: a wooden bunk bed and wooden ladder.
Caption without attention mechanism: a wooden shelf with a white and yellow surfboard.
```
***

![](https://github.com/surgicaI/image-captioning/blob/master/images/fig-2.png)
```
Caption with attention mechanism: a room with a chair , table , and a television.
Caption without attention mechanism: a room with a large wooden table and chairs.
```
***

![](https://github.com/surgicaI/image-captioning/blob/master/images/fig-3.png)
```
Caption with attention mechanism: a kitchen with a chair , refrigerator , stove and other kitchen supplies.
Caption without attention mechanism: a kitchen with a sink , refrigerator and a stove
```
***

![](https://github.com/surgicaI/image-captioning/blob/master/images/fig-4.png)
```
Caption with attention mechanism: a wine glass and a bottle on a table
Caption without attention mechanism: a wine glass is sitting on a table
```
***

![](https://github.com/surgicaI/image-captioning/blob/master/images/fig-5.png)
```
Caption with attention mechanism: a traffic light hanging over a street next to a building
Caption without attention mechanism: a traffic light sitting on the side of a road
```
***

![](https://github.com/surgicaI/image-captioning/blob/master/images/fig-6.png)
```
Caption with attention mechanism: a plane flying over a body of water
Caption without attention mechanism: a large white bird flying over a body of water
```
***

![](https://github.com/surgicaI/image-captioning/blob/master/images/fig-7.png)
```
Caption with attention mechanism: a baby brushing it ’s teeth with a tooth brush
Caption without attention mechanism: a baby is sitting on a toothbrush in a tub
```
***

![](https://github.com/surgicaI/image-captioning/blob/master/images/fig-8.png)
```
Caption with attention mechanism: a stop sign on a pole on a street
Caption without attention mechanism: a red stop sign sitting on the side of a road
```
***

![](https://github.com/surgicaI/image-captioning/blob/master/images/fig-9.png)
```
Caption with attention mechanism: a woman in a red jacket skiing down a hill
Caption without attention mechanism: a man riding skis down a snow covered slope
```


## References

1. Yann LeCun, Yoshua Bengio, et al. Convolutional networks for images, speech,
    and time series.The handbook of brain theory and neural networks, 3361(10):1995,
    1995.
2. LR Medsker and LC Jain. Recurrent neural networks.Design and Applications,
    5, 2001.
3. Margaret Mitchell, Xufeng Han, Jesse Dodge, Alyssa Mensch, Amit Goyal, Alex
    Berg, Kota Yamaguchi, Tamara Berg, Karl Stratos, and Hal Daum ́e III. Midge:
    Generating image descriptions from computer vision detections. InProceedings of
    the 13th Conference of the European Chapter of the Association for Computational
    Linguistics, pages 747–756. Association for Computational Linguistics, 2012.
4. Girish Kulkarni, Visruth Premraj, Vicente Ordonez, Sagnik Dhar, Siming Li, Yejin
    Choi, Alexander C Berg, and Tamara L Berg. Babytalk: Understanding and gen-
    erating simple image descriptions. IEEE Transactions on Pattern Analysis and
    Machine Intelligence, 35(12):2891–2903, 2013.
5. Ryan Kiros, Ruslan Salakhutdinov, and Richard S Zemel. Unifying visual-
    semantic embeddings with multimodal neural language models. arXiv preprint
    arXiv:1411.2539, 2014.
6. Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and
    tell: A neural image caption generator. InProceedings of the IEEE conference on
    computer vision and pattern recognition, pages 3156–3164, 2015.
7. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva
    Ramanan, Piotr Doll ́ar, and C Lawrence Zitnick. Microsoft coco: Common objects
    in context. InEuropean conference on computer vision, pages 740–755. Springer,
    2014.
8. Yunjey. yunjey/pytorch-tutorial, May 2017.


9. Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating
    image descriptions. InProceedings of the IEEE Conference on Computer Vision
    and Pattern Recognition, pages 3128–3137, 2015.
10. Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan
Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural
image caption generation with visual attention. InInternational Conference on
Machine Learning, pages 2048–2057, 2015.
11. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine trans-
lation by jointly learning to align and translate.arXiv preprint arXiv:1409.0473, 2014.
12. Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta,
Piotr Doll ́ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection
and evaluation server.arXiv preprint arXiv:1504.00325, 2015.
