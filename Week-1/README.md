# EIP-MNIST
MNIST using PyTorch 

## Train Accuracy
Model-1 (with Fully connected Layer) ~ 98 % 

Model-2 (witn No Fully connected Layer) ~ 68 % 

## Test Accuracy
Model-1 (with Fully connected Layer) ~ 99 %

Model-2 (witn No Fully connected Layer) ~ 99 % 

## Convolution
The Process of generating new function based on some operations done on input functions.

## Filters/Kernels
Kernel is a matrix or feature extractor, which traverses through the input from left to right, top to bottom. 
e.g. just like sliding window
Each element of Kernel will be multiplied with input and then summed to generate the output which we call feature.

## Epoch
epoch is the Total number of samples passed through the neural network at once.

## 1x1 Convolution
It is simple mapping of input to output, but it is very crucial when it comes to decrease the depth or dimentionality.

## 3x3 Convolution
The reason it is being called 3x3, because of the kernel size as 3x3 also it is the mostly used filter available now, you can achieve even 5x5 by using 3x3 two times which helps in reduceing the weights.

## Feature Map
Feature Map is generated when we apply the filter to an input image. For each layer when filter is applied the generated output is called Feature Map.

## Activation Function
This determines the output of the neural network, like yes or no, 0 or 1 etc. depending on the scenario.

## Receptive Field
Total number of pixels' information consumed by the next layer pixel is called Local receptive field whereas the number of pixels's information consumed at the last layer is called Global receptive field.
e.g. 

     Layer 1: input 5x5 --> Kernel = 3x3

     Layer 2: input 5x5 --> Kernel = 3x3 --> Local receptive field -> 9 pixels from the layer 1, while convolving
 
     Layer 3: input 5x5 --> Kernel = 3x3 --> Global receptive field -> 25 pixels which has all the information from 1 to 3                                                layers

