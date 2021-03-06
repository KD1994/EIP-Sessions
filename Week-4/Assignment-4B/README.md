## EIP-4 ~~> Week-4 ~~> Assignment-4B 

## Model-1 Approach:
Using the 8n+2 Formula to create ResNet-18 by using given Example model [here](https://keras.io/examples/cifar10_resnet/)

1) Conv Layer with 64 output filters and and **7x7** kernel size 
2) **without maxpooling** as the image is of size 32x32.

Training Accuracy   ~ 91.64%
Test Accuracy       ~ 88.48%

Can't Use GradCam as the Last Convolution output shape is 2x2x512, so need better version.

## Model-2 Approach:
Using the same structure mentioed [here](https://keras.io/examples/cifar10_resnet/) with just a change in filter size

1) Conv Layer with 64 output filters and and **3x3** kernel size
2) **without maxpooling** as the image is of size 32x32.

Training Accuracy   ~ 93.22%
Test Accuracy       ~ 90.16%

Can't Use GradCam as the Last Convolution output shape is 4x4x512, so need better version.

## Model-3 Approach:

Using the same structure mentioed [here](https://keras.io/examples/cifar10_resnet/) with below mentioend changes

1) Conv Layer with 64 output filters and and **3x3** kernel size
2) **without maxpooling** as the image is of size 32x32.
3) Last Block with strides=1, which helps in maintaining the output shape as **8x8x512**

Training Accuracy   ~ 92.54%
Test Accuracy       ~ 90.47%

With this approach achieved **accuracy 89.58%** in **28** epoch.
Below is the Model-3 Summary and logs.


## Model Summary
```
Model: "model_5"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_6 (InputLayer)            (None, 32, 32, 3)    0                                            
__________________________________________________________________________________________________
conv2d_99 (Conv2D)              (None, 32, 32, 64)   1792        input_6[0][0]                    
__________________________________________________________________________________________________
batch_normalization_84 (BatchNo (None, 32, 32, 64)   256         conv2d_99[0][0]                  
__________________________________________________________________________________________________
activation_83 (Activation)      (None, 32, 32, 64)   0           batch_normalization_84[0][0]     
__________________________________________________________________________________________________
conv2d_100 (Conv2D)             (None, 32, 32, 64)   36928       activation_83[0][0]              
__________________________________________________________________________________________________
batch_normalization_85 (BatchNo (None, 32, 32, 64)   256         conv2d_100[0][0]                 
__________________________________________________________________________________________________
activation_84 (Activation)      (None, 32, 32, 64)   0           batch_normalization_85[0][0]     
__________________________________________________________________________________________________
conv2d_101 (Conv2D)             (None, 32, 32, 64)   36928       activation_84[0][0]              
__________________________________________________________________________________________________
batch_normalization_86 (BatchNo (None, 32, 32, 64)   256         conv2d_101[0][0]                 
__________________________________________________________________________________________________
add_40 (Add)                    (None, 32, 32, 64)   0           activation_83[0][0]              
                                                                 batch_normalization_86[0][0]     
__________________________________________________________________________________________________
activation_85 (Activation)      (None, 32, 32, 64)   0           add_40[0][0]                     
__________________________________________________________________________________________________
conv2d_102 (Conv2D)             (None, 32, 32, 64)   36928       activation_85[0][0]              
__________________________________________________________________________________________________
batch_normalization_87 (BatchNo (None, 32, 32, 64)   256         conv2d_102[0][0]                 
__________________________________________________________________________________________________
activation_86 (Activation)      (None, 32, 32, 64)   0           batch_normalization_87[0][0]     
__________________________________________________________________________________________________
conv2d_103 (Conv2D)             (None, 32, 32, 64)   36928       activation_86[0][0]              
__________________________________________________________________________________________________
batch_normalization_88 (BatchNo (None, 32, 32, 64)   256         conv2d_103[0][0]                 
__________________________________________________________________________________________________
add_41 (Add)                    (None, 32, 32, 64)   0           activation_85[0][0]              
                                                                 batch_normalization_88[0][0]     
__________________________________________________________________________________________________
activation_87 (Activation)      (None, 32, 32, 64)   0           add_41[0][0]                     
__________________________________________________________________________________________________
conv2d_104 (Conv2D)             (None, 16, 16, 128)  73856       activation_87[0][0]              
__________________________________________________________________________________________________
batch_normalization_89 (BatchNo (None, 16, 16, 128)  512         conv2d_104[0][0]                 
__________________________________________________________________________________________________
activation_88 (Activation)      (None, 16, 16, 128)  0           batch_normalization_89[0][0]     
__________________________________________________________________________________________________
conv2d_105 (Conv2D)             (None, 16, 16, 128)  147584      activation_88[0][0]              
__________________________________________________________________________________________________
conv2d_106 (Conv2D)             (None, 16, 16, 128)  8320        activation_87[0][0]              
__________________________________________________________________________________________________
batch_normalization_90 (BatchNo (None, 16, 16, 128)  512         conv2d_105[0][0]                 
__________________________________________________________________________________________________
add_42 (Add)                    (None, 16, 16, 128)  0           conv2d_106[0][0]                 
                                                                 batch_normalization_90[0][0]     
__________________________________________________________________________________________________
activation_89 (Activation)      (None, 16, 16, 128)  0           add_42[0][0]                     
__________________________________________________________________________________________________
conv2d_107 (Conv2D)             (None, 16, 16, 128)  147584      activation_89[0][0]              
__________________________________________________________________________________________________
batch_normalization_91 (BatchNo (None, 16, 16, 128)  512         conv2d_107[0][0]                 
__________________________________________________________________________________________________
activation_90 (Activation)      (None, 16, 16, 128)  0           batch_normalization_91[0][0]     
__________________________________________________________________________________________________
conv2d_108 (Conv2D)             (None, 16, 16, 128)  147584      activation_90[0][0]              
__________________________________________________________________________________________________
batch_normalization_92 (BatchNo (None, 16, 16, 128)  512         conv2d_108[0][0]                 
__________________________________________________________________________________________________
add_43 (Add)                    (None, 16, 16, 128)  0           activation_89[0][0]              
                                                                 batch_normalization_92[0][0]     
__________________________________________________________________________________________________
activation_91 (Activation)      (None, 16, 16, 128)  0           add_43[0][0]                     
__________________________________________________________________________________________________
conv2d_109 (Conv2D)             (None, 8, 8, 256)    295168      activation_91[0][0]              
__________________________________________________________________________________________________
batch_normalization_93 (BatchNo (None, 8, 8, 256)    1024        conv2d_109[0][0]                 
__________________________________________________________________________________________________
activation_92 (Activation)      (None, 8, 8, 256)    0           batch_normalization_93[0][0]     
__________________________________________________________________________________________________
conv2d_110 (Conv2D)             (None, 8, 8, 256)    590080      activation_92[0][0]              
__________________________________________________________________________________________________
conv2d_111 (Conv2D)             (None, 8, 8, 256)    33024       activation_91[0][0]              
__________________________________________________________________________________________________
batch_normalization_94 (BatchNo (None, 8, 8, 256)    1024        conv2d_110[0][0]                 
__________________________________________________________________________________________________
add_44 (Add)                    (None, 8, 8, 256)    0           conv2d_111[0][0]                 
                                                                 batch_normalization_94[0][0]     
__________________________________________________________________________________________________
activation_93 (Activation)      (None, 8, 8, 256)    0           add_44[0][0]                     
__________________________________________________________________________________________________
conv2d_112 (Conv2D)             (None, 8, 8, 256)    590080      activation_93[0][0]              
__________________________________________________________________________________________________
batch_normalization_95 (BatchNo (None, 8, 8, 256)    1024        conv2d_112[0][0]                 
__________________________________________________________________________________________________
activation_94 (Activation)      (None, 8, 8, 256)    0           batch_normalization_95[0][0]     
__________________________________________________________________________________________________
conv2d_113 (Conv2D)             (None, 8, 8, 256)    590080      activation_94[0][0]              
__________________________________________________________________________________________________
batch_normalization_96 (BatchNo (None, 8, 8, 256)    1024        conv2d_113[0][0]                 
__________________________________________________________________________________________________
add_45 (Add)                    (None, 8, 8, 256)    0           activation_93[0][0]              
                                                                 batch_normalization_96[0][0]     
__________________________________________________________________________________________________
activation_95 (Activation)      (None, 8, 8, 256)    0           add_45[0][0]                     
__________________________________________________________________________________________________
conv2d_116 (Conv2D)             (None, 8, 8, 512)    131584      activation_95[0][0]              
__________________________________________________________________________________________________
conv2d_117 (Conv2D)             (None, 8, 8, 512)    2359808     conv2d_116[0][0]                 
__________________________________________________________________________________________________
batch_normalization_99 (BatchNo (None, 8, 8, 512)    2048        conv2d_117[0][0]                 
__________________________________________________________________________________________________
activation_97 (Activation)      (None, 8, 8, 512)    0           batch_normalization_99[0][0]     
__________________________________________________________________________________________________
conv2d_118 (Conv2D)             (None, 8, 8, 512)    2359808     activation_97[0][0]              
__________________________________________________________________________________________________
conv2d_119 (Conv2D)             (None, 8, 8, 512)    262656      conv2d_116[0][0]                 
__________________________________________________________________________________________________
batch_normalization_100 (BatchN (None, 8, 8, 512)    2048        conv2d_118[0][0]                 
__________________________________________________________________________________________________
add_46 (Add)                    (None, 8, 8, 512)    0           conv2d_119[0][0]                 
                                                                 batch_normalization_100[0][0]    
__________________________________________________________________________________________________
activation_98 (Activation)      (None, 8, 8, 512)    0           add_46[0][0]                     
__________________________________________________________________________________________________
conv2d_120 (Conv2D)             (None, 8, 8, 512)    2359808     activation_98[0][0]              
__________________________________________________________________________________________________
batch_normalization_101 (BatchN (None, 8, 8, 512)    2048        conv2d_120[0][0]                 
__________________________________________________________________________________________________
activation_99 (Activation)      (None, 8, 8, 512)    0           batch_normalization_101[0][0]    
__________________________________________________________________________________________________
conv2d_121 (Conv2D)             (None, 8, 8, 512)    2359808     activation_99[0][0]              
__________________________________________________________________________________________________
batch_normalization_102 (BatchN (None, 8, 8, 512)    2048        conv2d_121[0][0]                 
__________________________________________________________________________________________________
add_47 (Add)                    (None, 8, 8, 512)    0           activation_98[0][0]              
                                                                 batch_normalization_102[0][0]    
__________________________________________________________________________________________________
activation_100 (Activation)     (None, 8, 8, 512)    0           add_47[0][0]                     
__________________________________________________________________________________________________
average_pooling2d_5 (AveragePoo (None, 1, 1, 512)    0           activation_100[0][0]             
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 512)          0           average_pooling2d_5[0][0]        
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 10)           5130        flatten_5[0][0]                  
==================================================================================================
Total params: 12,627,082
Trainable params: 12,619,274
Non-trainable params: 7,808
__________________________________________________________________________________________________
```

## Model Logs

```
Epoch 1/50
390/390 [==============================] - 98s 252ms/step - loss: 2.9545 - acc: 0.3032 - val_loss: 2.2674 - val_acc: 0.3431

Epoch 00001: val_acc improved from -inf to 0.34310, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 2/50
390/390 [==============================] - 88s 226ms/step - loss: 1.8727 - acc: 0.4744 - val_loss: 2.0945 - val_acc: 0.4256

Epoch 00002: val_acc improved from 0.34310 to 0.42560, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 3/50
390/390 [==============================] - 88s 226ms/step - loss: 1.6021 - acc: 0.5562 - val_loss: 2.7243 - val_acc: 0.3359

Epoch 00003: val_acc did not improve from 0.42560
Epoch 4/50
390/390 [==============================] - 88s 226ms/step - loss: 1.4424 - acc: 0.6061 - val_loss: 1.7632 - val_acc: 0.5195

Epoch 00004: val_acc improved from 0.42560 to 0.51950, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 5/50
390/390 [==============================] - 88s 226ms/step - loss: 1.3397 - acc: 0.6431 - val_loss: 1.9122 - val_acc: 0.4999

Epoch 00005: val_acc did not improve from 0.51950
Epoch 6/50
390/390 [==============================] - 88s 226ms/step - loss: 1.2471 - acc: 0.6787 - val_loss: 2.0459 - val_acc: 0.4646

Epoch 00006: val_acc did not improve from 0.51950
Epoch 7/50
390/390 [==============================] - 88s 225ms/step - loss: 1.1830 - acc: 0.7011 - val_loss: 1.7065 - val_acc: 0.5624

Epoch 00007: val_acc improved from 0.51950 to 0.56240, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 8/50
390/390 [==============================] - 88s 226ms/step - loss: 1.1260 - acc: 0.7153 - val_loss: 1.7536 - val_acc: 0.5736

Epoch 00008: val_acc improved from 0.56240 to 0.57360, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 9/50
390/390 [==============================] - 88s 225ms/step - loss: 1.0850 - acc: 0.7298 - val_loss: 1.4211 - val_acc: 0.6155

Epoch 00009: val_acc improved from 0.57360 to 0.61550, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 10/50
390/390 [==============================] - 88s 225ms/step - loss: 1.0432 - acc: 0.7432 - val_loss: 1.5850 - val_acc: 0.6172

Epoch 00010: val_acc improved from 0.61550 to 0.61720, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 11/50
390/390 [==============================] - 88s 225ms/step - loss: 1.0057 - acc: 0.7532 - val_loss: 1.1001 - val_acc: 0.7203

Epoch 00011: val_acc improved from 0.61720 to 0.72030, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 12/50
390/390 [==============================] - 88s 225ms/step - loss: 0.9745 - acc: 0.7601 - val_loss: 1.6438 - val_acc: 0.5877

Epoch 00012: val_acc did not improve from 0.72030
Epoch 13/50
390/390 [==============================] - 88s 225ms/step - loss: 0.9597 - acc: 0.7677 - val_loss: 1.5831 - val_acc: 0.6118

Epoch 00013: val_acc did not improve from 0.72030
Epoch 14/50
390/390 [==============================] - 88s 224ms/step - loss: 0.9381 - acc: 0.7733 - val_loss: 1.2969 - val_acc: 0.6754

Epoch 00014: val_acc did not improve from 0.72030

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.0009486833062967954.
Epoch 15/50
390/390 [==============================] - 88s 225ms/step - loss: 0.7628 - acc: 0.8249 - val_loss: 0.7137 - val_acc: 0.8362

Epoch 00015: val_acc improved from 0.72030 to 0.83620, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 16/50
390/390 [==============================] - 87s 224ms/step - loss: 0.7108 - acc: 0.8339 - val_loss: 0.7510 - val_acc: 0.8189

Epoch 00016: val_acc did not improve from 0.83620
Epoch 17/50
390/390 [==============================] - 87s 224ms/step - loss: 0.6843 - acc: 0.8393 - val_loss: 1.0666 - val_acc: 0.7410

Epoch 00017: val_acc did not improve from 0.83620
Epoch 18/50
390/390 [==============================] - 87s 224ms/step - loss: 0.6605 - acc: 0.8450 - val_loss: 0.7524 - val_acc: 0.8210

Epoch 00018: val_acc did not improve from 0.83620

Epoch 00018: ReduceLROnPlateau reducing learning rate to 0.0003000000007813074.
Epoch 19/50
390/390 [==============================] - 88s 225ms/step - loss: 0.5945 - acc: 0.8642 - val_loss: 0.5752 - val_acc: 0.8727

Epoch 00019: val_acc improved from 0.83620 to 0.87270, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 20/50
390/390 [==============================] - 88s 226ms/step - loss: 0.5596 - acc: 0.8746 - val_loss: 0.5889 - val_acc: 0.8708

Epoch 00020: val_acc did not improve from 0.87270
Epoch 21/50
390/390 [==============================] - 88s 225ms/step - loss: 0.5451 - acc: 0.8769 - val_loss: 0.5761 - val_acc: 0.8686

Epoch 00021: val_acc did not improve from 0.87270
Epoch 22/50
390/390 [==============================] - 88s 226ms/step - loss: 0.5384 - acc: 0.8782 - val_loss: 0.5429 - val_acc: 0.8794

Epoch 00022: val_acc improved from 0.87270 to 0.87940, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 23/50
390/390 [==============================] - 88s 225ms/step - loss: 0.5215 - acc: 0.8819 - val_loss: 0.5645 - val_acc: 0.8703

Epoch 00023: val_acc did not improve from 0.87940
Epoch 24/50
390/390 [==============================] - 88s 225ms/step - loss: 0.5164 - acc: 0.8839 - val_loss: 0.5418 - val_acc: 0.8794

Epoch 00024: val_acc did not improve from 0.87940
Epoch 25/50
390/390 [==============================] - 87s 224ms/step - loss: 0.5053 - acc: 0.8853 - val_loss: 0.5702 - val_acc: 0.8728

Epoch 00025: val_acc did not improve from 0.87940
Epoch 26/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4959 - acc: 0.8866 - val_loss: 0.5425 - val_acc: 0.8785

Epoch 00026: val_acc did not improve from 0.87940
Epoch 27/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4836 - acc: 0.8919 - val_loss: 0.5822 - val_acc: 0.8693

Epoch 00027: val_acc did not improve from 0.87940

Epoch 00027: ReduceLROnPlateau reducing learning rate to 0.0001.
Epoch 28/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4627 - acc: 0.8978 - val_loss: 0.4891 - val_acc: 0.8958

Epoch 00028: val_acc improved from 0.87940 to 0.89580, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 29/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4464 - acc: 0.9024 - val_loss: 0.5071 - val_acc: 0.8910

Epoch 00029: val_acc did not improve from 0.89580
Epoch 30/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4445 - acc: 0.9035 - val_loss: 0.5012 - val_acc: 0.8912

Epoch 00030: val_acc did not improve from 0.89580
Epoch 31/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4363 - acc: 0.9060 - val_loss: 0.4832 - val_acc: 0.8980

Epoch 00031: val_acc improved from 0.89580 to 0.89800, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 32/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4311 - acc: 0.9088 - val_loss: 0.4841 - val_acc: 0.8961

Epoch 00032: val_acc did not improve from 0.89800
Epoch 33/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4308 - acc: 0.9079 - val_loss: 0.4902 - val_acc: 0.8936

Epoch 00033: val_acc did not improve from 0.89800
Epoch 34/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4281 - acc: 0.9081 - val_loss: 0.4758 - val_acc: 0.8969

Epoch 00034: val_acc did not improve from 0.89800
Epoch 35/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4219 - acc: 0.9099 - val_loss: 0.4868 - val_acc: 0.8999

Epoch 00035: val_acc improved from 0.89800 to 0.89990, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 36/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4186 - acc: 0.9100 - val_loss: 0.4807 - val_acc: 0.8997

Epoch 00036: val_acc did not improve from 0.89990
Epoch 37/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4142 - acc: 0.9119 - val_loss: 0.4750 - val_acc: 0.8987

Epoch 00037: val_acc did not improve from 0.89990
Epoch 38/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4121 - acc: 0.9120 - val_loss: 0.4872 - val_acc: 0.8964

Epoch 00038: val_acc did not improve from 0.89990
Epoch 39/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4020 - acc: 0.9151 - val_loss: 0.4753 - val_acc: 0.8988

Epoch 00039: val_acc did not improve from 0.89990
Epoch 40/50
390/390 [==============================] - 87s 224ms/step - loss: 0.4037 - acc: 0.9142 - val_loss: 0.4822 - val_acc: 0.8956

Epoch 00040: val_acc did not improve from 0.89990
Epoch 41/50
390/390 [==============================] - 87s 223ms/step - loss: 0.3979 - acc: 0.9171 - val_loss: 0.4724 - val_acc: 0.9010

Epoch 00041: val_acc improved from 0.89990 to 0.90100, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 42/50
390/390 [==============================] - 87s 223ms/step - loss: 0.3982 - acc: 0.9152 - val_loss: 0.4683 - val_acc: 0.8980

Epoch 00042: val_acc did not improve from 0.90100
Epoch 43/50
390/390 [==============================] - 87s 223ms/step - loss: 0.3915 - acc: 0.9177 - val_loss: 0.4571 - val_acc: 0.9046

Epoch 00043: val_acc improved from 0.90100 to 0.90460, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 44/50
390/390 [==============================] - 87s 223ms/step - loss: 0.3879 - acc: 0.9199 - val_loss: 0.4613 - val_acc: 0.9024

Epoch 00044: val_acc did not improve from 0.90460
Epoch 45/50
390/390 [==============================] - 87s 224ms/step - loss: 0.3856 - acc: 0.9197 - val_loss: 0.4597 - val_acc: 0.9013

Epoch 00045: val_acc did not improve from 0.90460
Epoch 46/50
390/390 [==============================] - 87s 224ms/step - loss: 0.3821 - acc: 0.9200 - val_loss: 0.4558 - val_acc: 0.9047

Epoch 00046: val_acc improved from 0.90460 to 0.90470, saving model to /content/CIFAR_10_Resnet20_weights.h5
Epoch 47/50
390/390 [==============================] - 88s 225ms/step - loss: 0.3800 - acc: 0.9211 - val_loss: 0.4866 - val_acc: 0.8967

Epoch 00047: val_acc did not improve from 0.90470
Epoch 48/50
390/390 [==============================] - 87s 224ms/step - loss: 0.3788 - acc: 0.9219 - val_loss: 0.4561 - val_acc: 0.9046

Epoch 00048: val_acc did not improve from 0.90470
Epoch 49/50
390/390 [==============================] - 87s 224ms/step - loss: 0.3716 - acc: 0.9231 - val_loss: 0.4787 - val_acc: 0.8990

Epoch 00049: val_acc did not improve from 0.90470
Epoch 50/50
390/390 [==============================] - 87s 224ms/step - loss: 0.3686 - acc: 0.9254 - val_loss: 0.4572 - val_acc: 0.9045

Epoch 00050: val_acc did not improve from 0.90470
Model took 4423.17 seconds to train
```
