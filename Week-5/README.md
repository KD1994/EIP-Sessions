## EIP-4 Week-5

## Training Accuracy

```
{
 'age_output_acc': 0.9403,
 'age_output_loss': 0.1693,
 'bag_output_acc': 0.9114,
 'bag_output_loss': 0.2311,
 'emotion_output_acc': 0.9289,
 'emotion_output_loss': 0.1977,
 'footwear_output_acc': 0.9082,
 'footwear_output_loss': 0.2401,
 'gender_output_acc': 0.9227,
 'gender_output_loss': 0.1718,
 'image_quality_output_acc': 0.9556,
 'image_quality_output_loss': 0.1237,
 'pose_output_acc': 0.9367,
 'pose_output_loss': 0.1659,
 'weight_output_acc': 0.9049,
 'weight_output_loss': 0.2537
}
```

## Test Accuracy

```
{
 'age_output_acc': 0.9875868055555556,
 'age_output_loss': 0.052596430635700624,
 'bag_output_acc': 0.9681423611111111,
 'bag_output_loss': 0.11016218314568202,
 'emotion_output_acc': 0.9840277777777777,
 'emotion_output_loss': 0.07065147958281967,
 'footwear_output_acc': 0.9728298611111111,
 'footwear_output_loss': 0.10611733283019728,
 'gender_output_acc': 0.9778645833333334,
 'gender_output_loss': 0.08309600591245625,
 'image_quality_output_acc': 0.9902777777777778,
 'image_quality_output_loss': 0.039623476176833115,
 'loss': 0.2502261357174979,
 'pose_output_acc': 0.97890625,
 'pose_output_loss': 0.07190398246877723,
 'weight_output_acc': 0.9662326388888889,
 'weight_output_loss': 0.11660327727182043
}
```

## Model

```
model = Sequential()
 
model.add(Convolution2D(32, 3, activation='relu',  use_bias=False,input_shape=(224,224,3))) #222
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(32, 7, strides=2, padding='same', use_bias=False, activation='relu')) #111
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(32, 3, use_bias=False,activation='relu')) #109
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(32, 3, use_bias=False,activation='relu')) # 107
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, 3, use_bias=False, activation='relu')) # 105
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, 3, use_bias=False, activation='relu')) # 103
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, 3, use_bias=False, activation='relu')) # 101
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, 3, use_bias=False,activation='relu')) # 99
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, 1, use_bias=False, activation='relu')) # 99
model.add(MaxPooling2D(pool_size=(2, 2))) # 49

model.add(Convolution2D(32, 3, use_bias=False,activation='relu')) # 47
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) # 23

model.add(Convolution2D(32, 3, use_bias=False,activation='relu')) # 21
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(32, 1, use_bias=False, activation='relu')) # 21
model.add(MaxPooling2D(pool_size=(2, 2))) #10

model.add(Convolution2D(32, 3, use_bias=False,activation='relu')) # 8
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, activation='relu')) # 8

neck = model.output
neck = Flatten(name="flatten")(neck)
neck = Dense(512, activation="relu")(neck)


def build_tower(in_layer):
    neck = Dropout(0.1)(in_layer)
    neck = Dense(128, activation="relu")(neck)
    return neck


def build_head(name, in_layer):
    return Dense(
        num_units[name], activation="softmax", name=f"{name}_output"
    )(in_layer)

# heads
gender = build_head("gender", build_tower(neck))
image_quality = build_head("image_quality", build_tower(neck))
age = build_head("age", build_tower(neck))
weight = build_head("weight", build_tower(neck))
bag = build_head("bag", build_tower(neck))
footwear = build_head("footwear", build_tower(neck))
emotion = build_head("emotion", build_tower(neck))
pose = build_head("pose", build_tower(neck))


model = Model(
    inputs=model.input, 
    outputs=[gender, image_quality, age, weight, bag, footwear, pose, emotion]
)
```

## Summary

```
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
conv2d_1_input (InputLayer)     (None, 224, 224, 3)  0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 222, 222, 32) 864         conv2d_1_input[0][0]             
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 222, 222, 32) 128         conv2d_1[0][0]                   
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 222, 222, 32) 0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 111, 111, 32) 50176       dropout_1[0][0]                  
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 111, 111, 32) 128         conv2d_2[0][0]                   
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 111, 111, 32) 0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 109, 109, 32) 9216        dropout_2[0][0]                  
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 109, 109, 32) 128         conv2d_3[0][0]                   
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 109, 109, 32) 0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 107, 107, 32) 9216        dropout_3[0][0]                  
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 107, 107, 32) 128         conv2d_4[0][0]                   
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 107, 107, 32) 0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 105, 105, 16) 4608        dropout_4[0][0]                  
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 105, 105, 16) 64          conv2d_5[0][0]                   
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 105, 105, 16) 0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 103, 103, 16) 2304        dropout_5[0][0]                  
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 103, 103, 16) 64          conv2d_6[0][0]                   
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 103, 103, 16) 0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 101, 101, 16) 2304        dropout_6[0][0]                  
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 101, 101, 16) 64          conv2d_7[0][0]                   
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 101, 101, 16) 0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 99, 99, 16)   2304        dropout_7[0][0]                  
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 99, 99, 16)   64          conv2d_8[0][0]                   
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 99, 99, 16)   0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 99, 99, 16)   256         dropout_8[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 49, 49, 16)   0           conv2d_9[0][0]                   
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 47, 47, 32)   4608        max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 47, 47, 32)   128         conv2d_10[0][0]                  
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 47, 47, 32)   0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 23, 23, 32)   0           dropout_9[0][0]                  
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 21, 21, 32)   9216        max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 21, 21, 32)   128         conv2d_11[0][0]                  
__________________________________________________________________________________________________
dropout_10 (Dropout)            (None, 21, 21, 32)   0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 21, 21, 32)   1024        dropout_10[0][0]                 
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 10, 10, 32)   0           conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 8, 8, 32)     9216        max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 8, 8, 32)     128         conv2d_13[0][0]                  
__________________________________________________________________________________________________
dropout_11 (Dropout)            (None, 8, 8, 32)     0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 8, 8, 10)     330         dropout_11[0][0]                 
__________________________________________________________________________________________________
flatten (Flatten)               (None, 640)          0           conv2d_14[0][0]                  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 512)          328192      flatten[0][0]                    
__________________________________________________________________________________________________
dropout_12 (Dropout)            (None, 512)          0           dense_1[0][0]                    
__________________________________________________________________________________________________Epoch 1/50
390/390 [==============================] - 32s 83ms/step - loss: 1.7546 - acc: 0.3659 - val_loss: 1.5890 - val_acc: 0.4776

Epoch 00001: val_acc improved from -inf to 0.47760, saving model to /content/best_weights.hdf5
Epoch 2/50
390/390 [==============================] - 30s 76ms/step - loss: 1.1866 - acc: 0.5871 - val_loss: 1.4148 - val_acc: 0.5595

Epoch 00002: val_acc improved from 0.47760 to 0.55950, saving model to /content/best_weights.hdf5
Epoch 3/50
390/390 [==============================] - 30s 76ms/step - loss: 0.8988 - acc: 0.6830 - val_loss: 1.0163 - val_acc: 0.6613

Epoch 00003: val_acc improved from 0.55950 to 0.66130, saving model to /content/best_weights.hdf5
Epoch 4/50
390/390 [==============================] - 30s 76ms/step - loss: 0.7791 - acc: 0.7259 - val_loss: 0.8604 - val_acc: 0.7019

Epoch 00004: val_acc improved from 0.66130 to 0.70190, saving model to /content/best_weights.hdf5
Epoch 5/50
390/390 [==============================] - 30s 76ms/step - loss: 0.7075 - acc: 0.7535 - val_loss: 0.8136 - val_acc: 0.7279

Epoch 00005: val_acc improved from 0.70190 to 0.72790, saving model to /content/best_weights.hdf5
Epoch 6/50
390/390 [==============================] - 29s 76ms/step - loss: 0.6553 - acc: 0.7723 - val_loss: 0.7628 - val_acc: 0.7391

Epoch 00006: val_acc improved from 0.72790 to 0.73910, saving model to /content/best_weights.hdf5
Epoch 7/50
390/390 [==============================] - 30s 76ms/step - loss: 0.6200 - acc: 0.7841 - val_loss: 0.7517 - val_acc: 0.7493

Epoch 00007: val_acc improved from 0.73910 to 0.74930, saving model to /content/best_weights.hdf5
Epoch 8/50
390/390 [==============================] - 30s 76ms/step - loss: 0.5829 - acc: 0.7976 - val_loss: 0.7312 - val_acc: 0.7540

Epoch 00008: val_acc improved from 0.74930 to 0.75400, saving model to /content/best_weights.hdf5
Epoch 9/50
390/390 [==============================] - 30s 76ms/step - loss: 0.5592 - acc: 0.8055 - val_loss: 0.6756 - val_acc: 0.7713

Epoch 00009: val_acc improved from 0.75400 to 0.77130, saving model to /content/best_weights.hdf5
Epoch 10/50
390/390 [==============================] - 30s 76ms/step - loss: 0.5307 - acc: 0.8165 - val_loss: 0.7462 - val_acc: 0.7502

Epoch 00010: val_acc did not improve from 0.77130
Epoch 11/50
390/390 [==============================] - 29s 76ms/step - loss: 0.5022 - acc: 0.8236 - val_loss: 0.7359 - val_acc: 0.7564

Epoch 00011: val_acc did not improve from 0.77130
Epoch 12/50
390/390 [==============================] - 30s 76ms/step - loss: 0.4929 - acc: 0.8281 - val_loss: 0.6597 - val_acc: 0.7801

Epoch 00012: val_acc improved from 0.77130 to 0.78010, saving model to /content/best_weights.hdf5
Epoch 13/50
390/390 [==============================] - 30s 76ms/step - loss: 0.4730 - acc: 0.8354 - val_loss: 0.6500 - val_acc: 0.7920

Epoch 00013: val_acc improved from 0.78010 to 0.79200, saving model to /content/best_weights.hdf5
Epoch 14/50
390/390 [==============================] - 29s 76ms/step - loss: 0.4558 - acc: 0.8413 - val_loss: 0.6848 - val_acc: 0.7730

Epoch 00014: val_acc did not improve from 0.79200
Epoch 15/50
390/390 [==============================] - 30s 76ms/step - loss: 0.4417 - acc: 0.8467 - val_loss: 0.7665 - val_acc: 0.7615

Epoch 00015: val_acc did not improve from 0.79200
Epoch 16/50
390/390 [==============================] - 29s 76ms/step - loss: 0.4262 - acc: 0.8517 - val_loss: 0.5722 - val_acc: 0.8129

Epoch 00016: val_acc improved from 0.79200 to 0.81290, saving model to /content/best_weights.hdf5
Epoch 17/50
390/390 [==============================] - 30s 76ms/step - loss: 0.4174 - acc: 0.8544 - val_loss: 0.6771 - val_acc: 0.7785

Epoch 00017: val_acc did not improve from 0.81290
Epoch 18/50
390/390 [==============================] - 30s 76ms/step - loss: 0.4033 - acc: 0.8587 - val_loss: 0.7395 - val_acc: 0.7734

Epoch 00018: val_acc did not improve from 0.81290
Epoch 19/50
390/390 [==============================] - 30s 76ms/step - loss: 0.3939 - acc: 0.8609 - val_loss: 0.6407 - val_acc: 0.7946

Epoch 00019: val_acc did not improve from 0.81290
Epoch 20/50
390/390 [==============================] - 30s 76ms/step - loss: 0.3843 - acc: 0.8649 - val_loss: 0.6613 - val_acc: 0.7928

Epoch 00020: val_acc did not improve from 0.81290
Epoch 21/50
390/390 [==============================] - 30s 76ms/step - loss: 0.3757 - acc: 0.8681 - val_loss: 0.6050 - val_acc: 0.8046

Epoch 00021: val_acc did not improve from 0.81290
Epoch 22/50
390/390 [==============================] - 29s 76ms/step - loss: 0.3660 - acc: 0.8714 - val_loss: 0.6098 - val_acc: 0.7985

Epoch 00022: val_acc did not improve from 0.81290
Epoch 23/50
390/390 [==============================] - 30s 76ms/step - loss: 0.3591 - acc: 0.8745 - val_loss: 0.6223 - val_acc: 0.8003

Epoch 00023: val_acc did not improve from 0.81290
Epoch 24/50
390/390 [==============================] - 29s 76ms/step - loss: 0.3511 - acc: 0.8755 - val_loss: 0.8018 - val_acc: 0.7572

Epoch 00024: val_acc did not improve from 0.81290
Epoch 25/50
390/390 [==============================] - 30s 76ms/step - loss: 0.3451 - acc: 0.8792 - val_loss: 0.6837 - val_acc: 0.7923

Epoch 00025: val_acc did not improve from 0.81290
Epoch 26/50
390/390 [==============================] - 29s 76ms/step - loss: 0.3372 - acc: 0.8822 - val_loss: 0.6177 - val_acc: 0.8064

Epoch 00026: val_acc did not improve from 0.81290
Epoch 27/50
390/390 [==============================] - 30s 76ms/step - loss: 0.3320 - acc: 0.8829 - val_loss: 0.6182 - val_acc: 0.7999

Epoch 00027: val_acc did not improve from 0.81290
Epoch 28/50
390/390 [==============================] - 29s 76ms/step - loss: 0.3188 - acc: 0.8871 - val_loss: 0.6517 - val_acc: 0.7960

Epoch 00028: val_acc did not improve from 0.81290
Epoch 29/50
390/390 [==============================] - 30s 76ms/step - loss: 0.3225 - acc: 0.8862 - val_loss: 0.6076 - val_acc: 0.8142

Epoch 00029: val_acc improved from 0.81290 to 0.81420, saving model to /content/best_weights.hdf5
Epoch 30/50
390/390 [==============================] - 29s 76ms/step - loss: 0.3196 - acc: 0.8877 - val_loss: 0.5671 - val_acc: 0.8134

Epoch 00030: val_acc did not improve from 0.81420
Epoch 31/50
390/390 [==============================] - 29s 76ms/step - loss: 0.3065 - acc: 0.8909 - val_loss: 0.5963 - val_acc: 0.8136

Epoch 00031: val_acc did not improve from 0.81420
Epoch 32/50
390/390 [==============================] - 29s 76ms/step - loss: 0.3056 - acc: 0.8918 - val_loss: 0.6097 - val_acc: 0.8120

Epoch 00032: val_acc did not improve from 0.81420
Epoch 33/50
390/390 [==============================] - 29s 76ms/step - loss: 0.3011 - acc: 0.8937 - val_loss: 0.6196 - val_acc: 0.8025

Epoch 00033: val_acc did not improve from 0.81420
Epoch 34/50
390/390 [==============================] - 29s 76ms/step - loss: 0.2890 - acc: 0.8975 - val_loss: 0.6057 - val_acc: 0.8183

Epoch 00034: val_acc improved from 0.81420 to 0.81830, saving model to /content/best_weights.hdf5
Epoch 35/50
390/390 [==============================] - 30s 76ms/step - loss: 0.2877 - acc: 0.8984 - val_loss: 0.7143 - val_acc: 0.7941

Epoch 00035: val_acc did not improve from 0.81830
Epoch 36/50
390/390 [==============================] - 29s 75ms/step - loss: 0.2858 - acc: 0.8985 - val_loss: 0.6755 - val_acc: 0.7977

Epoch 00036: val_acc did not improve from 0.81830
Epoch 37/50
390/390 [==============================] - 29s 75ms/step - loss: 0.2829 - acc: 0.8988 - val_loss: 0.7908 - val_acc: 0.7713

Epoch 00037: val_acc did not improve from 0.81830
Epoch 38/50
390/390 [==============================] - 29s 76ms/step - loss: 0.2732 - acc: 0.9028 - val_loss: 0.6195 - val_acc: 0.8124

Epoch 00038: val_acc did not improve from 0.81830
Epoch 39/50
390/390 [==============================] - 29s 76ms/step - loss: 0.2700 - acc: 0.9041 - val_loss: 0.7104 - val_acc: 0.7905

Epoch 00039: val_acc did not improve from 0.81830
Epoch 40/50
390/390 [==============================] - 30s 76ms/step - loss: 0.2683 - acc: 0.9046 - val_loss: 0.6636 - val_acc: 0.7996

Epoch 00040: val_acc did not improve from 0.81830
Epoch 41/50
390/390 [==============================] - 29s 75ms/step - loss: 0.2599 - acc: 0.9088 - val_loss: 0.5922 - val_acc: 0.8203

Epoch 00041: val_acc improved from 0.81830 to 0.82030, saving model to /content/best_weights.hdf5
Epoch 42/50
390/390 [==============================] - 30s 76ms/step - loss: 0.2661 - acc: 0.9058 - val_loss: 0.5709 - val_acc: 0.8273

Epoch 00042: val_acc improved from 0.82030 to 0.82730, saving model to /content/best_weights.hdf5
Epoch 43/50
390/390 [==============================] - 29s 75ms/step - loss: 0.2576 - acc: 0.9065 - val_loss: 0.6646 - val_acc: 0.8030

Epoch 00043: val_acc did not improve from 0.82730
Epoch 44/50
390/390 [==============================] - 30s 76ms/step - loss: 0.2548 - acc: 0.9097 - val_loss: 0.6797 - val_acc: 0.8032

Epoch 00044: val_acc did not improve from 0.82730
Epoch 45/50
390/390 [==============================] - 29s 76ms/step - loss: 0.2523 - acc: 0.9099 - val_loss: 0.6980 - val_acc: 0.8035

Epoch 00045: val_acc did not improve from 0.82730
Epoch 46/50
390/390 [==============================] - 30s 76ms/step - loss: 0.2429 - acc: 0.9133 - val_loss: 0.5845 - val_acc: 0.8283

Epoch 00046: val_acc improved from 0.82730 to 0.82830, saving model to /content/best_weights.hdf5
Epoch 47/50
390/390 [==============================] - 29s 76ms/step - loss: 0.2474 - acc: 0.9114 - val_loss: 0.6655 - val_acc: 0.8131

Epoch 00047: val_acc did not improve from 0.82830
Epoch 48/50
390/390 [==============================] - 30s 76ms/step - loss: 0.2401 - acc: 0.9153 - val_loss: 0.6603 - val_acc: 0.8098

Epoch 00048: val_acc did not improve from 0.82830
Epoch 49/50
390/390 [==============================] - 30s 76ms/step - loss: 0.2416 - acc: 0.9142 - val_loss: 0.6708 - val_acc: 0.8066

Epoch 00049: val_acc did not improve from 0.82830
Epoch 50/50
390/390 [==============================] - 30s 76ms/step - loss: 0.2358 - acc: 0.9153 - val_loss: 0.5687 - val_acc: 0.8324

Epoch 00050: val_acc improved from 0.82830 to 0.83240, saving model to /content/best_weights.hdf5
Model took 1485.89 seconds to train
__________________________________________________________________________________________________
dropout_14 (Dropout)            (None, 512)          0           dense_1[0][0]                    
__________________________________________________________________________________________________
dropout_15 (Dropout)            (None, 512)          0           dense_1[0][0]                    
__________________________________________________________________________________________________
dropout_16 (Dropout)            (None, 512)          0           dense_1[0][0]                    
__________________________________________________________________________________________________
dropout_17 (Dropout)            (None, 512)          0           dense_1[0][0]                    
__________________________________________________________________________________________________
dropout_19 (Dropout)            (None, 512)          0           dense_1[0][0]                    
__________________________________________________________________________________________________
dropout_18 (Dropout)            (None, 512)          0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 128)          65664       dropout_12[0][0]                 
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 128)          65664       dropout_13[0][0]                 
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 128)          65664       dropout_14[0][0]                 
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 128)          65664       dropout_15[0][0]                 
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 128)          65664       dropout_16[0][0]                 
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 128)          65664       dropout_17[0][0]                 
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 128)          65664       dropout_19[0][0]                 
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 128)          65664       dropout_18[0][0]                 
__________________________________________________________________________________________________
gender_output (Dense)           (None, 2)            258         dense_2[0][0]                    
__________________________________________________________________________________________________
image_quality_output (Dense)    (None, 3)            387         dense_3[0][0]                    
__________________________________________________________________________________________________
age_output (Dense)              (None, 5)            645         dense_4[0][0]                    
__________________________________________________________________________________________________
weight_output (Dense)           (None, 4)            516         dense_5[0][0]                    
__________________________________________________________________________________________________
bag_output (Dense)              (None, 3)            387         dense_6[0][0]                    
__________________________________________________________________________________________________
footwear_output (Dense)         (None, 3)            387         dense_7[0][0]                    
__________________________________________________________________________________________________
pose_output (Dense)             (None, 3)            387         dense_9[0][0]                    
__________________________________________________________________________________________________
emotion_output (Dense)          (None, 4)            516         dense_8[0][0]                    
==================================================================================================
Total params: 963,781
Trainable params: 963,205
Non-trainable params: 576
```

## Logs

```
Epoch 1/50
360/360 [==============================] - 79s 218ms/step - loss: 4.0410 - gender_output_loss: 0.6829 - image_quality_output_loss: 0.9966 - age_output_loss: 1.4390 - weight_output_loss: 1.0074 - bag_output_loss: 0.9325 - footwear_output_loss: 1.0045 - pose_output_loss: 0.9411 - emotion_output_loss: 0.9287 - gender_output_acc: 0.5610 - image_quality_output_acc: 0.5446 - age_output_acc: 0.3910 - weight_output_acc: 0.6273 - bag_output_acc: 0.5540 - footwear_output_acc: 0.5142 - pose_output_acc: 0.6147 - emotion_output_acc: 0.7115 - val_loss: 3.9962 - val_gender_output_loss: 0.6782 - val_image_quality_output_loss: 0.9815 - val_age_output_loss: 1.4234 - val_weight_output_loss: 1.0003 - val_bag_output_loss: 0.9238 - val_footwear_output_loss: 0.9896 - val_pose_output_loss: 0.9350 - val_emotion_output_loss: 0.9223 - val_gender_output_acc: 0.5758 - val_image_quality_output_acc: 0.5510 - val_age_output_acc: 0.4006 - val_weight_output_acc: 0.6338 - val_bag_output_acc: 0.5649 - val_footwear_output_acc: 0.5316 - val_pose_output_acc: 0.6172 - val_emotion_output_acc: 0.7129

Epoch 00001: val_loss improved from inf to 3.99625, saving model to /content/BEST_HVC_MODEL.h5
Epoch 2/50
360/360 [==============================] - 71s 196ms/step - loss: 3.9562 - gender_output_loss: 0.6567 - image_quality_output_loss: 0.9793 - age_output_loss: 1.4130 - weight_output_loss: 0.9863 - bag_output_loss: 0.9159 - footwear_output_loss: 0.9479 - pose_output_loss: 0.9282 - emotion_output_loss: 0.9118 - gender_output_acc: 0.5952 - image_quality_output_acc: 0.5503 - age_output_acc: 0.4000 - weight_output_acc: 0.6336 - bag_output_acc: 0.5635 - footwear_output_acc: 0.5582 - pose_output_acc: 0.6169 - emotion_output_acc: 0.7133 - val_loss: 3.9957 - val_gender_output_loss: 0.6620 - val_image_quality_output_loss: 1.0037 - val_age_output_loss: 1.4141 - val_weight_output_loss: 1.0042 - val_bag_output_loss: 0.9114 - val_footwear_output_loss: 0.9498 - val_pose_output_loss: 0.9336 - val_emotion_output_loss: 0.9300 - val_gender_output_acc: 0.6161 - val_image_quality_output_acc: 0.5511 - val_age_output_acc: 0.4005 - val_weight_output_acc: 0.6338 - val_bag_output_acc: 0.5657 - val_footwear_output_acc: 0.5671 - val_pose_output_acc: 0.6172 - val_emotion_output_acc: 0.7129

Epoch 00002: val_loss improved from 3.99625 to 3.99568, saving model to /content/BEST_HVC_MODEL.h5
Epoch 3/50
360/360 [==============================] - 71s 196ms/step - loss: 3.9109 - gender_output_loss: 0.6464 - image_quality_output_loss: 0.9529 - age_output_loss: 1.4086 - weight_output_loss: 0.9815 - bag_output_loss: 0.9081 - footwear_output_loss: 0.9312 - pose_output_loss: 0.9187 - emotion_output_loss: 0.9071 - gender_output_acc: 0.6080 - image_quality_output_acc: 0.5491 - age_output_acc: 0.4002 - weight_output_acc: 0.6336 - bag_output_acc: 0.5636 - footwear_output_acc: 0.5729 - pose_output_acc: 0.6174 - emotion_output_acc: 0.7130 - val_loss: 3.9138 - val_gender_output_loss: 0.6656 - val_image_quality_output_loss: 0.9219 - val_age_output_loss: 1.4143 - val_weight_output_loss: 1.0140 - val_bag_output_loss: 0.9161 - val_footwear_output_loss: 0.9482 - val_pose_output_loss: 0.9289 - val_emotion_output_loss: 0.9223 - val_gender_output_acc: 0.5935 - val_image_quality_output_acc: 0.5519 - val_age_output_acc: 0.4007 - val_weight_output_acc: 0.6338 - val_bag_output_acc: 0.5639 - val_footwear_output_acc: 0.5676 - val_pose_output_acc: 0.6172 - val_emotion_output_acc: 0.7129

Epoch 00003: val_loss improved from 3.99568 to 3.91376, saving model to /content/BEST_HVC_MODEL.h5
Epoch 4/50
360/360 [==============================] - 71s 197ms/step - loss: 3.8671 - gender_output_loss: 0.6316 - image_quality_output_loss: 0.9267 - age_output_loss: 1.4038 - weight_output_loss: 0.9827 - bag_output_loss: 0.9044 - footwear_output_loss: 0.9138 - pose_output_loss: 0.9083 - emotion_output_loss: 0.9017 - gender_output_acc: 0.6302 - image_quality_output_acc: 0.5515 - age_output_acc: 0.4005 - weight_output_acc: 0.6335 - bag_output_acc: 0.5650 - footwear_output_acc: 0.5813 - pose_output_acc: 0.6168 - emotion_output_acc: 0.7129 - val_loss: 3.8344 - val_gender_output_loss: 0.6320 - val_image_quality_output_loss: 0.9090 - val_age_output_loss: 1.3962 - val_weight_output_loss: 0.9734 - val_bag_output_loss: 0.9181 - val_footwear_output_loss: 0.9059 - val_pose_output_loss: 0.8954 - val_emotion_output_loss: 0.8954 - val_gender_output_acc: 0.6319 - val_image_quality_output_acc: 0.5563 - val_age_output_acc: 0.4015 - val_weight_output_acc: 0.6338 - val_bag_output_acc: 0.5671 - val_footwear_output_acc: 0.5847 - val_pose_output_acc: 0.6183 - val_emotion_output_acc: 0.7129

Epoch 00004: val_loss improved from 3.91376 to 3.83443, saving model to /content/BEST_HVC_MODEL.h5
Epoch 5/50
360/360 [==============================] - 70s 195ms/step - loss: 3.8371 - gender_output_loss: 0.6203 - image_quality_output_loss: 0.9170 - age_output_loss: 1.3972 - weight_output_loss: 0.9734 - bag_output_loss: 0.8975 - footwear_output_loss: 0.9038 - pose_output_loss: 0.8904 - emotion_output_loss: 0.9034 - gender_output_acc: 0.6367 - image_quality_output_acc: 0.5563 - age_output_acc: 0.4024 - weight_output_acc: 0.6334 - bag_output_acc: 0.5720 - footwear_output_acc: 0.5871 - pose_output_acc: 0.6190 - emotion_output_acc: 0.7128 - val_loss: 3.8229 - val_gender_output_loss: 0.6216 - val_image_quality_output_loss: 0.9078 - val_age_output_loss: 1.3939 - val_weight_output_loss: 0.9712 - val_bag_output_loss: 0.9204 - val_footwear_output_loss: 0.8996 - val_pose_output_loss: 0.8788 - val_emotion_output_loss: 0.8950 - val_gender_output_acc: 0.6466 - val_image_quality_output_acc: 0.5537 - val_age_output_acc: 0.4028 - val_weight_output_acc: 0.6339 - val_bag_output_acc: 0.5657 - val_footwear_output_acc: 0.5775 - val_pose_output_acc: 0.6218 - val_emotion_output_acc: 0.7129

Epoch 00005: val_loss improved from 3.83443 to 3.82287, saving model to /content/BEST_HVC_MODEL.h5
Epoch 6/50
360/360 [==============================] - 71s 197ms/step - loss: 3.7987 - gender_output_loss: 0.6108 - image_quality_output_loss: 0.9064 - age_output_loss: 1.3868 - weight_output_loss: 0.9698 - bag_output_loss: 0.8905 - footwear_output_loss: 0.8864 - pose_output_loss: 0.8685 - emotion_output_loss: 0.8999 - gender_output_acc: 0.6519 - image_quality_output_acc: 0.5583 - age_output_acc: 0.4032 - weight_output_acc: 0.6346 - bag_output_acc: 0.5754 - footwear_output_acc: 0.5956 - pose_output_acc: 0.6215 - emotion_output_acc: 0.7129 - val_loss: 3.9968 - val_gender_output_loss: 0.6805 - val_image_quality_output_loss: 0.9654 - val_age_output_loss: 1.4312 - val_weight_output_loss: 0.9961 - val_bag_output_loss: 0.9305 - val_footwear_output_loss: 0.9763 - val_pose_output_loss: 0.9754 - val_emotion_output_loss: 0.9217 - val_gender_output_acc: 0.5902 - val_image_quality_output_acc: 0.5488 - val_age_output_acc: 0.4028 - val_weight_output_acc: 0.6338 - val_bag_output_acc: 0.5647 - val_footwear_output_acc: 0.5270 - val_pose_output_acc: 0.5776 - val_emotion_output_acc: 0.7129

Epoch 00006: val_loss did not improve from 3.82287
Epoch 7/50
360/360 [==============================] - 71s 197ms/step - loss: 3.7611 - gender_output_loss: 0.5994 - image_quality_output_loss: 0.8964 - age_output_loss: 1.3802 - weight_output_loss: 0.9659 - bag_output_loss: 0.8826 - footwear_output_loss: 0.8708 - pose_output_loss: 0.8407 - emotion_output_loss: 0.8914 - gender_output_acc: 0.6601 - image_quality_output_acc: 0.5640 - age_output_acc: 0.4056 - weight_output_acc: 0.6354 - bag_output_acc: 0.5845 - footwear_output_acc: 0.6043 - pose_output_acc: 0.6368 - emotion_output_acc: 0.7127 - val_loss: 3.8472 - val_gender_output_loss: 0.6177 - val_image_quality_output_loss: 0.9781 - val_age_output_loss: 1.3802 - val_weight_output_loss: 0.9634 - val_bag_output_loss: 0.9017 - val_footwear_output_loss: 0.8760 - val_pose_output_loss: 0.8363 - val_emotion_output_loss: 0.8849 - val_gender_output_acc: 0.6509 - val_image_quality_output_acc: 0.5398 - val_age_output_acc: 0.4089 - val_weight_output_acc: 0.6352 - val_bag_output_acc: 0.5713 - val_footwear_output_acc: 0.6050 - val_pose_output_acc: 0.6378 - val_emotion_output_acc: 0.7129

Epoch 00007: val_loss did not improve from 3.82287
Epoch 8/50
360/360 [==============================] - 71s 197ms/step - loss: 3.7278 - gender_output_loss: 0.5893 - image_quality_output_loss: 0.8898 - age_output_loss: 1.3702 - weight_output_loss: 0.9581 - bag_output_loss: 0.8726 - footwear_output_loss: 0.8645 - pose_output_loss: 0.8161 - emotion_output_loss: 0.8886 - gender_output_acc: 0.6800 - image_quality_output_acc: 0.5677 - age_output_acc: 0.4107 - weight_output_acc: 0.6346 - bag_output_acc: 0.5867 - footwear_output_acc: 0.6089 - pose_output_acc: 0.6499 - emotion_output_acc: 0.7128 - val_loss: 3.8363 - val_gender_output_loss: 0.6693 - val_image_quality_output_loss: 0.8868 - val_age_output_loss: 1.4263 - val_weight_output_loss: 0.9829 - val_bag_output_loss: 0.9478 - val_footwear_output_loss: 0.8794 - val_pose_output_loss: 0.8596 - val_emotion_output_loss: 0.8884 - val_gender_output_acc: 0.6307 - val_image_quality_output_acc: 0.5666 - val_age_output_acc: 0.4058 - val_weight_output_acc: 0.6329 - val_bag_output_acc: 0.5686 - val_footwear_output_acc: 0.5994 - val_pose_output_acc: 0.6124 - val_emotion_output_acc: 0.7129

Epoch 00008: val_loss did not improve from 3.82287
Epoch 9/50
360/360 [==============================] - 70s 196ms/step - loss: 3.6768 - gender_output_loss: 0.5789 - image_quality_output_loss: 0.8785 - age_output_loss: 1.3497 - weight_output_loss: 0.9552 - bag_output_loss: 0.8650 - footwear_output_loss: 0.8538 - pose_output_loss: 0.7905 - emotion_output_loss: 0.8785 - gender_output_acc: 0.6798 - image_quality_output_acc: 0.5772 - age_output_acc: 0.4163 - weight_output_acc: 0.6357 - bag_output_acc: 0.5916 - footwear_output_acc: 0.6120 - pose_output_acc: 0.6595 - emotion_output_acc: 0.7128 - val_loss: 3.6503 - val_gender_output_loss: 0.5937 - val_image_quality_output_loss: 0.8719 - val_age_output_loss: 1.3412 - val_weight_output_loss: 0.9440 - val_bag_output_loss: 0.8761 - val_footwear_output_loss: 0.8501 - val_pose_output_loss: 0.7655 - val_emotion_output_loss: 0.8677 - val_gender_output_acc: 0.6760 - val_image_quality_output_acc: 0.5713 - val_age_output_acc: 0.4188 - val_weight_output_acc: 0.6387 - val_bag_output_acc: 0.5870 - val_footwear_output_acc: 0.6126 - val_pose_output_acc: 0.6726 - val_emotion_output_acc: 0.7129

Epoch 00009: val_loss improved from 3.82287 to 3.65032, saving model to /content/BEST_HVC_MODEL.h5
Epoch 10/50
360/360 [==============================] - 70s 196ms/step - loss: 3.6301 - gender_output_loss: 0.5727 - image_quality_output_loss: 0.8661 - age_output_loss: 1.3367 - weight_output_loss: 0.9448 - bag_output_loss: 0.8541 - footwear_output_loss: 0.8423 - pose_output_loss: 0.7660 - emotion_output_loss: 0.8696 - gender_output_acc: 0.6893 - image_quality_output_acc: 0.5842 - age_output_acc: 0.4227 - weight_output_acc: 0.6373 - bag_output_acc: 0.6023 - footwear_output_acc: 0.6125 - pose_output_acc: 0.6688 - emotion_output_acc: 0.7127 - val_loss: 3.5659 - val_gender_output_loss: 0.5833 - val_image_quality_output_loss: 0.8544 - val_age_output_loss: 1.3067 - val_weight_output_loss: 0.9299 - val_bag_output_loss: 0.8503 - val_footwear_output_loss: 0.8329 - val_pose_output_loss: 0.7393 - val_emotion_output_loss: 0.8518 - val_gender_output_acc: 0.6811 - val_image_quality_output_acc: 0.5884 - val_age_output_acc: 0.4442 - val_weight_output_acc: 0.6386 - val_bag_output_acc: 0.5957 - val_footwear_output_acc: 0.6207 - val_pose_output_acc: 0.6832 - val_emotion_output_acc: 0.7137

Epoch 00010: val_loss improved from 3.65032 to 3.56587, saving model to /content/BEST_HVC_MODEL.h5
Epoch 11/50
360/360 [==============================] - 70s 196ms/step - loss: 3.5564 - gender_output_loss: 0.5661 - image_quality_output_loss: 0.8450 - age_output_loss: 1.3033 - weight_output_loss: 0.9354 - bag_output_loss: 0.8462 - footwear_output_loss: 0.8326 - pose_output_loss: 0.7431 - emotion_output_loss: 0.8609 - gender_output_acc: 0.6961 - image_quality_output_acc: 0.5928 - age_output_acc: 0.4356 - weight_output_acc: 0.6380 - bag_output_acc: 0.6064 - footwear_output_acc: 0.6188 - pose_output_acc: 0.6780 - emotion_output_acc: 0.7126 - val_loss: 3.5307 - val_gender_output_loss: 0.5636 - val_image_quality_output_loss: 0.8691 - val_age_output_loss: 1.2742 - val_weight_output_loss: 0.9253 - val_bag_output_loss: 0.8394 - val_footwear_output_loss: 0.8282 - val_pose_output_loss: 0.7168 - val_emotion_output_loss: 0.8455 - val_gender_output_acc: 0.6997 - val_image_quality_output_acc: 0.6013 - val_age_output_acc: 0.4540 - val_weight_output_acc: 0.6411 - val_bag_output_acc: 0.6048 - val_footwear_output_acc: 0.6201 - val_pose_output_acc: 0.6970 - val_emotion_output_acc: 0.7136

Epoch 00011: val_loss improved from 3.56587 to 3.53071, saving model to /content/BEST_HVC_MODEL.h5
Epoch 12/50
360/360 [==============================] - 71s 196ms/step - loss: 3.4806 - gender_output_loss: 0.5607 - image_quality_output_loss: 0.8230 - age_output_loss: 1.2704 - weight_output_loss: 0.9270 - bag_output_loss: 0.8396 - footwear_output_loss: 0.8281 - pose_output_loss: 0.7195 - emotion_output_loss: 0.8420 - gender_output_acc: 0.7047 - image_quality_output_acc: 0.6064 - age_output_acc: 0.4533 - weight_output_acc: 0.6413 - bag_output_acc: 0.6138 - footwear_output_acc: 0.6198 - pose_output_acc: 0.6918 - emotion_output_acc: 0.7144 - val_loss: 3.4641 - val_gender_output_loss: 0.5541 - val_image_quality_output_loss: 0.8825 - val_age_output_loss: 1.2247 - val_weight_output_loss: 0.9085 - val_bag_output_loss: 0.8362 - val_footwear_output_loss: 0.8047 - val_pose_output_loss: 0.6875 - val_emotion_output_loss: 0.8260 - val_gender_output_acc: 0.7088 - val_image_quality_output_acc: 0.5910 - val_age_output_acc: 0.4861 - val_weight_output_acc: 0.6453 - val_bag_output_acc: 0.6118 - val_footwear_output_acc: 0.6331 - val_pose_output_acc: 0.7020 - val_emotion_output_acc: 0.7156

Epoch 00012: val_loss improved from 3.53071 to 3.46407, saving model to /content/BEST_HVC_MODEL.h5
Epoch 13/50
360/360 [==============================] - 71s 197ms/step - loss: 3.3560 - gender_output_loss: 0.5522 - image_quality_output_loss: 0.7827 - age_output_loss: 1.2148 - weight_output_loss: 0.9097 - bag_output_loss: 0.8219 - footwear_output_loss: 0.8077 - pose_output_loss: 0.6967 - emotion_output_loss: 0.8309 - gender_output_acc: 0.7126 - image_quality_output_acc: 0.6343 - age_output_acc: 0.4809 - weight_output_acc: 0.6470 - bag_output_acc: 0.6251 - footwear_output_acc: 0.6319 - pose_output_acc: 0.7008 - emotion_output_acc: 0.7170 - val_loss: 3.2132 - val_gender_output_loss: 0.5369 - val_image_quality_output_loss: 0.7660 - val_age_output_loss: 1.1367 - val_weight_output_loss: 0.8788 - val_bag_output_loss: 0.8065 - val_footwear_output_loss: 0.7808 - val_pose_output_loss: 0.6654 - val_emotion_output_loss: 0.7932 - val_gender_output_acc: 0.7304 - val_image_quality_output_acc: 0.6516 - val_age_output_acc: 0.5217 - val_weight_output_acc: 0.6468 - val_bag_output_acc: 0.6253 - val_footwear_output_acc: 0.6497 - val_pose_output_acc: 0.7255 - val_emotion_output_acc: 0.7168

Epoch 00013: val_loss improved from 3.46407 to 3.21316, saving model to /content/BEST_HVC_MODEL.h5
Epoch 14/50
360/360 [==============================] - 71s 197ms/step - loss: 3.2158 - gender_output_loss: 0.5410 - image_quality_output_loss: 0.7423 - age_output_loss: 1.1482 - weight_output_loss: 0.8855 - bag_output_loss: 0.8123 - footwear_output_loss: 0.7932 - pose_output_loss: 0.6709 - emotion_output_loss: 0.8065 - gender_output_acc: 0.7204 - image_quality_output_acc: 0.6549 - age_output_acc: 0.5188 - weight_output_acc: 0.6523 - bag_output_acc: 0.6309 - footwear_output_acc: 0.6369 - pose_output_acc: 0.7146 - emotion_output_acc: 0.7168 - val_loss: 2.8715 - val_gender_output_loss: 0.5071 - val_image_quality_output_loss: 0.6383 - val_age_output_loss: 1.0039 - val_weight_output_loss: 0.8361 - val_bag_output_loss: 0.7520 - val_footwear_output_loss: 0.7397 - val_pose_output_loss: 0.6023 - val_emotion_output_loss: 0.7487 - val_gender_output_acc: 0.7415 - val_image_quality_output_acc: 0.7156 - val_age_output_acc: 0.5864 - val_weight_output_acc: 0.6676 - val_bag_output_acc: 0.6630 - val_footwear_output_acc: 0.6659 - val_pose_output_acc: 0.7562 - val_emotion_output_acc: 0.7326

Epoch 00014: val_loss improved from 3.21316 to 2.87152, saving model to /content/BEST_HVC_MODEL.h5
Epoch 15/50
360/360 [==============================] - 71s 197ms/step - loss: 3.0553 - gender_output_loss: 0.5255 - image_quality_output_loss: 0.6931 - age_output_loss: 1.0746 - weight_output_loss: 0.8692 - bag_output_loss: 0.7850 - footwear_output_loss: 0.7745 - pose_output_loss: 0.6448 - emotion_output_loss: 0.7824 - gender_output_acc: 0.7306 - image_quality_output_acc: 0.6874 - age_output_acc: 0.5479 - weight_output_acc: 0.6573 - bag_output_acc: 0.6437 - footwear_output_acc: 0.6516 - pose_output_acc: 0.7292 - emotion_output_acc: 0.7233 - val_loss: 2.7351 - val_gender_output_loss: 0.4949 - val_image_quality_output_loss: 0.6177 - val_age_output_loss: 0.9271 - val_weight_output_loss: 0.8109 - val_bag_output_loss: 0.7317 - val_footwear_output_loss: 0.7218 - val_pose_output_loss: 0.5719 - val_emotion_output_loss: 0.7249 - val_gender_output_acc: 0.7518 - val_image_quality_output_acc: 0.7284 - val_age_output_acc: 0.6293 - val_weight_output_acc: 0.6764 - val_bag_output_acc: 0.6750 - val_footwear_output_acc: 0.6823 - val_pose_output_acc: 0.7708 - val_emotion_output_acc: 0.7322

Epoch 00015: val_loss improved from 2.87152 to 2.73509, saving model to /content/BEST_HVC_MODEL.h5
Epoch 16/50
360/360 [==============================] - 71s 196ms/step - loss: 2.8869 - gender_output_loss: 0.5136 - image_quality_output_loss: 0.6398 - age_output_loss: 1.0026 - weight_output_loss: 0.8407 - bag_output_loss: 0.7597 - footwear_output_loss: 0.7542 - pose_output_loss: 0.6197 - emotion_output_loss: 0.7521 - gender_output_acc: 0.7442 - image_quality_output_acc: 0.7135 - age_output_acc: 0.5876 - weight_output_acc: 0.6682 - bag_output_acc: 0.6562 - footwear_output_acc: 0.6618 - pose_output_acc: 0.7392 - emotion_output_acc: 0.7305 - val_loss: 2.5570 - val_gender_output_loss: 0.4820 - val_image_quality_output_loss: 0.5445 - val_age_output_loss: 0.8568 - val_weight_output_loss: 0.7848 - val_bag_output_loss: 0.7098 - val_footwear_output_loss: 0.7207 - val_pose_output_loss: 0.5540 - val_emotion_output_loss: 0.6916 - val_gender_output_acc: 0.7657 - val_image_quality_output_acc: 0.7685 - val_age_output_acc: 0.6647 - val_weight_output_acc: 0.6891 - val_bag_output_acc: 0.6824 - val_footwear_output_acc: 0.6759 - val_pose_output_acc: 0.7749 - val_emotion_output_acc: 0.7454

Epoch 00016: val_loss improved from 2.73509 to 2.55696, saving model to /content/BEST_HVC_MODEL.h5
Epoch 17/50
360/360 [==============================] - 71s 196ms/step - loss: 2.7068 - gender_output_loss: 0.5058 - image_quality_output_loss: 0.5901 - age_output_loss: 0.9150 - weight_output_loss: 0.8122 - bag_output_loss: 0.7366 - footwear_output_loss: 0.7354 - pose_output_loss: 0.5838 - emotion_output_loss: 0.7268 - gender_output_acc: 0.7451 - image_quality_output_acc: 0.7348 - age_output_acc: 0.6257 - weight_output_acc: 0.6748 - bag_output_acc: 0.6730 - footwear_output_acc: 0.6678 - pose_output_acc: 0.7560 - emotion_output_acc: 0.7372 - val_loss: 2.4711 - val_gender_output_loss: 0.4890 - val_image_quality_output_loss: 0.5316 - val_age_output_loss: 0.8045 - val_weight_output_loss: 0.7710 - val_bag_output_loss: 0.7013 - val_footwear_output_loss: 0.6942 - val_pose_output_loss: 0.5529 - val_emotion_output_loss: 0.6756 - val_gender_output_acc: 0.7702 - val_image_quality_output_acc: 0.7782 - val_age_output_acc: 0.6764 - val_weight_output_acc: 0.6923 - val_bag_output_acc: 0.6902 - val_footwear_output_acc: 0.7022 - val_pose_output_acc: 0.7820 - val_emotion_output_acc: 0.7514

Epoch 00017: val_loss improved from 2.55696 to 2.47109, saving model to /content/BEST_HVC_MODEL.h5
Epoch 18/50
360/360 [==============================] - 71s 197ms/step - loss: 2.5289 - gender_output_loss: 0.4808 - image_quality_output_loss: 0.5433 - age_output_loss: 0.8363 - weight_output_loss: 0.7739 - bag_output_loss: 0.7102 - footwear_output_loss: 0.7037 - pose_output_loss: 0.5628 - emotion_output_loss: 0.6900 - gender_output_acc: 0.7635 - image_quality_output_acc: 0.7623 - age_output_acc: 0.6615 - weight_output_acc: 0.6930 - bag_output_acc: 0.6803 - footwear_output_acc: 0.6872 - pose_output_acc: 0.7628 - emotion_output_acc: 0.7486 - val_loss: 2.1432 - val_gender_output_loss: 0.4303 - val_image_quality_output_loss: 0.4503 - val_age_output_loss: 0.6693 - val_weight_output_loss: 0.7030 - val_bag_output_loss: 0.6404 - val_footwear_output_loss: 0.6361 - val_pose_output_loss: 0.4814 - val_emotion_output_loss: 0.6057 - val_gender_output_acc: 0.7977 - val_image_quality_output_acc: 0.8218 - val_age_output_acc: 0.7477 - val_weight_output_acc: 0.7234 - val_bag_output_acc: 0.7234 - val_footwear_output_acc: 0.7279 - val_pose_output_acc: 0.8176 - val_emotion_output_acc: 0.7786

Epoch 00018: val_loss improved from 2.47109 to 2.14322, saving model to /content/BEST_HVC_MODEL.h5
Epoch 19/50
360/360 [==============================] - 71s 198ms/step - loss: 2.3519 - gender_output_loss: 0.4605 - image_quality_output_loss: 0.4959 - age_output_loss: 0.7576 - weight_output_loss: 0.7463 - bag_output_loss: 0.6764 - footwear_output_loss: 0.6781 - pose_output_loss: 0.5353 - emotion_output_loss: 0.6539 - gender_output_acc: 0.7743 - image_quality_output_acc: 0.7897 - age_output_acc: 0.6964 - weight_output_acc: 0.7012 - bag_output_acc: 0.7010 - footwear_output_acc: 0.7039 - pose_output_acc: 0.7744 - emotion_output_acc: 0.7567 - val_loss: 1.8242 - val_gender_output_loss: 0.3997 - val_image_quality_output_loss: 0.3484 - val_age_output_loss: 0.5420 - val_weight_output_loss: 0.6428 - val_bag_output_loss: 0.5847 - val_footwear_output_loss: 0.5856 - val_pose_output_loss: 0.4363 - val_emotion_output_loss: 0.5473 - val_gender_output_acc: 0.8147 - val_image_quality_output_acc: 0.8696 - val_age_output_acc: 0.8093 - val_weight_output_acc: 0.7498 - val_bag_output_acc: 0.7537 - val_footwear_output_acc: 0.7549 - val_pose_output_acc: 0.8333 - val_emotion_output_acc: 0.8076

Epoch 00019: val_loss improved from 2.14322 to 1.82417, saving model to /content/BEST_HVC_MODEL.h5
Epoch 20/50
360/360 [==============================] - 70s 196ms/step - loss: 2.1873 - gender_output_loss: 0.4461 - image_quality_output_loss: 0.4571 - age_output_loss: 0.6823 - weight_output_loss: 0.7157 - bag_output_loss: 0.6437 - footwear_output_loss: 0.6578 - pose_output_loss: 0.4995 - emotion_output_loss: 0.6207 - gender_output_acc: 0.7829 - image_quality_output_acc: 0.8041 - age_output_acc: 0.7273 - weight_output_acc: 0.7144 - bag_output_acc: 0.7165 - footwear_output_acc: 0.7100 - pose_output_acc: 0.7970 - emotion_output_acc: 0.7716 - val_loss: 1.6616 - val_gender_output_loss: 0.3860 - val_image_quality_output_loss: 0.3090 - val_age_output_loss: 0.4767 - val_weight_output_loss: 0.6054 - val_bag_output_loss: 0.5532 - val_footwear_output_loss: 0.5640 - val_pose_output_loss: 0.4010 - val_emotion_output_loss: 0.5006 - val_gender_output_acc: 0.8302 - val_image_quality_output_acc: 0.8847 - val_age_output_acc: 0.8359 - val_weight_output_acc: 0.7732 - val_bag_output_acc: 0.7615 - val_footwear_output_acc: 0.7684 - val_pose_output_acc: 0.8546 - val_emotion_output_acc: 0.8141

Epoch 00020: val_loss improved from 1.82417 to 1.66165, saving model to /content/BEST_HVC_MODEL.h5
Epoch 21/50
360/360 [==============================] - 71s 196ms/step - loss: 2.0350 - gender_output_loss: 0.4209 - image_quality_output_loss: 0.4099 - age_output_loss: 0.6337 - weight_output_loss: 0.6778 - bag_output_loss: 0.6159 - footwear_output_loss: 0.6229 - pose_output_loss: 0.4760 - emotion_output_loss: 0.5789 - gender_output_acc: 0.7957 - image_quality_output_acc: 0.8309 - age_output_acc: 0.7503 - weight_output_acc: 0.7248 - bag_output_acc: 0.7298 - footwear_output_acc: 0.7280 - pose_output_acc: 0.8090 - emotion_output_acc: 0.7842 - val_loss: 1.6506 - val_gender_output_loss: 0.3686 - val_image_quality_output_loss: 0.3070 - val_age_output_loss: 0.4864 - val_weight_output_loss: 0.5949 - val_bag_output_loss: 0.5427 - val_footwear_output_loss: 0.5428 - val_pose_output_loss: 0.3951 - val_emotion_output_loss: 0.4944 - val_gender_output_acc: 0.8326 - val_image_quality_output_acc: 0.8832 - val_age_output_acc: 0.8277 - val_weight_output_acc: 0.7769 - val_bag_output_acc: 0.7628 - val_footwear_output_acc: 0.7781 - val_pose_output_acc: 0.8537 - val_emotion_output_acc: 0.8269

Epoch 00021: val_loss improved from 1.66165 to 1.65064, saving model to /content/BEST_HVC_MODEL.h5
Epoch 22/50
360/360 [==============================] - 70s 195ms/step - loss: 1.8664 - gender_output_loss: 0.3988 - image_quality_output_loss: 0.3658 - age_output_loss: 0.5588 - weight_output_loss: 0.6400 - bag_output_loss: 0.5808 - footwear_output_loss: 0.5989 - pose_output_loss: 0.4476 - emotion_output_loss: 0.5544 - gender_output_acc: 0.8115 - image_quality_output_acc: 0.8482 - age_output_acc: 0.7828 - weight_output_acc: 0.7457 - bag_output_acc: 0.7468 - footwear_output_acc: 0.7433 - pose_output_acc: 0.8187 - emotion_output_acc: 0.7951 - val_loss: 1.4855 - val_gender_output_loss: 0.3482 - val_image_quality_output_loss: 0.2705 - val_age_output_loss: 0.4115 - val_weight_output_loss: 0.5515 - val_bag_output_loss: 0.5108 - val_footwear_output_loss: 0.5202 - val_pose_output_loss: 0.3770 - val_emotion_output_loss: 0.4522 - val_gender_output_acc: 0.8472 - val_image_quality_output_acc: 0.8954 - val_age_output_acc: 0.8576 - val_weight_output_acc: 0.7837 - val_bag_output_acc: 0.7832 - val_footwear_output_acc: 0.7859 - val_pose_output_acc: 0.8585 - val_emotion_output_acc: 0.8404

Epoch 00022: val_loss improved from 1.65064 to 1.48553, saving model to /content/BEST_HVC_MODEL.h5
Epoch 23/50
360/360 [==============================] - 70s 195ms/step - loss: 1.7741 - gender_output_loss: 0.3801 - image_quality_output_loss: 0.3453 - age_output_loss: 0.5280 - weight_output_loss: 0.6189 - bag_output_loss: 0.5566 - footwear_output_loss: 0.5745 - pose_output_loss: 0.4338 - emotion_output_loss: 0.5192 - gender_output_acc: 0.8206 - image_quality_output_acc: 0.8571 - age_output_acc: 0.7956 - weight_output_acc: 0.7530 - bag_output_acc: 0.7587 - footwear_output_acc: 0.7525 - pose_output_acc: 0.8274 - emotion_output_acc: 0.8050 - val_loss: 1.5585 - val_gender_output_loss: 0.3362 - val_image_quality_output_loss: 0.3177 - val_age_output_loss: 0.4431 - val_weight_output_loss: 0.5491 - val_bag_output_loss: 0.5075 - val_footwear_output_loss: 0.5179 - val_pose_output_loss: 0.3682 - val_emotion_output_loss: 0.4531 - val_gender_output_acc: 0.8513 - val_image_quality_output_acc: 0.8805 - val_age_output_acc: 0.8444 - val_weight_output_acc: 0.7944 - val_bag_output_acc: 0.7860 - val_footwear_output_acc: 0.7946 - val_pose_output_acc: 0.8643 - val_emotion_output_acc: 0.8379

Epoch 00023: val_loss did not improve from 1.48553
Epoch 24/50
360/360 [==============================] - 70s 196ms/step - loss: 1.6542 - gender_output_loss: 0.3630 - image_quality_output_loss: 0.3180 - age_output_loss: 0.4841 - weight_output_loss: 0.5899 - bag_output_loss: 0.5262 - footwear_output_loss: 0.5443 - pose_output_loss: 0.4042 - emotion_output_loss: 0.4911 - gender_output_acc: 0.8323 - image_quality_output_acc: 0.8704 - age_output_acc: 0.8150 - weight_output_acc: 0.7653 - bag_output_acc: 0.7769 - footwear_output_acc: 0.7668 - pose_output_acc: 0.8332 - emotion_output_acc: 0.8144 - val_loss: 1.1187 - val_gender_output_loss: 0.2843 - val_image_quality_output_loss: 0.1856 - val_age_output_loss: 0.2761 - val_weight_output_loss: 0.4639 - val_bag_output_loss: 0.4134 - val_footwear_output_loss: 0.4368 - val_pose_output_loss: 0.2900 - val_emotion_output_loss: 0.3685 - val_gender_output_acc: 0.8913 - val_image_quality_output_acc: 0.9458 - val_age_output_acc: 0.9200 - val_weight_output_acc: 0.8348 - val_bag_output_acc: 0.8430 - val_footwear_output_acc: 0.8418 - val_pose_output_acc: 0.8998 - val_emotion_output_acc: 0.8812

Epoch 00024: val_loss improved from 1.48553 to 1.11868, saving model to /content/BEST_HVC_MODEL.h5
Epoch 25/50
360/360 [==============================] - 70s 196ms/step - loss: 1.5373 - gender_output_loss: 0.3412 - image_quality_output_loss: 0.3004 - age_output_loss: 0.4323 - weight_output_loss: 0.5549 - bag_output_loss: 0.5033 - footwear_output_loss: 0.5190 - pose_output_loss: 0.3822 - emotion_output_loss: 0.4566 - gender_output_acc: 0.8437 - image_quality_output_acc: 0.8784 - age_output_acc: 0.8320 - weight_output_acc: 0.7799 - bag_output_acc: 0.7869 - footwear_output_acc: 0.7820 - pose_output_acc: 0.8482 - emotion_output_acc: 0.8285 - val_loss: 1.1905 - val_gender_output_loss: 0.2839 - val_image_quality_output_loss: 0.2092 - val_age_output_loss: 0.3253 - val_weight_output_loss: 0.4649 - val_bag_output_loss: 0.4162 - val_footwear_output_loss: 0.4279 - val_pose_output_loss: 0.2943 - val_emotion_output_loss: 0.3667 - val_gender_output_acc: 0.8825 - val_image_quality_output_acc: 0.9299 - val_age_output_acc: 0.8950 - val_weight_output_acc: 0.8379 - val_bag_output_acc: 0.8363 - val_footwear_output_acc: 0.8389 - val_pose_output_acc: 0.8946 - val_emotion_output_acc: 0.8721

Epoch 00025: val_loss did not improve from 1.11868
Epoch 26/50
360/360 [==============================] - 72s 200ms/step - loss: 1.4436 - gender_output_loss: 0.3306 - image_quality_output_loss: 0.2725 - age_output_loss: 0.4135 - weight_output_loss: 0.5232 - bag_output_loss: 0.4758 - footwear_output_loss: 0.4919 - pose_output_loss: 0.3582 - emotion_output_loss: 0.4246 - gender_output_acc: 0.8496 - image_quality_output_acc: 0.8915 - age_output_acc: 0.8427 - weight_output_acc: 0.7919 - bag_output_acc: 0.8008 - footwear_output_acc: 0.7887 - pose_output_acc: 0.8549 - emotion_output_acc: 0.8423 - val_loss: 1.0280 - val_gender_output_loss: 0.2731 - val_image_quality_output_loss: 0.1698 - val_age_output_loss: 0.2549 - val_weight_output_loss: 0.4159 - val_bag_output_loss: 0.4103 - val_footwear_output_loss: 0.4047 - val_pose_output_loss: 0.2606 - val_emotion_output_loss: 0.3211 - val_gender_output_acc: 0.8846 - val_image_quality_output_acc: 0.9438 - val_age_output_acc: 0.9220 - val_weight_output_acc: 0.8478 - val_bag_output_acc: 0.8285 - val_footwear_output_acc: 0.8462 - val_pose_output_acc: 0.9095 - val_emotion_output_acc: 0.8862

Epoch 00026: val_loss improved from 1.11868 to 1.02795, saving model to /content/BEST_HVC_MODEL.h5
Epoch 27/50
360/360 [==============================] - 71s 197ms/step - loss: 1.3488 - gender_output_loss: 0.3124 - image_quality_output_loss: 0.2463 - age_output_loss: 0.3731 - weight_output_loss: 0.5041 - bag_output_loss: 0.4560 - footwear_output_loss: 0.4744 - pose_output_loss: 0.3513 - emotion_output_loss: 0.4060 - gender_output_acc: 0.8596 - image_quality_output_acc: 0.9057 - age_output_acc: 0.8630 - weight_output_acc: 0.8016 - bag_output_acc: 0.8074 - footwear_output_acc: 0.8055 - pose_output_acc: 0.8637 - emotion_output_acc: 0.8450 - val_loss: 0.9924 - val_gender_output_loss: 0.2526 - val_image_quality_output_loss: 0.1779 - val_age_output_loss: 0.2397 - val_weight_output_loss: 0.3956 - val_bag_output_loss: 0.3752 - val_footwear_output_loss: 0.3790 - val_pose_output_loss: 0.2564 - val_emotion_output_loss: 0.3194 - val_gender_output_acc: 0.8961 - val_image_quality_output_acc: 0.9395 - val_age_output_acc: 0.9203 - val_weight_output_acc: 0.8508 - val_bag_output_acc: 0.8497 - val_footwear_output_acc: 0.8567 - val_pose_output_acc: 0.9109 - val_emotion_output_acc: 0.8932

Epoch 00027: val_loss improved from 1.02795 to 0.99243, saving model to /content/BEST_HVC_MODEL.h5
Epoch 28/50
360/360 [==============================] - 70s 195ms/step - loss: 1.2854 - gender_output_loss: 0.3065 - image_quality_output_loss: 0.2437 - age_output_loss: 0.3468 - weight_output_loss: 0.4816 - bag_output_loss: 0.4457 - footwear_output_loss: 0.4495 - pose_output_loss: 0.3188 - emotion_output_loss: 0.3889 - gender_output_acc: 0.8674 - image_quality_output_acc: 0.9048 - age_output_acc: 0.8696 - weight_output_acc: 0.8075 - bag_output_acc: 0.8122 - footwear_output_acc: 0.8170 - pose_output_acc: 0.8736 - emotion_output_acc: 0.8533 - val_loss: 0.7777 - val_gender_output_loss: 0.2145 - val_image_quality_output_loss: 0.1202 - val_age_output_loss: 0.1743 - val_weight_output_loss: 0.3453 - val_bag_output_loss: 0.3173 - val_footwear_output_loss: 0.3291 - val_pose_output_loss: 0.2061 - val_emotion_output_loss: 0.2559 - val_gender_output_acc: 0.9253 - val_image_quality_output_acc: 0.9687 - val_age_output_acc: 0.9535 - val_weight_output_acc: 0.8827 - val_bag_output_acc: 0.8810 - val_footwear_output_acc: 0.8845 - val_pose_output_acc: 0.9391 - val_emotion_output_acc: 0.9185

Epoch 00028: val_loss improved from 0.99243 to 0.77769, saving model to /content/BEST_HVC_MODEL.h5
Epoch 29/50
360/360 [==============================] - 70s 195ms/step - loss: 1.2231 - gender_output_loss: 0.2946 - image_quality_output_loss: 0.2208 - age_output_loss: 0.3442 - weight_output_loss: 0.4573 - bag_output_loss: 0.4155 - footwear_output_loss: 0.4272 - pose_output_loss: 0.3071 - emotion_output_loss: 0.3662 - gender_output_acc: 0.8668 - image_quality_output_acc: 0.9159 - age_output_acc: 0.8790 - weight_output_acc: 0.8146 - bag_output_acc: 0.8243 - footwear_output_acc: 0.8236 - pose_output_acc: 0.8807 - emotion_output_acc: 0.8598 - val_loss: 0.7850 - val_gender_output_loss: 0.2208 - val_image_quality_output_loss: 0.1194 - val_age_output_loss: 0.1820 - val_weight_output_loss: 0.3465 - val_bag_output_loss: 0.3097 - val_footwear_output_loss: 0.3245 - val_pose_output_loss: 0.2101 - val_emotion_output_loss: 0.2606 - val_gender_output_acc: 0.9136 - val_image_quality_output_acc: 0.9661 - val_age_output_acc: 0.9459 - val_weight_output_acc: 0.8727 - val_bag_output_acc: 0.8808 - val_footwear_output_acc: 0.8819 - val_pose_output_acc: 0.9334 - val_emotion_output_acc: 0.9211

Epoch 00029: val_loss did not improve from 0.77769
Epoch 30/50
360/360 [==============================] - 71s 196ms/step - loss: 1.1311 - gender_output_loss: 0.2881 - image_quality_output_loss: 0.2080 - age_output_loss: 0.3001 - weight_output_loss: 0.4344 - bag_output_loss: 0.3967 - footwear_output_loss: 0.4065 - pose_output_loss: 0.2948 - emotion_output_loss: 0.3361 - gender_output_acc: 0.8731 - image_quality_output_acc: 0.9163 - age_output_acc: 0.8918 - weight_output_acc: 0.8286 - bag_output_acc: 0.8373 - footwear_output_acc: 0.8394 - pose_output_acc: 0.8881 - emotion_output_acc: 0.8736 - val_loss: 0.8779 - val_gender_output_loss: 0.2279 - val_image_quality_output_loss: 0.1707 - val_age_output_loss: 0.2062 - val_weight_output_loss: 0.3494 - val_bag_output_loss: 0.3251 - val_footwear_output_loss: 0.3409 - val_pose_output_loss: 0.2182 - val_emotion_output_loss: 0.2703 - val_gender_output_acc: 0.9113 - val_image_quality_output_acc: 0.9385 - val_age_output_acc: 0.9319 - val_weight_output_acc: 0.8781 - val_bag_output_acc: 0.8771 - val_footwear_output_acc: 0.8745 - val_pose_output_acc: 0.9277 - val_emotion_output_acc: 0.9087

Epoch 00030: val_loss did not improve from 0.77769
Epoch 31/50
360/360 [==============================] - 71s 196ms/step - loss: 1.1085 - gender_output_loss: 0.2695 - image_quality_output_loss: 0.2111 - age_output_loss: 0.2921 - weight_output_loss: 0.4265 - bag_output_loss: 0.3752 - footwear_output_loss: 0.3983 - pose_output_loss: 0.2834 - emotion_output_loss: 0.3333 - gender_output_acc: 0.8812 - image_quality_output_acc: 0.9201 - age_output_acc: 0.8954 - weight_output_acc: 0.8299 - bag_output_acc: 0.8477 - footwear_output_acc: 0.8369 - pose_output_acc: 0.8896 - emotion_output_acc: 0.8752 - val_loss: 0.7481 - val_gender_output_loss: 0.2030 - val_image_quality_output_loss: 0.1331 - val_age_output_loss: 0.1672 - val_weight_output_loss: 0.3150 - val_bag_output_loss: 0.2884 - val_footwear_output_loss: 0.3075 - val_pose_output_loss: 0.1989 - val_emotion_output_loss: 0.2365 - val_gender_output_acc: 0.9248 - val_image_quality_output_acc: 0.9548 - val_age_output_acc: 0.9479 - val_weight_output_acc: 0.8847 - val_bag_output_acc: 0.8924 - val_footwear_output_acc: 0.8902 - val_pose_output_acc: 0.9361 - val_emotion_output_acc: 0.9160

Epoch 00031: val_loss improved from 0.77769 to 0.74808, saving model to /content/BEST_HVC_MODEL.h5
Epoch 32/50
360/360 [==============================] - 71s 196ms/step - loss: 1.0428 - gender_output_loss: 0.2593 - image_quality_output_loss: 0.1842 - age_output_loss: 0.2805 - weight_output_loss: 0.4049 - bag_output_loss: 0.3663 - footwear_output_loss: 0.3843 - pose_output_loss: 0.2658 - emotion_output_loss: 0.3144 - gender_output_acc: 0.8891 - image_quality_output_acc: 0.9299 - age_output_acc: 0.9007 - weight_output_acc: 0.8438 - bag_output_acc: 0.8523 - footwear_output_acc: 0.8464 - pose_output_acc: 0.8988 - emotion_output_acc: 0.8819 - val_loss: 0.5488 - val_gender_output_loss: 0.1658 - val_image_quality_output_loss: 0.0815 - val_age_output_loss: 0.1118 - val_weight_output_loss: 0.2637 - val_bag_output_loss: 0.2250 - val_footwear_output_loss: 0.2467 - val_pose_output_loss: 0.1497 - val_emotion_output_loss: 0.1835 - val_gender_output_acc: 0.9420 - val_image_quality_output_acc: 0.9775 - val_age_output_acc: 0.9739 - val_weight_output_acc: 0.9148 - val_bag_output_acc: 0.9233 - val_footwear_output_acc: 0.9215 - val_pose_output_acc: 0.9568 - val_emotion_output_acc: 0.9481

Epoch 00032: val_loss improved from 0.74808 to 0.54881, saving model to /content/BEST_HVC_MODEL.h5
Epoch 33/50
360/360 [==============================] - 71s 196ms/step - loss: 0.9853 - gender_output_loss: 0.2539 - image_quality_output_loss: 0.1774 - age_output_loss: 0.2568 - weight_output_loss: 0.3801 - bag_output_loss: 0.3561 - footwear_output_loss: 0.3585 - pose_output_loss: 0.2551 - emotion_output_loss: 0.3019 - gender_output_acc: 0.8904 - image_quality_output_acc: 0.9349 - age_output_acc: 0.9064 - weight_output_acc: 0.8520 - bag_output_acc: 0.8555 - footwear_output_acc: 0.8533 - pose_output_acc: 0.9034 - emotion_output_acc: 0.8893 - val_loss: 0.5852 - val_gender_output_loss: 0.1717 - val_image_quality_output_loss: 0.0867 - val_age_output_loss: 0.1313 - val_weight_output_loss: 0.2648 - val_bag_output_loss: 0.2450 - val_footwear_output_loss: 0.2486 - val_pose_output_loss: 0.1606 - val_emotion_output_loss: 0.1858 - val_gender_output_acc: 0.9348 - val_image_quality_output_acc: 0.9773 - val_age_output_acc: 0.9628 - val_weight_output_acc: 0.9080 - val_bag_output_acc: 0.9139 - val_footwear_output_acc: 0.9157 - val_pose_output_acc: 0.9490 - val_emotion_output_acc: 0.9465

Epoch 00033: val_loss did not improve from 0.54881
Epoch 34/50
360/360 [==============================] - 71s 196ms/step - loss: 0.9681 - gender_output_loss: 0.2530 - image_quality_output_loss: 0.1673 - age_output_loss: 0.2581 - weight_output_loss: 0.3725 - bag_output_loss: 0.3447 - footwear_output_loss: 0.3535 - pose_output_loss: 0.2593 - emotion_output_loss: 0.2960 - gender_output_acc: 0.8921 - image_quality_output_acc: 0.9375 - age_output_acc: 0.9071 - weight_output_acc: 0.8550 - bag_output_acc: 0.8612 - footwear_output_acc: 0.8581 - pose_output_acc: 0.9008 - emotion_output_acc: 0.8897 - val_loss: 0.5596 - val_gender_output_loss: 0.1663 - val_image_quality_output_loss: 0.0842 - val_age_output_loss: 0.1271 - val_weight_output_loss: 0.2513 - val_bag_output_loss: 0.2280 - val_footwear_output_loss: 0.2403 - val_pose_output_loss: 0.1496 - val_emotion_output_loss: 0.1773 - val_gender_output_acc: 0.9418 - val_image_quality_output_acc: 0.9746 - val_age_output_acc: 0.9628 - val_weight_output_acc: 0.9140 - val_bag_output_acc: 0.9198 - val_footwear_output_acc: 0.9174 - val_pose_output_acc: 0.9516 - val_emotion_output_acc: 0.9464

Epoch 00034: val_loss did not improve from 0.54881
Epoch 35/50
360/360 [==============================] - 71s 197ms/step - loss: 0.9398 - gender_output_loss: 0.2428 - image_quality_output_loss: 0.1735 - age_output_loss: 0.2488 - weight_output_loss: 0.3680 - bag_output_loss: 0.3246 - footwear_output_loss: 0.3429 - pose_output_loss: 0.2401 - emotion_output_loss: 0.2766 - gender_output_acc: 0.8953 - image_quality_output_acc: 0.9363 - age_output_acc: 0.9137 - weight_output_acc: 0.8623 - bag_output_acc: 0.8674 - footwear_output_acc: 0.8608 - pose_output_acc: 0.9072 - emotion_output_acc: 0.8978 - val_loss: 0.5021 - val_gender_output_loss: 0.1680 - val_image_quality_output_loss: 0.0770 - val_age_output_loss: 0.1014 - val_weight_output_loss: 0.2404 - val_bag_output_loss: 0.2146 - val_footwear_output_loss: 0.2178 - val_pose_output_loss: 0.1379 - val_emotion_output_loss: 0.1591 - val_gender_output_acc: 0.9394 - val_image_quality_output_acc: 0.9787 - val_age_output_acc: 0.9763 - val_weight_output_acc: 0.9222 - val_bag_output_acc: 0.9249 - val_footwear_output_acc: 0.9332 - val_pose_output_acc: 0.9613 - val_emotion_output_acc: 0.9541

Epoch 00035: val_loss improved from 0.54881 to 0.50211, saving model to /content/BEST_HVC_MODEL.h5
Epoch 36/50
360/360 [==============================] - 71s 197ms/step - loss: 0.8872 - gender_output_loss: 0.2335 - image_quality_output_loss: 0.1606 - age_output_loss: 0.2391 - weight_output_loss: 0.3427 - bag_output_loss: 0.3053 - footwear_output_loss: 0.3133 - pose_output_loss: 0.2295 - emotion_output_loss: 0.2673 - gender_output_acc: 0.9062 - image_quality_output_acc: 0.9437 - age_output_acc: 0.9168 - weight_output_acc: 0.8681 - bag_output_acc: 0.8787 - footwear_output_acc: 0.8773 - pose_output_acc: 0.9127 - emotion_output_acc: 0.9010 - val_loss: 0.4457 - val_gender_output_loss: 0.1419 - val_image_quality_output_loss: 0.0717 - val_age_output_loss: 0.0840 - val_weight_output_loss: 0.2115 - val_bag_output_loss: 0.1872 - val_footwear_output_loss: 0.1977 - val_pose_output_loss: 0.1278 - val_emotion_output_loss: 0.1462 - val_gender_output_acc: 0.9513 - val_image_quality_output_acc: 0.9803 - val_age_output_acc: 0.9777 - val_weight_output_acc: 0.9320 - val_bag_output_acc: 0.9378 - val_footwear_output_acc: 0.9374 - val_pose_output_acc: 0.9633 - val_emotion_output_acc: 0.9570

Epoch 00036: val_loss improved from 0.50211 to 0.44565, saving model to /content/BEST_HVC_MODEL.h5
Epoch 37/50
360/360 [==============================] - 71s 196ms/step - loss: 0.8694 - gender_output_loss: 0.2278 - image_quality_output_loss: 0.1476 - age_output_loss: 0.2374 - weight_output_loss: 0.3378 - bag_output_loss: 0.3111 - footwear_output_loss: 0.3211 - pose_output_loss: 0.2247 - emotion_output_loss: 0.2580 - gender_output_acc: 0.9032 - image_quality_output_acc: 0.9463 - age_output_acc: 0.9140 - weight_output_acc: 0.8682 - bag_output_acc: 0.8757 - footwear_output_acc: 0.8733 - pose_output_acc: 0.9114 - emotion_output_acc: 0.9052 - val_loss: 0.4063 - val_gender_output_loss: 0.1395 - val_image_quality_output_loss: 0.0637 - val_age_output_loss: 0.0799 - val_weight_output_loss: 0.1917 - val_bag_output_loss: 0.1801 - val_footwear_output_loss: 0.1780 - val_pose_output_loss: 0.1091 - val_emotion_output_loss: 0.1279 - val_gender_output_acc: 0.9490 - val_image_quality_output_acc: 0.9827 - val_age_output_acc: 0.9786 - val_weight_output_acc: 0.9384 - val_bag_output_acc: 0.9405 - val_footwear_output_acc: 0.9456 - val_pose_output_acc: 0.9676 - val_emotion_output_acc: 0.9647

Epoch 00037: val_loss improved from 0.44565 to 0.40634, saving model to /content/BEST_HVC_MODEL.h5
Epoch 38/50
360/360 [==============================] - 71s 198ms/step - loss: 0.8424 - gender_output_loss: 0.2183 - image_quality_output_loss: 0.1609 - age_output_loss: 0.2162 - weight_output_loss: 0.3335 - bag_output_loss: 0.2907 - footwear_output_loss: 0.3115 - pose_output_loss: 0.2218 - emotion_output_loss: 0.2406 - gender_output_acc: 0.9062 - image_quality_output_acc: 0.9421 - age_output_acc: 0.9257 - weight_output_acc: 0.8731 - bag_output_acc: 0.8845 - footwear_output_acc: 0.8788 - pose_output_acc: 0.9145 - emotion_output_acc: 0.9112 - val_loss: 0.3877 - val_gender_output_loss: 0.1264 - val_image_quality_output_loss: 0.0577 - val_age_output_loss: 0.0723 - val_weight_output_loss: 0.1928 - val_bag_output_loss: 0.1708 - val_footwear_output_loss: 0.1761 - val_pose_output_loss: 0.1105 - val_emotion_output_loss: 0.1250 - val_gender_output_acc: 0.9635 - val_image_quality_output_acc: 0.9845 - val_age_output_acc: 0.9824 - val_weight_output_acc: 0.9409 - val_bag_output_acc: 0.9466 - val_footwear_output_acc: 0.9500 - val_pose_output_acc: 0.9695 - val_emotion_output_acc: 0.9628

Epoch 00038: val_loss improved from 0.40634 to 0.38774, saving model to /content/BEST_HVC_MODEL.h5
Epoch 39/50
360/360 [==============================] - 71s 196ms/step - loss: 0.7918 - gender_output_loss: 0.2067 - image_quality_output_loss: 0.1480 - age_output_loss: 0.2053 - weight_output_loss: 0.3108 - bag_output_loss: 0.2886 - footwear_output_loss: 0.2862 - pose_output_loss: 0.2045 - emotion_output_loss: 0.2270 - gender_output_acc: 0.9110 - image_quality_output_acc: 0.9455 - age_output_acc: 0.9300 - weight_output_acc: 0.8828 - bag_output_acc: 0.8872 - footwear_output_acc: 0.8921 - pose_output_acc: 0.9225 - emotion_output_acc: 0.9174 - val_loss: 0.6577 - val_gender_output_loss: 0.1641 - val_image_quality_output_loss: 0.1229 - val_age_output_loss: 0.1666 - val_weight_output_loss: 0.2665 - val_bag_output_loss: 0.2517 - val_footwear_output_loss: 0.2407 - val_pose_output_loss: 0.1568 - val_emotion_output_loss: 0.1927 - val_gender_output_acc: 0.9418 - val_image_quality_output_acc: 0.9585 - val_age_output_acc: 0.9428 - val_weight_output_acc: 0.9034 - val_bag_output_acc: 0.8998 - val_footwear_output_acc: 0.9161 - val_pose_output_acc: 0.9476 - val_emotion_output_acc: 0.9356

Epoch 00039: val_loss did not improve from 0.38774
Epoch 40/50
360/360 [==============================] - 70s 195ms/step - loss: 0.7884 - gender_output_loss: 0.2064 - image_quality_output_loss: 0.1420 - age_output_loss: 0.2103 - weight_output_loss: 0.3032 - bag_output_loss: 0.2820 - footwear_output_loss: 0.2942 - pose_output_loss: 0.2059 - emotion_output_loss: 0.2245 - gender_output_acc: 0.9137 - image_quality_output_acc: 0.9496 - age_output_acc: 0.9267 - weight_output_acc: 0.8869 - bag_output_acc: 0.8878 - footwear_output_acc: 0.8837 - pose_output_acc: 0.9220 - emotion_output_acc: 0.9178 - val_loss: 0.4732 - val_gender_output_loss: 0.1425 - val_image_quality_output_loss: 0.0771 - val_age_output_loss: 0.1055 - val_weight_output_loss: 0.2102 - val_bag_output_loss: 0.2035 - val_footwear_output_loss: 0.1913 - val_pose_output_loss: 0.1220 - val_emotion_output_loss: 0.1458 - val_gender_output_acc: 0.9478 - val_image_quality_output_acc: 0.9775 - val_age_output_acc: 0.9709 - val_weight_output_acc: 0.9286 - val_bag_output_acc: 0.9269 - val_footwear_output_acc: 0.9377 - val_pose_output_acc: 0.9642 - val_emotion_output_acc: 0.9515

Epoch 00040: val_loss did not improve from 0.38774
Epoch 41/50
360/360 [==============================] - 71s 196ms/step - loss: 0.7650 - gender_output_loss: 0.2050 - image_quality_output_loss: 0.1374 - age_output_loss: 0.1997 - weight_output_loss: 0.3063 - bag_output_loss: 0.2725 - footwear_output_loss: 0.2798 - pose_output_loss: 0.1987 - emotion_output_loss: 0.2256 - gender_output_acc: 0.9161 - image_quality_output_acc: 0.9503 - age_output_acc: 0.9262 - weight_output_acc: 0.8832 - bag_output_acc: 0.8949 - footwear_output_acc: 0.8924 - pose_output_acc: 0.9266 - emotion_output_acc: 0.9184 - val_loss: 0.4650 - val_gender_output_loss: 0.1401 - val_image_quality_output_loss: 0.0791 - val_age_output_loss: 0.1061 - val_weight_output_loss: 0.2062 - val_bag_output_loss: 0.1911 - val_footwear_output_loss: 0.1876 - val_pose_output_loss: 0.1217 - val_emotion_output_loss: 0.1347 - val_gender_output_acc: 0.9492 - val_image_quality_output_acc: 0.9772 - val_age_output_acc: 0.9702 - val_weight_output_acc: 0.9369 - val_bag_output_acc: 0.9294 - val_footwear_output_acc: 0.9357 - val_pose_output_acc: 0.9602 - val_emotion_output_acc: 0.9607

Epoch 00041: val_loss did not improve from 0.38774
Epoch 42/50
360/360 [==============================] - 70s 195ms/step - loss: 0.7275 - gender_output_loss: 0.1944 - image_quality_output_loss: 0.1281 - age_output_loss: 0.1876 - weight_output_loss: 0.2927 - bag_output_loss: 0.2601 - footwear_output_loss: 0.2739 - pose_output_loss: 0.1900 - emotion_output_loss: 0.2184 - gender_output_acc: 0.9202 - image_quality_output_acc: 0.9528 - age_output_acc: 0.9371 - weight_output_acc: 0.8916 - bag_output_acc: 0.9009 - footwear_output_acc: 0.8946 - pose_output_acc: 0.9282 - emotion_output_acc: 0.9234 - val_loss: 0.4273 - val_gender_output_loss: 0.1374 - val_image_quality_output_loss: 0.0607 - val_age_output_loss: 0.0979 - val_weight_output_loss: 0.1998 - val_bag_output_loss: 0.1851 - val_footwear_output_loss: 0.1808 - val_pose_output_loss: 0.1127 - val_emotion_output_loss: 0.1285 - val_gender_output_acc: 0.9492 - val_image_quality_output_acc: 0.9832 - val_age_output_acc: 0.9731 - val_weight_output_acc: 0.9381 - val_bag_output_acc: 0.9305 - val_footwear_output_acc: 0.9420 - val_pose_output_acc: 0.9638 - val_emotion_output_acc: 0.9617

Epoch 00042: val_loss did not improve from 0.38774
Epoch 43/50
360/360 [==============================] - 70s 195ms/step - loss: 0.7243 - gender_output_loss: 0.1918 - image_quality_output_loss: 0.1308 - age_output_loss: 0.1900 - weight_output_loss: 0.2809 - bag_output_loss: 0.2615 - footwear_output_loss: 0.2627 - pose_output_loss: 0.1949 - emotion_output_loss: 0.2107 - gender_output_acc: 0.9223 - image_quality_output_acc: 0.9523 - age_output_acc: 0.9331 - weight_output_acc: 0.8934 - bag_output_acc: 0.8972 - footwear_output_acc: 0.9007 - pose_output_acc: 0.9285 - emotion_output_acc: 0.9244 - val_loss: 0.3465 - val_gender_output_loss: 0.1088 - val_image_quality_output_loss: 0.0622 - val_age_output_loss: 0.0669 - val_weight_output_loss: 0.1586 - val_bag_output_loss: 0.1508 - val_footwear_output_loss: 0.1472 - val_pose_output_loss: 0.0960 - val_emotion_output_loss: 0.1017 - val_gender_output_acc: 0.9656 - val_image_quality_output_acc: 0.9832 - val_age_output_acc: 0.9840 - val_weight_output_acc: 0.9563 - val_bag_output_acc: 0.9479 - val_footwear_output_acc: 0.9573 - val_pose_output_acc: 0.9762 - val_emotion_output_acc: 0.9735

Epoch 00043: val_loss improved from 0.38774 to 0.34647, saving model to /content/BEST_HVC_MODEL.h5
Epoch 44/50
360/360 [==============================] - 70s 195ms/step - loss: 0.6979 - gender_output_loss: 0.1867 - image_quality_output_loss: 0.1284 - age_output_loss: 0.1772 - weight_output_loss: 0.2676 - bag_output_loss: 0.2551 - footwear_output_loss: 0.2575 - pose_output_loss: 0.1884 - emotion_output_loss: 0.2076 - gender_output_acc: 0.9238 - image_quality_output_acc: 0.9556 - age_output_acc: 0.9377 - weight_output_acc: 0.8988 - bag_output_acc: 0.9010 - footwear_output_acc: 0.9019 - pose_output_acc: 0.9301 - emotion_output_acc: 0.9260 - val_loss: 0.4201 - val_gender_output_loss: 0.1206 - val_image_quality_output_loss: 0.0662 - val_age_output_loss: 0.0943 - val_weight_output_loss: 0.1827 - val_bag_output_loss: 0.1925 - val_footwear_output_loss: 0.1694 - val_pose_output_loss: 0.1075 - val_emotion_output_loss: 0.1297 - val_gender_output_acc: 0.9589 - val_image_quality_output_acc: 0.9814 - val_age_output_acc: 0.9718 - val_weight_output_acc: 0.9424 - val_bag_output_acc: 0.9247 - val_footwear_output_acc: 0.9474 - val_pose_output_acc: 0.9707 - val_emotion_output_acc: 0.9556

Epoch 00044: val_loss did not improve from 0.34647
Epoch 45/50
360/360 [==============================] - 69s 192ms/step - loss: 0.6841 - gender_output_loss: 0.1826 - image_quality_output_loss: 0.1258 - age_output_loss: 0.1775 - weight_output_loss: 0.2549 - bag_output_loss: 0.2414 - footwear_output_loss: 0.2472 - pose_output_loss: 0.1900 - emotion_output_loss: 0.2063 - gender_output_acc: 0.9257 - image_quality_output_acc: 0.9550 - age_output_acc: 0.9393 - weight_output_acc: 0.9045 - bag_output_acc: 0.9066 - footwear_output_acc: 0.9043 - pose_output_acc: 0.9299 - emotion_output_acc: 0.9258 - val_loss: 0.2851 - val_gender_output_loss: 0.0952 - val_image_quality_output_loss: 0.0434 - val_age_output_loss: 0.0535 - val_weight_output_loss: 0.1370 - val_bag_output_loss: 0.1234 - val_footwear_output_loss: 0.1311 - val_pose_output_loss: 0.0798 - val_emotion_output_loss: 0.0930 - val_gender_output_acc: 0.9759 - val_image_quality_output_acc: 0.9899 - val_age_output_acc: 0.9900 - val_weight_output_acc: 0.9617 - val_bag_output_acc: 0.9666 - val_footwear_output_acc: 0.9675 - val_pose_output_acc: 0.9786 - val_emotion_output_acc: 0.9792

Epoch 00045: val_loss improved from 0.34647 to 0.28511, saving model to /content/BEST_HVC_MODEL.h5
Epoch 46/50
360/360 [==============================] - 70s 194ms/step - loss: 0.6486 - gender_output_loss: 0.1762 - image_quality_output_loss: 0.1186 - age_output_loss: 0.1651 - weight_output_loss: 0.2676 - bag_output_loss: 0.2383 - footwear_output_loss: 0.2401 - pose_output_loss: 0.1662 - emotion_output_loss: 0.1843 - gender_output_acc: 0.9280 - image_quality_output_acc: 0.9589 - age_output_acc: 0.9442 - weight_output_acc: 0.8982 - bag_output_acc: 0.9079 - footwear_output_acc: 0.9086 - pose_output_acc: 0.9398 - emotion_output_acc: 0.9333 - val_loss: 0.2308 - val_gender_output_loss: 0.0803 - val_image_quality_output_loss: 0.0316 - val_age_output_loss: 0.0415 - val_weight_output_loss: 0.1205 - val_bag_output_loss: 0.1084 - val_footwear_output_loss: 0.1111 - val_pose_output_loss: 0.0614 - val_emotion_output_loss: 0.0731 - val_gender_output_acc: 0.9796 - val_image_quality_output_acc: 0.9938 - val_age_output_acc: 0.9924 - val_weight_output_acc: 0.9706 - val_bag_output_acc: 0.9720 - val_footwear_output_acc: 0.9717 - val_pose_output_acc: 0.9873 - val_emotion_output_acc: 0.9839

Epoch 00046: val_loss improved from 0.28511 to 0.23079, saving model to /content/BEST_HVC_MODEL.h5
Epoch 47/50
360/360 [==============================] - 70s 193ms/step - loss: 0.6565 - gender_output_loss: 0.1718 - image_quality_output_loss: 0.1237 - age_output_loss: 0.1693 - weight_output_loss: 0.2537 - bag_output_loss: 0.2311 - footwear_output_loss: 0.2401 - pose_output_loss: 0.1659 - emotion_output_loss: 0.1977 - gender_output_acc: 0.9297 - image_quality_output_acc: 0.9556 - age_output_acc: 0.9403 - weight_output_acc: 0.9049 - bag_output_acc: 0.9114 - footwear_output_acc: 0.9082 - pose_output_acc: 0.9367 - emotion_output_acc: 0.9289 - val_loss: 0.2097 - val_gender_output_loss: 0.0751 - val_image_quality_output_loss: 0.0305 - val_age_output_loss: 0.0379 - val_weight_output_loss: 0.1068 - val_bag_output_loss: 0.0942 - val_footwear_output_loss: 0.0992 - val_pose_output_loss: 0.0552 - val_emotion_output_loss: 0.0680 - val_gender_output_acc: 0.9807 - val_image_quality_output_acc: 0.9941 - val_age_output_acc: 0.9921 - val_weight_output_acc: 0.9747 - val_bag_output_acc: 0.9736 - val_footwear_output_acc: 0.9751 - val_pose_output_acc: 0.9882 - val_emotion_output_acc: 0.9866

Epoch 00047: val_loss improved from 0.23079 to 0.20968, saving model to /content/BEST_HVC_MODEL.h5
Epoch 48/50
360/360 [==============================] - 70s 194ms/step - loss: 0.6365 - gender_output_loss: 0.1760 - image_quality_output_loss: 0.1107 - age_output_loss: 0.1732 - weight_output_loss: 0.2519 - bag_output_loss: 0.2283 - footwear_output_loss: 0.2427 - pose_output_loss: 0.1665 - emotion_output_loss: 0.1705 - gender_output_acc: 0.9285 - image_quality_output_acc: 0.9606 - age_output_acc: 0.9406 - weight_output_acc: 0.9089 - bag_output_acc: 0.9142 - footwear_output_acc: 0.9076 - pose_output_acc: 0.9372 - emotion_output_acc: 0.9383 - val_loss: 0.2824 - val_gender_output_loss: 0.0979 - val_image_quality_output_loss: 0.0430 - val_age_output_loss: 0.0594 - val_weight_output_loss: 0.1329 - val_bag_output_loss: 0.1302 - val_footwear_output_loss: 0.1230 - val_pose_output_loss: 0.0760 - val_emotion_output_loss: 0.0787 - val_gender_output_acc: 0.9686 - val_image_quality_output_acc: 0.9886 - val_age_output_acc: 0.9855 - val_weight_output_acc: 0.9623 - val_bag_output_acc: 0.9552 - val_footwear_output_acc: 0.9636 - val_pose_output_acc: 0.9800 - val_emotion_output_acc: 0.9795

Epoch 00048: val_loss did not improve from 0.20968
Epoch 49/50
360/360 [==============================] - 70s 195ms/step - loss: 0.6319 - gender_output_loss: 0.1705 - image_quality_output_loss: 0.1114 - age_output_loss: 0.1727 - weight_output_loss: 0.2406 - bag_output_loss: 0.2257 - footwear_output_loss: 0.2339 - pose_output_loss: 0.1675 - emotion_output_loss: 0.1760 - gender_output_acc: 0.9304 - image_quality_output_acc: 0.9595 - age_output_acc: 0.9385 - weight_output_acc: 0.9090 - bag_output_acc: 0.9156 - footwear_output_acc: 0.9113 - pose_output_acc: 0.9361 - emotion_output_acc: 0.9364 - val_loss: 0.2409 - val_gender_output_loss: 0.0854 - val_image_quality_output_loss: 0.0367 - val_age_output_loss: 0.0524 - val_weight_output_loss: 0.1139 - val_bag_output_loss: 0.1053 - val_footwear_output_loss: 0.1016 - val_pose_output_loss: 0.0620 - val_emotion_output_loss: 0.0714 - val_gender_output_acc: 0.9724 - val_image_quality_output_acc: 0.9918 - val_age_output_acc: 0.9885 - val_weight_output_acc: 0.9708 - val_bag_output_acc: 0.9694 - val_footwear_output_acc: 0.9734 - val_pose_output_acc: 0.9850 - val_emotion_output_acc: 0.9845

Epoch 00049: val_loss did not improve from 0.20968
Epoch 50/50
360/360 [==============================] - 70s 196ms/step - loss: 0.6041 - gender_output_loss: 0.1765 - image_quality_output_loss: 0.1053 - age_output_loss: 0.1597 - weight_output_loss: 0.2366 - bag_output_loss: 0.2209 - footwear_output_loss: 0.2260 - pose_output_loss: 0.1694 - emotion_output_loss: 0.1639 - gender_output_acc: 0.9286 - image_quality_output_acc: 0.9632 - age_output_acc: 0.9443 - weight_output_acc: 0.9100 - bag_output_acc: 0.9160 - footwear_output_acc: 0.9163 - pose_output_acc: 0.9402 - emotion_output_acc: 0.9420 - val_loss: 0.2502 - val_gender_output_loss: 0.0831 - val_image_quality_output_loss: 0.0396 - val_age_output_loss: 0.0526 - val_weight_output_loss: 0.1166 - val_bag_output_loss: 0.1102 - val_footwear_output_loss: 0.1061 - val_pose_output_loss: 0.0719 - val_emotion_output_loss: 0.0707 - val_gender_output_acc: 0.9779 - val_image_quality_output_acc: 0.9903 - val_age_output_acc: 0.9876 - val_weight_output_acc: 0.9662 - val_bag_output_acc: 0.9681 - val_footwear_output_acc: 0.9728 - val_pose_output_acc: 0.9789 - val_emotion_output_acc: 0.9840

Epoch 00050: val_loss did not improve from 0.20968
```

## Plots
