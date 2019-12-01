# EIP-WEEK-2
EIP-WEEK-2 assignment --> Code #9 

## Train Accuracy
Model-1 (with Fully connected Layer) ~ 99.42 % 

## Test Accuracy
Model-2 (witn No Fully connected Layer) ~ 99.43 % (epoch --> 13)

## Approach
I have used 1x1 Conv layer 2 times to reduce the output shape twicein order to satisfy the criteria of < 15k parameters.
e.g
```
28x28x1   | 3x3x1x16    | 26
26x26x16  | 3x3x16x32   | 24
24x24x32  | 1x1x32x16   | 24

Maxpool()               | 12

12x12x16  | 3x3x16x16   | 10
10x10x16  | 1x1x16x10   | 10
10x10x10  | 3x3x10x10   | 8
8x8x10    | 3x3x10x10   | 6
6x6x10    | 3x3x10x10   | 4
4x4x10    | 4x4x10x10   | 1

Flatten()
softmax()
```

## Logs
```
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 7s 119us/step - loss: 0.0480 - acc: 0.9850 - val_loss: 0.0349 - val_acc: 0.9898

Epoch 00001: val_acc improved from -inf to 0.98980, saving model to /content/best_weights.hdf5
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022075055.
60000/60000 [==============================] - 6s 94us/step - loss: 0.0402 - acc: 0.9873 - val_loss: 0.0301 - val_acc: 0.9898

Epoch 00002: val_acc did not improve from 0.98980
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0017462165.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0339 - acc: 0.9894 - val_loss: 0.0345 - val_acc: 0.9887

Epoch 00003: val_acc did not improve from 0.98980
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0014443909.
60000/60000 [==============================] - 6s 95us/step - loss: 0.0297 - acc: 0.9905 - val_loss: 0.0233 - val_acc: 0.9919

Epoch 00004: val_acc improved from 0.98980 to 0.99190, saving model to /content/best_weights.hdf5
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0012315271.
60000/60000 [==============================] - 6s 95us/step - loss: 0.0295 - acc: 0.9901 - val_loss: 0.0234 - val_acc: 0.9931

Epoch 00005: val_acc improved from 0.99190 to 0.99310, saving model to /content/best_weights.hdf5
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0010733453.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0264 - acc: 0.9915 - val_loss: 0.0238 - val_acc: 0.9919

Epoch 00006: val_acc did not improve from 0.99310
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0009511731.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0252 - acc: 0.9920 - val_loss: 0.0217 - val_acc: 0.9928

Epoch 00007: val_acc did not improve from 0.99310
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.000853971.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0254 - acc: 0.9915 - val_loss: 0.0256 - val_acc: 0.9916

Epoch 00008: val_acc did not improve from 0.99310
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0007747934.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0235 - acc: 0.9923 - val_loss: 0.0229 - val_acc: 0.9932

Epoch 00009: val_acc improved from 0.99310 to 0.99320, saving model to /content/best_weights.hdf5
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007090522.
60000/60000 [==============================] - 6s 92us/step - loss: 0.0228 - acc: 0.9927 - val_loss: 0.0213 - val_acc: 0.9932

Epoch 00010: val_acc did not improve from 0.99320
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0006535948.
60000/60000 [==============================] - 6s 94us/step - loss: 0.0207 - acc: 0.9932 - val_loss: 0.0211 - val_acc: 0.9932

Epoch 00011: val_acc did not improve from 0.99320
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.0006061831.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0211 - acc: 0.9934 - val_loss: 0.0211 - val_acc: 0.9937

Epoch 00012: val_acc improved from 0.99320 to 0.99370, saving model to /content/best_weights.hdf5
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0005651846.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0198 - acc: 0.9939 - val_loss: 0.0193 - val_acc: 0.9942

Epoch 00013: val_acc improved from 0.99370 to 0.99420, saving model to /content/best_weights.hdf5
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005293806.
60000/60000 [==============================] - 6s 92us/step - loss: 0.0196 - acc: 0.9935 - val_loss: 0.0216 - val_acc: 0.9932

Epoch 00014: val_acc did not improve from 0.99420
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0004978427.
60000/60000 [==============================] - 6s 92us/step - loss: 0.0189 - acc: 0.9938 - val_loss: 0.0206 - val_acc: 0.9939

Epoch 00015: val_acc did not improve from 0.99420
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0004698512.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0196 - acc: 0.9934 - val_loss: 0.0222 - val_acc: 0.9931

Epoch 00016: val_acc did not improve from 0.99420
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.0004448399.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0194 - acc: 0.9936 - val_loss: 0.0217 - val_acc: 0.9935

Epoch 00017: val_acc did not improve from 0.99420
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004223568.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0180 - acc: 0.9942 - val_loss: 0.0199 - val_acc: 0.9941

Epoch 00018: val_acc did not improve from 0.99420
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.000402037.
60000/60000 [==============================] - 6s 92us/step - loss: 0.0177 - acc: 0.9942 - val_loss: 0.0198 - val_acc: 0.9943

Epoch 00019: val_acc improved from 0.99420 to 0.99430, saving model to /content/best_weights.hdf5
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.0003835827.
60000/60000 [==============================] - 5s 92us/step - loss: 0.0175 - acc: 0.9941 - val_loss: 0.0211 - val_acc: 0.9938

Epoch 00020: val_acc did not improve from 0.99430
```

## Evaluation Result
```
from keras.models import load_model

# Load Saved Model
best_model = load_model('/content/best_weights.hdf5')

score = best_model.evaluate(X_test, Y_test, verbose=0)
print(score)

[0.01984984504247259, 0.9943]
```
