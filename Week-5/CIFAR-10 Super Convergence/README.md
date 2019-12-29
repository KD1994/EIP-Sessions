## CIFAR-10 Super Convergence

## Train Accuracy ~ 95.93%

## Test Accuracy ~ 93.57%

## Parameters

```
BATCH_SIZE = 512
MOMENTUM = 0.85 to 0.95
LEARNING_RATE = 0.4
WEIGHT_DECAY =  0.000125
EPOCHS = 24
```

## Model Logs

```
Epoch: 1 LR: [0.08000000000000002]
Epoch: 1/24 	 Time: 17.30s 	 Training Loss: 1.3928 	 Training Accu: 0.4924 	 Val Loss: 1.0925 	 Val Accu: 0.6142
Epoch: 2 LR: [0.0950326557343032]
Epoch: 2/24 	 Time: 17.39s 	 Training Loss: 0.8961 	 Training Accu: 0.6804 	 Val Loss: 0.7232 	 Val Accu: 0.7455
Epoch: 3 LR: [0.13730586370688685]
Epoch: 3/24 	 Time: 17.80s 	 Training Loss: 0.7275 	 Training Accu: 0.7445 	 Val Loss: 0.6681 	 Val Accu: 0.7663
Epoch: 4 LR: [0.19887614163979533]
Epoch: 4/24 	 Time: 17.49s 	 Training Loss: 0.6529 	 Training Accu: 0.7716 	 Val Loss: 0.8195 	 Val Accu: 0.7285
Epoch: 5 LR: [0.26817392963764386]
Epoch: 5/24 	 Time: 17.55s 	 Training Loss: 0.5927 	 Training Accu: 0.7913 	 Val Loss: 0.5780 	 Val Accu: 0.8009
Epoch: 6 LR: [0.3321776053239276]
Epoch: 6/24 	 Time: 18.55s 	 Training Loss: 0.5332 	 Training Accu: 0.8136 	 Val Loss: 0.5245 	 Val Accu: 0.8177
Epoch: 7 LR: [0.3788603534196228]
Epoch: 7/24 	 Time: 19.75s 	 Training Loss: 0.4884 	 Training Accu: 0.8297 	 Val Loss: 0.5626 	 Val Accu: 0.8137
Epoch: 8 LR: [0.39945010291405725]
Epoch: 8/24 	 Time: 19.03s 	 Training Loss: 0.4476 	 Training Accu: 0.8430 	 Val Loss: 0.4507 	 Val Accu: 0.8450
Epoch: 9 LR: [0.39770897179845427]
Epoch: 9/24 	 Time: 18.89s 	 Training Loss: 0.4095 	 Training Accu: 0.8562 	 Val Loss: 0.4942 	 Val Accu: 0.8320
Epoch: 10 LR: [0.3886505044782719]
Epoch: 10/24 	 Time: 17.58s 	 Training Loss: 0.3816 	 Training Accu: 0.8663 	 Val Loss: 0.4016 	 Val Accu: 0.8632
Epoch: 11 LR: [0.37301448938755233]
Epoch: 11/24 	 Time: 17.62s 	 Training Loss: 0.3566 	 Training Accu: 0.8759 	 Val Loss: 0.4491 	 Val Accu: 0.8568
Epoch: 12 LR: [0.3513461083015133]
Epoch: 12/24 	 Time: 17.58s 	 Training Loss: 0.3362 	 Training Accu: 0.8819 	 Val Loss: 0.4057 	 Val Accu: 0.8633
Epoch: 13 LR: [0.32440087382346416]
Epoch: 13/24 	 Time: 17.57s 	 Training Loss: 0.3196 	 Training Accu: 0.8883 	 Val Loss: 0.5277 	 Val Accu: 0.8323
Epoch: 14 LR: [0.2931182868864218]
Epoch: 14/24 	 Time: 17.68s 	 Training Loss: 0.2942 	 Training Accu: 0.8978 	 Val Loss: 0.3516 	 Val Accu: 0.8800
Epoch: 15 LR: [0.258589079123784]
Epoch: 15/24 	 Time: 17.55s 	 Training Loss: 0.2762 	 Training Accu: 0.9041 	 Val Loss: 0.3438 	 Val Accu: 0.8828
Epoch: 16 LR: [0.2220171822710773]
Epoch: 16/24 	 Time: 17.58s 	 Training Loss: 0.2619 	 Training Accu: 0.9089 	 Val Loss: 0.2921 	 Val Accu: 0.8989
Epoch: 17 LR: [0.18467775061368402]
Epoch: 17/24 	 Time: 17.57s 	 Training Loss: 0.2401 	 Training Accu: 0.9170 	 Val Loss: 0.3093 	 Val Accu: 0.8963
Epoch: 18 LR: [0.14787270011411002]
Epoch: 18/24 	 Time: 17.71s 	 Training Loss: 0.2227 	 Training Accu: 0.9233 	 Val Loss: 0.2928 	 Val Accu: 0.9015
Epoch: 19 LR: [0.11288531443842416]
Epoch: 19/24 	 Time: 17.57s 	 Training Loss: 0.2043 	 Training Accu: 0.9311 	 Val Loss: 0.2642 	 Val Accu: 0.9119
Epoch: 20 LR: [0.0809355006359754]
Epoch: 20/24 	 Time: 17.57s 	 Training Loss: 0.1829 	 Training Accu: 0.9376 	 Val Loss: 0.2471 	 Val Accu: 0.9166
Epoch: 21 LR: [0.05313725457499649]
Epoch: 21/24 	 Time: 17.57s 	 Training Loss: 0.1642 	 Training Accu: 0.9446 	 Val Loss: 0.2215 	 Val Accu: 0.9259
Epoch: 22 LR: [0.03045981918903087]
Epoch: 22/24 	 Time: 17.56s 	 Training Loss: 0.1455 	 Training Accu: 0.9512 	 Val Loss: 0.2112 	 Val Accu: 0.9297
Epoch: 23 LR: [0.013693889831681681]
Epoch: 23/24 	 Time: 17.67s 	 Training Loss: 0.1338 	 Training Accu: 0.9554 	 Val Loss: 0.1990 	 Val Accu: 0.9353
Epoch: 24 LR: [0.003424045059389316]
Epoch: 24/24 	 Time: 18.23s 	 Training Loss: 0.1244 	 Training Accu: 0.9593 	 Val Loss: 0.1980 	 Val Accu: 0.9357
```
