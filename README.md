# SGL1,2
Group regularization method has long been used for structured pruning of convolutional neural networks. Motivated by Group Lasso and Group L1/2, we propose a group L1,2 regularization method, which possesses strong penalty ability in early learning stage. Moreover, we propose a smooth group L1,2
regularization `SGL1,2` by replacing the non-smooth absolute value function with a smooth function, which can eliminate oscillation and improve accuracy. 

## Regularization Training
### 1. example:mnist + lenet
```bash
python main.py --dataset=mnist --network=lenet --penalty=3
```
### 2. the reg_param for `SGL1,2`

|Network  |SGL1/2
| ------   |:---: |
|LeNet    |\  
|ResNet20   |7.e-06
|VGG16    |3.e-08 
|AlexNet    |4.e-08 
|ResNet50  |5.e-07  
