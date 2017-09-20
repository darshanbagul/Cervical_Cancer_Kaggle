# Cervical Cancer Screening - Kaggle Challenge

## Introduction

Recently, Intel partnered with MobileODT to challenge Kagglers to develop an algorithm which accurately identifies a woman’s cervix type based on images. Their motivation: doing so will prevent ineffectual treatments and allow healthcare providers to give proper referral for cases that require more advanced treatment.

The following notebook is my solution for the presented [task](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening).

In this competition, we had to develop algorithms to correctly classify cervix types based on cervical images. These different types of cervix in our data set are all considered normal (not cancerous), but since the transformation zones aren't always visible, some of the patients require further testing while some don't. This decision is very important for the healthcare provider and critical for the patient. Identifying the transformation zones is not an easy task for the healthcare providers, therefore, an algorithm-aided decision will significantly improve the quality and efficiency of cervical cancer screening for these patients.

## Dataset
The training dataset comprises of 1481 images belonging to 3 different categories, with the following distribution:
  1. Type 1 - 252 images
  
  2. Type 2 - 780 images
  
  3. Type 3 - 449 images

The competition was held in two stages where we were provided 2 test datasets for reporting our results. After stage 1, the output classes of stage 1 test images were released, so as to give kagglers a chance to improve and fine tune their models. The number of images provided for testing ast 2 stages are:

  1. Stage 1 Test: 512 images

  2. Stage 2 Test (Final): 4018 images

The final loss and accuracy were to be reported by tagging 4018 images.

The dataset is available on Kaggle [here](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data)

## Preprocessing Data

Data preprocessing comprises of the following steps:
  
  1. Resizing all images to same size (32 x 32 x 3)
  
  2. Normalizing pixel values
  
  3. Applying image deformations (Random Scaling + Rotations) for regularization
  
  4. Storing data in a loadable numpy format

## Model

Using a CNN was a default choice given we have to build an image classifier.

We shall be using:
  
  1. Two 2D-Convolutional layers followed by Max Pooling layers
  
  2. ReLU activations

  3. Dropout between output of second convolutional block and input of fully connected layer

  4. Two fully connected layers for classification with dropout

  5. Hyperbolic Tan activation for FC-1 layer
  
  6. Softmax activation for FC-2 layer (Obvious choice, given a multiclass classification problem)
  
  7. Adamax optimizer - a variant of Adam based on the infinity norm
  
### Model architecture:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 4, 30, 30)         112       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 15, 15)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 13, 13)         296       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 8, 6, 6)           0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 8, 6, 6)           0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 288)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 12)                3468      
_________________________________________________________________
dropout_4 (Dropout)          (None, 12)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 3)                 39        
=================================================================
Total params: 3,915
Trainable params: 3,915
Non-trainable params: 0
_________________________________________________________________
```

## Results

The simple convolutional model implemented in this notebook was able to generate a score of **0.96407.**

This helped me achieve a rank of **#110 on Kaggle leaderboard.**

## Future considerations:

1. I believe a higher score can be achieved by Transfer Learning. Fine tuning a pretrained model such as Inception-V3, VGG19, ResNet-50 can definitely boost the model accuracy.

2. Many kagglers reported improved results by using R-CNN like approach i.e generating bounding boxes around regions of interest and generating probability predictions.

I would definitely consider exploring these ideas in future implementations!
