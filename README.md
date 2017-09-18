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
