# CauSSL: Causality-inspired Semi-supervised Learning for Medical Image Segmentation

### Introduction

We provide the codes for CPSCauSSL and MCCauSSL with the 3D V-Net architecture targeted for the Pancreas-CT Dataset.

### Requirements

1. Pytorch
2. TensorBoardX
3. Python == 3.6
4. Some basic python packages such as Numpy

### Usage
   
1. Train the model:
   python train_CT_CPSCauSSL.py
   python train_CT_MCCauSSL.py

2. Test the model:
   For the CPSCauSSL method, the testing has been included in "train_CT_CPSCauSSL.py".
   For the MCCauSSL method: python test_CT_norm_mct.py

### Acknowledgement
This code is based on the framework of UA-MT. We thank the authors for their codebase.

