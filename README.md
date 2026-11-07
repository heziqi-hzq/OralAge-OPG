# OralAge-OPG

- This repository provides training code for the OralAge-OPG model introduced in the manuscript **OralAge: A Novel Approach for Oral Biological Age Estimation and Its Clinical Associations**.

## Brief Introduction

OralAge-OPG predicts age from panoramic radiographs using an ImageNet-pretrained Inception-ResNet-v2 backbone and a composite loss combining cross-entropy, mean, and variance terms. The model was developed using radiographs from 506 relatively healthy adults and evaluated by 5-fold cross-validation. Independent testing used radiographs from 163 participants at another campus, where images were acquired with different equipment. The mean absolute error was 5.22 years in internal cross-validation and 5.82 years in independent testing. The study then examined associations of the age-bias-corrected OralAge Index with disease status and 3-year implant outcomes.

## Data Availability Statement

Original data are partly available upon reasonable request for research collaboration.
