# DKF-PredRNN++: Dynamic Physics-Guided Learning for SST Forecasting via Online State Correction

## What is this repository for?
This repository contains the official PyTorch implementation of **DKF-PredRNN++**, as presented in the paper *"Dynamic Physics-Guided Learning for SST Forecasting via Online State Correction"*. 

The code realizes a Dynamic Physics-Guided Learning (DPGL) paradigm by embedding a Differentiable Kalman Filter (DKF) directly into a spatiotemporal recurrent network. It dynamically corrects internal states during sea surface temperature (SST) sequence forecasting.

## Dependencies
To configure the environment, you will need **Python 3.9**. Install the required dependencies via pip:
```bash
pip install -r requirements.txt
