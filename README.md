# 📊 Coverage Based Detection: Advanced Distribution Shift Detection

Welcome to the official repository for our research paper: **"Window-Based Distribution Shift Detection for Deep Neural Networks"** 📜 **NeurIPS 2023**.

## 📝 **TL;DR**:
This repository contains the official code for replicating our [experiments](#-replicating-results) detailed in the paper. Additionally, we provide a practical guide, accessible through a [Python notebook](#-explore-with-the-distribution-shift-detector-notebook), to better understand our method. Also, Within the notebook, we illustrate how to employ this distribution shift detector with your own dataset.
<!-- Our methodology is intricately designed to pinpoint deviations in the input distribution that could potentially compromise the accuracy of a deep neural network's predictions. Grounded in the principles of selective prediction, our approach continuously monitors the network's behavior across a designated test window, issuing alerts upon anomaly detection. Notably, it outperforms contemporary techniques in the ImageNet dataset, delivering superior efficiency in both temporal and spatial realms. -->

<table>
<tr>
<td><img src="figures\val_window_1000_temp_1_c_num_10_delta_0.01_uc_mech_Ent.png" alt="Val Window Image" style="width: 100%;"/></td>
<td><img src="figures\window_1000_temp_1_c_num_10_delta_0.01_uc_mech_Ent.png" alt="Window Image" style="width: 100%;"/></td>
</tr>
<tr>
<td style="text-align:center">No shift case</td>
<td style="text-align:center">Shift case</td>
</tr>
</table>


## 🚀 Getting Started

### 🔧 Prerequisites
Install the requirements:
```bash
pip install -r requirements.txt
```
#### 📦 Dataset Download Guide

##### ImageNet

1. Register at [ImageNet website](http://www.image-net.org/signup).
2. Download 'ILSVRC2012' from the 'Downloads' section.

##### ImageNet-O and ImageNet-A

1. Visit this [GitHub repository](https://github.com/hendrycks/natural-adv-examples).
2. Follow the provided download instructions.


## 🔄 Replicating Results

### 1. **Configuration**
Adapt the `config.yml` file with the relevant parameters:

```yml
PATH_TO_IMAGENET: <path to ImageNet dataset>
PATH_TO_IMAGENET_O: <path to ImageNet-O dataset>
PATH_TO_IMAGENET_A: <path to ImageNet-A dataset>
PATH_TO_RESULTS: <path to store results>
PATH_TO_SAVE_OUTPUTS: <path to save logits/embs>
PATH_TO_SAVE_ACCURACIES: <path to save model accuracy on the shifts used in the paper>
```


### 2. **Execution**
Run the script with your chosen model parameters:

```bash
chmod +x all_methods.sh
./all_methods.sh mobilenetv3_small_075 resnet50 vit_tiny_patch16_224
```
The given command executes the following:

- **Inference**:
  - Processes the ImageNet dataset (test set).
  - Processes the Out-Of-Distribution (OOD) datasets (detailed in our paper).

- **Evaluation**:
  - Assesses the shift detection capabilities of all detectors on the OOD datasets.

- **Result Saving**:
  - **Logits/Embeddings**: Saves the logits/embs of all models for all datasets (both In-Distribution and OOD). 
    - 📍 Path: `<PATH_TO_SAVE_OUTPUTS>`
  - **Model Accuracy**: Records the accuracy of each model across all datasets.
    - 📍 Path: `<PATH_TO_SAVE_ACCURACIES>`
  - **Performance Metrics**: Retains all metrics (as mentioned in our paper) used for detector evaluations. Notably:
    - In the specified `<PATH_TO_RESULTS>`, a new 'imagenet' directory is created.
    - Within 'imagenet', nested folders follow the naming convention: `imagenet_vs_<OOD_DATASET>_<MODEL>`. Each folder contains shift detection performances for a specific OOD dataset.
    - The 'imagenet' directory also holds an average performance summary for each model across all OOD datasets.

- **Visualizations**:
  - Generates and saves the detection times figure from our paper, which depicts the run-time in seconds against the detection-training set size.
    - 📍 Path: `<PATH_TO_RESULTS>`



> **Note**: Our framework seamlessly integrates with any model from the [`timm` library](https://github.com/huggingface/pytorch-image-models), making them valid arguments for the `all_methods.sh` script.

## 📓 Explore with the Distribution Shift Detector Notebook

Dive into our hands-on Jupyter notebook here:
[Distribution Shift Detector Notebook](./playground/Distribution%20shift%20playground.ipynb)

In this interactive guide:
- **Walkthrough**: Understand the step-by-step process to leverage our distribution shift detector.
- **Illustrative Examples**: Witness the power of our methodology through visual and practical demonstrations.
- **Your Own Dataset**: Learn how to apply our distribution shift detection on your custom datasets, making it adaptable to your unique needs.



<!-- ## Distribution Shift Playground Notebook

Navigate to our interactive Jupyter notebook located at:
[Distribution Shift Playground](./playground/Distribution%20shift%20playground.ipynb)

This notebook offers a detailed walkthrough on how to utilize our distribution shift detector. It also includes illustrative examples to enhance understanding. -->
