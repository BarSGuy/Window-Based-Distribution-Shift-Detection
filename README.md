# 📊 Coverage Based Detection: Advanced Distribution Shift Detection

Welcome to the official repository for our research paper: **"Window-Based Distribution Shift Detection for Deep Neural Networks"** 📜 **NeurIPS 2023**.

## 📝 **TL;DR**:
Our methodology is intricately designed to pinpoint deviations in the input distribution that could potentially compromise the accuracy of a deep neural network's predictions. Grounded in the principles of selective prediction, our approach continuously monitors the network's behavior across a designated test window, issuing alerts upon anomaly detection. Notably, it outperforms contemporary techniques in the ImageNet dataset, delivering superior efficiency in both temporal and spatial realms.

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
Before diving in, ensure the following dependencies are installed:
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
PATH_TO_SAVE_ACCURACIES: <path to save model accuracy on detected shifts>
```


### 2. **Execution**
Run the script with your chosen model parameters:

```bash
chmod +x all_methods.sh
./all_methods.sh mobilenetv3_small_075 resnet50 vit_tiny_patch16_224
```
The above command will:
   * Infer the ImageNet dataset (test set) and the Out-Of-Distribution (OOD) datasets (as detailed in our paper).
   * Evaluate the shift detection performance of all detectors over these OOD datasets.
   * Consolidate and save results, specifically:
     - Logits/embs of all datasets (both ID and OOD) required by the detectors.
     - Accuracy of the models over all datasets (both ID and OOD).
     - All performance metrics used for evaluating the detectors (refer to the paper for the metrics used).


> **Note**: Our framework seamlessly integrates with any model from the [`timm` library](https://github.com/huggingface/pytorch-image-models), making them valid arguments for the `all_methods.sh` script.

## Distribution Shift Playground Notebook

Navigate to our interactive Jupyter notebook located at:
[Distribution Shift Playground](./playground/Distribution%20shift%20playground.ipynb)

This notebook offers a detailed walkthrough on how to utilize our distribution shift detector. It also includes illustrative examples to enhance understanding.
