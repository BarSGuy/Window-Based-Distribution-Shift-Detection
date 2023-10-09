# Coverage Based Detection - a Method for Detecting a Distribution Shift

Official implementation of our paper: Window-Based Distribution Shift Detection for Deep Neural Networks.

**TLDR**:
The method aims to detect deviations in the input distribution that could potentially harm the accuracy of the network's
predictions. The proposed method is based on selective prediction principles and involves continuously monitoring the
network's operation over a test window and firing off an alarm when a deviation is detected. The method outperforms the
state-of-the-art approaches for this task on the ImageNet dataset, while being more efficient in time and space
complexities.

# Install
To install the required packages, run the following command:

    pip install -r requirements.txt

# reproduce Results
To reproduce the results, follow these steps:

1) Open the config.yml file and set the following parameters:


    PATH_TO_IMAGENET:<path to ImageNet dataset>
    PATH_TO_IMAGENET_O:<path to ImageNet-O dataset>
    PATH_TO_IMAGENET_A:<path to ImageNet-A dataset>
    PATH_TO_RESULTS:<path to save the results>
    PATH_TO_SAVE_OUTPUTS:<path to save logits/embs>
    PATH_TO_SAVE_ACCURACIES:<path to save accuracy of the models on shifts>

2) Run the following command:
    
    ./all_methods.sh mobilenetv3_small_075 resnet50 vit_tiny_patch16_224

Note: You can use any model from the timm library as an argument to the **all_methods.sh** script.
