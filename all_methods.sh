#!/usr/bin/env bash

## ./all_methods.sh vit_tiny_patch16_224 xcit_tiny_12_p8_224 convnext_tiny deit_tiny_distilled_patch16_224 gcvit_tiny swin_s3_tiny_224 tinynet_d convnext_nano_ols maxvit_nano_rw_256 xcit_nano_12_p8_224

# the run command for the paper: ./all_methods.sh mobilenetv3_small_075 resnet50 vit_tiny_patch16_224

# inference and detect all OOD datasets for all models

echo "Inferencing and detecting"

for var in "$@"
do
  echo "$var"
  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_o' \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_a' \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_gaussian_noise' \
  --IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD=0.1 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_gaussian_noise' \
  --IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD=0.3 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_gaussian_noise' \
  --IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD=0.5 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_gaussian_noise' \
  --IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD=1 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_zoom' \
  --IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR=0.9 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_zoom' \
  --IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR=0.7 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_zoom' \
  --IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR=0.5 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_zoom' \
  --IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR=0.3 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_rotate' \
  --IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE=5 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_rotate' \
  --IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE=10 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_rotate' \
  --IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE=20 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='imagenet_rotate' \
  --IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE=25 \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='fgsm' \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --IMAGENET.DEVICE_INDEX=0 \
  --IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS=0.0001 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='fgsm' \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --IMAGENET.DEVICE_INDEX=0 \
  --IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS=0.0003 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='fgsm' \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --IMAGENET.DEVICE_INDEX=0 \
  --IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS=0.0005 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='fgsm' \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --IMAGENET.DEVICE_INDEX=0 \
  --IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS=0.00007 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"

  python main.py --IMAGENET.EXPERIMENT.OOD='pgd' \
  --IMAGENET.EXPERIMENT.DETECTORS 'my_detector' 'drift_ks_embs' 'drift_ks_softmaxes' 'drift_mmd_embs' 'drift_mmd_softmaxes' 'drift_single_softmaxes' 'drift_single_entropies' \
  --IMAGENET.NUM_WORKERS=4 \
  --IMAGENET.DEVICE_INDEX=0 \
  --MY_DETECTOR.C_NUM=10 \
  --MY_DETECTOR.DELTA=0.01 \
  --MY_DETECTOR.UC_MECH='Ent' \
  --MY_DETECTOR.SIGNIFICANCE_LEVEL=0.05 \
  --MY_DETECTOR.TEMP=1 \
  --IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS=15 \
  --IMAGENET.MODEL="$var"
done

# Averaging Results
echo "Averaging Results"
for var in "$@"
do
  python average_methods.py --IMAGENET.MODEL="$var"
done

# Plotting detection times
echo "Plotting detection times"
python plot_detection_times.py