# Reproducibility Notes

## Artifact Policy

This portfolio branch tracks source code, the report PDF, generated figure assets, and curated final metrics. It intentionally excludes:

- raw datasets
- pretrained weights
- model checkpoints
- raw Excel logs
- Apptainer/Singularity images
- cluster stdout/stderr logs

Those files are large, environment-specific, or externally licensed. The public repo should remain inspectable and cloneable.

## Dataset

The paper experiments use the Kaggle Brain Tumor MRI dataset:

<https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

Expected directory shape:

```text
Kaggle Brain MRI/
  glioma/
  meningioma/
  notumor/
  pituitary/
```

Set `CALIBRACT_DATA_DIR` or pass `--dataset-path` to `scripts/run_benchmark.py`.

## Pretrained Weights

By default, CalibraCT downloads ResNet-50 weights with PyTorch. For offline cluster runs, put `resnet50-11ad3fa6.pth` in a directory and set `CALIBRACT_PRETRAINED_DIR` to that path.

## Compute Canada / Apptainer

The original work used an Apptainer container and SLURM GPU jobs. A typical run binds the project, dataset, and pretrained weight directories into the container, then launches:

```bash
python /mnt/scripts/run_benchmark.py \
  --config /mnt/configs/kaggle_brain_mri.yaml \
  --dataset-path "/mnt/datasets/Kaggle Brain MRI" \
  --output-dir /mnt/outputs
```

Keep container images outside Git. If a container recipe is needed later, add a small definition file rather than the built `.sif`.
