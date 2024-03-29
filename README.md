# AdUni
Training
====
For training the model, you may modify the `run.sh` file. Define `--data_folder` as the path to your dataset. We also provide our splits in the third section. To run, use `bash run.sh`

Testing
====
For testing the model, you may modify the `run.sh` file. Set `--resume` to True and define `--ckpt` as the path to your save model. To run, use `bash run.sh`

Trained Packages
====
We provide trained models with AdUni and different sampling technics with one train-test [splits](https://drive.google.com/drive/folders/14eIMNuP2hqJKkJWF_McMKhsbzQpLmEj0?usp=sharing).

| method | Dataset | Test acc. | Test f1 | url |
|-------------------|-------------------|---------------------|--------------------|--------------------|
| AdUni | ISIC2018 | 87.1% | 79.0% | [model](https://drive.google.com/file/d/1XN-jyzkBCiYMGUYNHMj3hwusx6ROwh_G/view?usp=sharing) |
| AdUni+upsample | ISIC2018 | 88.2% | 79.8% | [model](https://drive.google.com/file/d/1BjjxmuvIn23ZuLye52U0V3Xf3Q5rAbYX/view?usp=sharing) | 
| AdUni | APTOS2019 | 83.3%  | 69.4%| [model](https://drive.google.com/file/d/13-mWo2_VHvU8CE5ObCQm7Y76m9VXLvus/view?usp=sharing) |
| AdUni+upsample | APTOS2019 | 83.9% | 71.1% | [model](https://drive.google.com/file/d/1ceGo_NYSqmp4jdTWg-GWjnDNpTxnvdp7/view?usp=sharing) | 

Reference
====
@inproceedings{cong2022aduni,
  title={Adaptive Unified Contrastive Learning for Imbalanced Classification},
  author={Cong Cong, Yixing Yang, Sidong Liu, Maurice Pagnucco, Antonio DiIeva, Shlomo Berkovsky, Yang Song},
  booktitle={International Workshop on Machine Learning in Medical Imaging},
  year={2022},
  organization={Springer}
}
