# Data
## Medical
### NCT-CRC-HE-100K
https://zenodo.org/records/1214456
```bash
# make directory
mkdir /local00/bioinf/NCT-CRC-HE-100K
cd /local00/bioinf/NCT-CRC-HE-100K
# download data
wget https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip -O train.zip
wget https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip -O val.zip
# unzip
unzip train.zip
unzip val.zip
rm train.zip
rm val.zip
# rename to match image datasets
mv NCT-CRC-HE-100K train
mv CRC-VAL-HE-7K val
```

### https://paperswithcode.com/sota/image-classification-on-chaoyang

## iNat18
### create lowshot dataset
python analysis/data/create_inat18_lowshot_dataset.py --src /local00/bioinf/inat18 --dst /local00/bioinf/inat18_1shot_split1 --shots 1 --seed 1
python analysis/data/create_inat18_lowshot_dataset.py --src /local00/bioinf/inat18 --dst /local00/bioinf/inat18_1shot_split2 --shots 1 --seed 2
python analysis/data/create_inat18_lowshot_dataset.py --src /local00/bioinf/inat18 --dst /local00/bioinf/inat18_1shot_split3 --shots 1 --seed 3
python analysis/data/create_inat18_lowshot_dataset.py --src /local00/bioinf/inat18 --dst /local00/bioinf/inat18_5shot_split1 --shots 5 --seed 1
python analysis/data/create_inat18_lowshot_dataset.py --src /local00/bioinf/inat18 --dst /local00/bioinf/inat18_5shot_split2 --shots 5 --seed 2
python analysis/data/create_inat18_lowshot_dataset.py --src /local00/bioinf/inat18 --dst /local00/bioinf/inat18_5shot_split3 --shots 5 --seed 3
python analysis/data/create_inat18_lowshot_dataset.py --src /local00/bioinf/inat18 --dst /local00/bioinf/inat18_10shot_split1 --shots 10 --seed 1
python analysis/data/create_inat18_lowshot_dataset.py --src /local00/bioinf/inat18 --dst /local00/bioinf/inat18_10shot_split2 --shots 10 --seed 2
python analysis/data/create_inat18_lowshot_dataset.py --src /local00/bioinf/inat18 --dst /local00/bioinf/inat18_10shot_split3 --shots 10 --seed 3
### per-class zips for fast copying to compute nodes
python main_create_zips.py --src /local00/bioinf/inat18/train --dst /local00/bioinf/inat18/train_zip --zips --image_folder
python main_create_zips.py --src /local00/bioinf/inat18/val --dst /local00/bioinf/inat18/val_zip --zips --image_folder
python main_create_zips.py --src /local00/bioinf/inat18_1shot_split1/train --dst /local00/bioinf/inat18_1shot_split1/train_zip --zips --image_folder
python main_create_zips.py --src /local00/bioinf/inat18_1shot_split2/train --dst /local00/bioinf/inat18_1shot_split2/train_zip --zips --image_folder
python main_create_zips.py --src /local00/bioinf/inat18_1shot_split3/train --dst /local00/bioinf/inat18_1shot_split3/train_zip --zips --image_folder
python main_create_zips.py --src /local00/bioinf/inat18_5shot_split1/train --dst /local00/bioinf/inat18_5shot_split1/train_zip --zips --image_folder
python main_create_zips.py --src /local00/bioinf/inat18_5shot_split2/train --dst /local00/bioinf/inat18_5shot_split2/train_zip --zips --image_folder
python main_create_zips.py --src /local00/bioinf/inat18_5shot_split3/train --dst /local00/bioinf/inat18_5shot_split3/train_zip --zips --image_folder
python main_create_zips.py --src /local00/bioinf/inat18_10shot_split1/train --dst /local00/bioinf/inat18_10shot_split1/train_zip --zips --image_folder
python main_create_zips.py --src /local00/bioinf/inat18_10shot_split2/train --dst /local00/bioinf/inat18_10shot_split2/train_zip --zips --image_folder
python main_create_zips.py --src /local00/bioinf/inat18_10shot_split3/train --dst /local00/bioinf/inat18_10shot_split3/train_zip --zips --image_folder


# caltech101
wget https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip
unzip caltech-101.zip
rm caltech-101.zip
rm -rf __MACOSX/
cd caltech-101
tar -xvzf 101_ObjectCategories.tar.gz
rm 101_ObjectCategories.tar.gz
rm caltech-101/101_ObjectCategories/BACKGROUND_Google/tmp
rm Annotations.tar
rm show_annotation.m

# oxford-pets
mkdir oxford-pets
cd oxford-pets
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xvzf images.tar.gz
rm images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvzf annotations.tar.gz
rm annotations.tar.gz
rm -rf annotations/trimaps
rm -rf annotations/xmls

# oxford-flowers
mkdir oxford-flowers
cd oxford-flowers
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xvzf 102flowers.tgz
rm 102flowers.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

# sun397
mkdir sun-397
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
tar -xvzf SUN397.tar.gz
rm SUN397.tar.gz

# DTD
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvzf dtd-r1.0.1.tar.gz
rm dtd-r1.0.1.tar.gz

# SVHN
mkdir svhn
python
from torchvision.datasets import SVHN
SVHN(root=".", split="train", download=True)
SVHN(root=".", split="test", download=True)
# Models

```bash
# MAE
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -O mae_base16.pth
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth -O mae_large16.pth
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth -O mae_huge14.pth
wget https://dl.fbaipublicfiles.com/maws/pretrain/mae_in1k/vit_2b14.pt -O mae_twob14.pt
# MAE IG3B
wget https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_l16.pt -O mae_ig3b_large16.pt
wget https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_h14.pt -O mae_ig3b_huge14.pt
wget https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_2b14.pt -O mae_ig3b_twob14.pt
# MAE-WS IG3B
wget https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_l16.pt -O maews_ig3b_large16.pt
wget https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_h14.pt -O maews_ig3b_huge14.pt
wget https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_2b14.pt -O maews_ig3b_twob14.pt
# MAE finetuned
wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth -O mae_base16_finetuned.pth
wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth -O mae_large16_finetuned.pth
wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth -O mae_huge14_finetuned.pth
# MAE from MAE-CT
wget https://ml.jku.at/research/maect/download/mae_reimpl_base16.th -O maereimpl_base16.th
wget https://ml.jku.at/research/maect/download/mae_reimpl_large16.th -O maereimpl_large16.th
wget https://ml.jku.at/research/maect/download/mae_reimpl_huge16.th -O maereimpl_huge16.th
# long sequence MAE
wget https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitb_dec384d12h8b_800ep_img448_crop0.2-1.0_maskds2.pth -O mae_base16res448e800.pth
wget https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitl_dec512d16h8b_800ep_img448_crop0.2-1.0_maskds2.pth -O mae_large16res448e800.pth
wget https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitb_dec384d12h8b_1600ep_img448_crop0.2-1.0_maskds2.pth -O mae_base16res448.pth
wget https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitl_dec512d16h8b_1600ep_img448_crop0.2-1.0_maskds2.pth -O mae_large16res448.pth

# Data2Vec 2.0
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_imagenet.pt -O data2vec2_base16.pt
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet.pt -O data2vec2_large16.pt
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/huge_imagenet.pt -O data2vec2_huge14.pt
# Data2Vec 2.0 finetuned
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet_ft.pt -O data2vec2_large16_finetuned.pt
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/huge_imagenet_ft.pt -O data2vec2_huge14_finetuned.pt

# MAE-CT
wget https://ml.jku.at/research/maect/download/maect_base16.th -O maect_base16.th
wget https://ml.jku.at/research/maect/download/maect_large16.th -O maect_large16.th
wget https://ml.jku.at/research/maect/download/maect_huge16.th -O maect_huge16.th
wget https://ml.jku.at/research/maect/download/maect_huge14.th -O maect_huge14.th
# MAE-CT-aug
wget https://ml.jku.at/research/maect/download/maectaug_base16.th -O maectaug_base16.th
wget https://ml.jku.at/research/maect/download/maectaug_large16.th -O maectaug_large16.th
wget https://ml.jku.at/research/maect/download/maectaug_huge16.th -O maectaug_huge16.th
wget https://ml.jku.at/research/maect/download/maectaug_huge14.th -O maectaug_huge14.th

# LayerGrafting
wget https://www.dropbox.com/sh/e9czo0xtivdqvff/AAC730kZx8Bj6pEIhFswpvVla/checkpoint_final.pth.tar -O layergrafting_base16.pth.tar
wget https://www.dropbox.com/sh/fk92wphgu8fq772/AABklx8vjQmDZgz8vg6BbTPWa/checkpoint_final.pth.tar -O layergrafting_large16.pth.tar
# MUGS
wget https://huggingface.co/zhoupans/Mugs/resolve/main/pretrained%20models/vit_small_800ep/vit_small_backbone_800ep.pth -O mugs_small16.pth
wget https://huggingface.co/zhoupans/Mugs/resolve/main/pretrained%20models/vit_base_400ep/vit_base_backbone_400ep.pth -O mugs_base16.pth
wget https://huggingface.co/zhoupans/Mugs/resolve/main/pretrained%20models/vit_large_backbone_250ep.pth -O mugs_large16.pth
# I-JEPA
wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar -O ijepa_huge14.pth.tar
wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar -O ijepa_huge16res448.pth.tar
# IBOT
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth -O ibot_base16.pth
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_rand_mask/checkpoint_teacher.pth -O ibot_base16.pth
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_rand_mask/checkpoint.pth -O ibot_large16_rand.pth
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_pt22k/checkpoint.pth -O ibot_large16_in22k.pth
# DINO
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth -O dino_base8.pth
# CrossMAE
wget https://huggingface.co/longlian/CrossMAE/resolve/main/vits-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vits-pretrain-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth -O crossmae_small16.pth
wget https://huggingface.co/longlian/CrossMAE/resolve/main/vitb-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vitb-pretrain-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth -O crossmae_base16.pth
wget https://huggingface.co/longlian/CrossMAE/resolve/main/vitl-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vitl-pretrain-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth -O crossmae_large16.pth 
# dBOT
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/mmodal/dbot/84.5_dbot_base_pre.pth -O dbot_base16.pth
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/mmodal/dbot/86.6_dbot_large_pre.pth -O dbot_large16.pth
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/mmodal/dbot/87.4_dbot_huge_pre.pth -O dbot_huge14.pth
# data2vec 
wget https://dl.fbaipublicfiles.com/fairseq/data2vec/data2vec_vision/large_1600/checkpoint-799.pth -O data2vec_large16.pt
# DINOv2
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth -O dinov2_large14.pth

# AIM
wget https://huggingface.co/apple/AIM/resolve/main/aim_600m_2bimgs_attnprobe_backbone.pth -O aim_600m.pth

# CMAE
wget https://cmae.s3.us-west-1.amazonaws.com/weight/cmae_vit-base-p16_32xb128-coslr-1600e_in1k.pth -O cmae_base16.pth

# VideoMAE
gdown.download(id="1nU-H1u3eJ-VuyCveU7v-WIOcAVxs5Hww", output="videomae_small16.pth")
gdown.download(id="1tEhLyskjb755TJ65ptsrafUG2llSwQE1", output="videomae_base16.pth")
gdown.download(id="1qLOXWb_MGEvaI7tvuAe94CV7S2HXRwT3", output="videomae_large16.pth")
gdown.download(id="1AJQR1Rsi2N1pDn9tLyJ8DQrUREiBA1bO", output="videomae_huge16.pth")
```


# segmentation

using dbot implementation
- clone dbot
- link ade20k dataset (see mmseg setup instructions)

```
conda create --name maeseg python=3.8
conda activate maeseg
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
# if cpu: torchvision==0.8.2 is not available --> download from https://github.com/pytorch/vision/tags?after=v0.10.0-rc1
# then install with pip install --no-cache-dir -v .
pip install torchvision==0.8.2
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 22.04-dev
# potentially have to comment out version check of cuda https://github.com/NVIDIA/apex/pull/323#discussion_r287021798
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

git clone -b v0.11.0 https://github.com/open-mmlab/mmsegmentation
cd mmsegmentation
pip install --no-cache-dir -v -e .
cd ..

# for windows install "Microsoft Visual C++ 14.x with Visual Studio 2022 (x86, x64, ARM, ARM64)" from
# https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.2_standalone:_Build_Tools_for_Visual_Studio_2019_.28x86.2C_x64.2C_ARM.2C_ARM64.29
# pip install -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7/index.html mmcv-full==1.2.7
pip install -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html mmcv-full==1.2.7
pip install scipy
pip install timm==0.3.2
pip install yapf==0.40.1
```

conda activate /home/it4i-balkin/scratch/env/maeseg4
./run.sh dbot_base_ade20k-seg ade20k_seg base 1 models/mae_base16.pth optimizer_config.use_fp16=True optimizer.lr=1e-4 optimizer.paramwise_cfg.layer_decay_rate=0.65 model.backbone.drop_path_rate=0.2 