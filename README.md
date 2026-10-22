# AU-CLIP: Action-Unit-Guided Vision-Language Learning for Dynamic Facial Expression Recognition

<p align="center">
  <img src="img/framework.png" alt="Overview of the AU-CLIP framework" width="800">
</p>

## Abstract

Dynamic facial expression recognition (DFER) is important for understanding human affective states from video sequences. However, existing methods often rely on coarse-grained expression supervision and limited textual semantics, making it difficult to capture subtle facial movements that distinguish different expressions. Moreover, temporal modeling often overlooks irrelevant or weak-expression frames, which may introduce noise and weaken discriminative expression representations. 
To address these challenges, we propose AU-CLIP, an action-unit-guided dual-path learning framework with reliability-aware temporal modeling for DFER. Specifically, we construct class-dependent Action Unit (AU) priors to identify facial actions associated with different expressions. Based on these priors, we design AU-guided visual supervision and AU-enriched textual descriptions to enhance fine-grained facial representation learning and expression semantics, respectively. To capture facial dynamics while mitigating temporal noise, we develop a local-to-global temporal modeling strategy that models facial changes within short snippets and integrates information across snippets to obtain global temporal context. Global-local consistency is used to reduce the influence of unreliable temporal content and select visual tokens for video-conditioned text refinement. Finally, global and local visual representations are matched with the refined textual features, and a stage-wise optimization strategy combines global and local expression classification, prediction consistency, and AU supervision. Experiments on three public datasets demonstrate the effectiveness of AU-CLIP.

## Configuration

Before running experiments, edit the configuration file for the dataset you want to use:

- DFEW: `configs/dfew/dfew.yaml`
- FERV39K: `configs/ferv39k/ferv39k.yaml`
- MAFW: `configs/mafw/mafw.yaml`

Set `DATA.ROOT` to the video root and update `DATA.TRAIN_FILE` and `DATA.VAL_FILE` to the corresponding split lists. Check that `DATA.LABEL_LIST` points to the class descriptions in `labels2prompt/` and `AU.LABEL_PATH` points to the AU annotations in `AU_datasets/`. The default model uses the ViT-B/16 CLIP checkpoint, which is downloaded automatically when `MODEL.PRETRAINED` is `None`.

## Setup and Run

Install the required packages in a Linux environment with CUDA-enabled PyTorch:

```bash
git clone https://github.com/zhangCMay/AU-CLIP.git
cd AU-CLIP
pip install -r requirements.txt
```

Run training from the repository root with the matching entry point and configuration:

```bash
# DFEW
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12355 \
  main_dfew.py --config=configs/dfew/dfew.yaml --output=outputs/dfew

# FERV39K
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12355 \
  main_ferv.py --config=configs/ferv39k/ferv39k.yaml --output=outputs/ferv39k

# MAFW
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12355 \
  main_mafw.py --config=configs/mafw/mafw.yaml --output=outputs/mafw
```

