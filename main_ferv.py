import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
import time
import numpy as np
import random
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, plot_confusion_matrix
from utils.logger import create_logger
from datasets.build import build_dataloader
from datasets.blending import FixMixupBlending
from utils.config import get_config
from models import AU_clip
import torch.nn.functional as F

K_TABLE_FERV39K = {
    0: [0.26197021302251783, 0.2637489312360733, 0.2964310301204501, 0.13190483696240127,
        0.6939537051210115, 0.8330800031639156, 0.17347610820526463, 0.8420321208563623,
        0.8882934142237333, 0.7356394920285989, 0.15360866990050534, 0.24664740708336055,
        0.17509971698083543, 0.1276861973423668, 0.8057327549388998, 0.41178926421868306,
        0.2908397350173472, 0.19836039970638378],  # happy
    1: [0.2470265618083506, 0.1433979246927842, 0.6881822893577378, 0.11893982182697348,
        0.31161634476423544, 0.5253221085983302, 0.12597712051107948, 0.4422623779963743,
        0.27645211415344356, 0.42395131504088984, 0.17558170449960314, 0.27142133890425624,
        0.1741388696479932, 0.1227455638528141, 0.4485394778155102, 0.41664204112281555,
        0.2908397350173472, 0.2121307850723057],  # sad
    2: [0.19974525492944142, 0.17419319637385966, 0.4053456334801515, 0.13706230525786608,
        0.16464150662864904, 0.3564013123982043, 0.1117065857762667, 0.30225692349830036,
        0.17199394819786643, 0.28648655803768563, 0.14721999753281687, 0.22595757296022984,
        0.14108230584951664, 0.11916179838996314, 0.31830472814177574, 0.36213049142283893,
        0.2908397350173472, 0.16214811486455724],  # neutral
    3: [0.16978330014266482, 0.1474484099458089, 0.6385375278419455, 0.15955087911488458,
        0.2257386547042889, 0.4972455283651611, 0.15269462267081732, 0.470889570126452,
        0.1713443932394683, 0.28682403662377504, 0.17023111167232707, 0.28941226670462944,
        0.14671370568908396, 0.12489530852760639, 0.6019091910383833, 0.5023565940498623,
        0.2908397350173472, 0.1480589426871144],  # angry
    4: [0.3340537870187493, 0.31316038897162674, 0.4508609549567622, 0.2215642997098194,
        0.18256790904316805, 0.36794740171201545, 0.14342865607335295, 0.3325723523253354,
        0.20291790550542987, 0.24479399356303455, 0.15335881131831128, 0.24479399356303455,
        0.14621845929676547, 0.12047847974103684, 0.4921072671708563, 0.5073560806954985,
        0.2908397350173472, 0.15234747176484023],  # surprise
    5: [0.2556613250379876, 0.15655681634966592, 0.7784678175136164, 0.13563151267233145,
        0.353602732625092, 0.8594560438530944, 0.3484770561236273, 0.7265667054811852,
        0.2573762546624107, 0.4833627557878272, 0.26077234063862126, 0.3163550480772833,
        0.1807057096038953, 0.1488894096862477, 0.6161944289089377, 0.44359769238463886,
        0.2908397350173472, 0.21474224698158992],  # disgust
    6: [0.3416233199942143, 0.22096625856984287, 0.591078697660434, 0.2210340665261555,
        0.19066749128518232, 0.35671400783212986, 0.1239196009045801, 0.3647782154323708,
        0.2014525981038694, 0.28661445976904537, 0.21138892966648018, 0.30233592318202696,
        0.18447197078258057, 0.12964812274717674, 0.5106650931015151, 0.5397724065827655,
        0.2908397350173472, 0.1432476989807878],  # fear
}

K_SCALE = 5.0

POSW_GLOBAL_FERV39K = [4.789480561907873, 4.554315802670344, 1.4431454725929194, 1.2930490140261892,
                       10.91287980639957, 2.960842161727236, 12.93426639408712, 4.830240821160679,
                       9.943312337902928, 3.558962749536942, 5.529216712106698, 3.1841620626151017,
                       10.882526485181709, 8.180273518441775, 3.816699282452707, 5.920877919237679,
                       128.54385964912282, 5.021610601427115]

POSW_DISTINCT_FERV39K = {
    0: [4.737603191312538, 4.541120445181658, 1.9399841017488075, 3.0183148051141764,
        1.8016286336521163, 1.0785805325010287, 12.648500823723229, 1.2691427006299645,
        1.6937363437727604, 1.317200331162874, 4.711181580035848, 2.9899824693213124,
        10.53271340275071, 5.179984484096199, 3.398861609039164, 5.603832541529828,
        123.02155688622754, 4.838200473559589],  # happy
    1: [6.529455713698681, 6.403070594284875, 0.9404640415904496, 2.296684603935295,
        6.959404493938317, 2.2288739727344087, 17.447373238147538, 4.3383833033160295,
        11.864181771764105, 2.7964965153512904, 7.251361198673599, 3.9084577356744514,
        13.404059172443365, 8.52803252339983, 4.647724725397893, 6.739948542682693,
        163.66993464052288, 6.3258459637262385],  # sad
    2: [5.821125337847061, 5.533247914651048, 1.3883817189396908, 2.207288738270369,
        7.343384759233286, 2.429757851965737, 15.828382838283828, 4.222092172640819,
        10.11337959256294, 2.900508929686524, 6.057539122083498, 3.320439218398755,
        11.676357179096906, 6.140436112143123, 4.243682840832774, 6.373498598199793,
        133.32849462365593, 5.4467695324594905],  # neutral
    3: [5.302377957822691, 4.934069565696346, 0.9931917412861688, 1.7729132885491468,
        6.890146105679446, 2.4329960965506254, 14.58004338394794, 3.350684488955296,
        14.718704406186168, 3.104384928949674, 5.258444915623457, 3.049844939385396,
        9.561835204156658, 6.84933153619176, 3.5544705136334813, 5.460348394447276,
        118.57380688124306, 5.117713863888022],  # angry
    4: [4.789480561907873, 4.554315802670344, 1.4431454725929194, 1.2930490140261892,
        10.91287980639957, 2.960842161727236, 12.934266394087121, 4.830240821160679,
        9.943312337902928, 3.5589627495369416, 5.529216712106698, 3.1841620626151013,
        10.882526485181709, 8.180273518441775, 3.8166992824527073, 5.920877919237679,
        128.5438596491228, 5.021610601427115],  # surprise
    5: [4.931708599982467, 4.588288734720845, 0.9527561327561328, 2.4336242768699887,
        4.1825214460784315, 1.9008788853161844, 13.448644031603672, 2.618535750574897,
        8.57043847241867, 2.3589654487688643, 5.042957935161204, 3.0051497573102877,
        9.752105514063246, 5.854031604538087, 3.6751191874524975, 5.442254593925545,
        109.20032573289902, 4.523510204081632],  # disgust
    6: [6.047279943964511, 5.658614604015001, 0.9386300138090498, 1.6440804169769174,
        10.258299142111152, 2.639197009886665, 17.34305682163476, 4.947487684729064,
        15.796605453533667, 3.207639227713111, 6.6501077176530226, 3.6865150221256116,
        12.636096679466908, 7.38081354990976, 4.699301359516617, 6.53645443196005,
        152.21573604060913, 5.966762839007502],  # fear
}

MAJOR_CLASSES_FERV39K = {0, 1, 2, 3}


def compute_au_loss_stage_b(au_logits, au_targets, labels, posw_option='global'):
    """
    au_logits:  [B,18] float
    au_targets: [B,18] float {0,1}
    labels:     [B] long, emotion class ids (0..C-1)
    """
    device = au_logits.device
    B = labels.shape[0]
    # pos_weight: [B,18]
    if config.DATA.DATASET=='FERV39K':
        # knowledge weight: [B,18]
        k = torch.stack(
            [torch.tensor(K_TABLE_FERV39K[int(c)], device=device, dtype=torch.float32) for c in labels],
            dim=0
        ) * K_SCALE
        if posw_option == 'global':
            pw = torch.tensor(POSW_GLOBAL_FERV39K, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
        elif posw_option == 'distinct':
            pw = torch.stack(
                [torch.tensor(POSW_DISTINCT_FERV39K[int(c)], device=device, dtype=torch.float32) for c in labels],
                dim=0
            )
        elif posw_option == 'minor':
            pw = torch.stack(
                [torch.tensor(POSW_DISTINCT_FERV39K[int(c)], device=device, dtype=torch.float32) for c in labels],
                dim=0
            )
            # overwrite major classes with 1s
            mask_major = torch.tensor([int(c) in MAJOR_CLASSES_FERV39K for c in labels], device=device, dtype=torch.bool)
            if mask_major.any():
                pw = pw.clone()
                pw[mask_major] = 1.0
        else:
            raise ValueError(f'Unknown posw_option: {posw_option}')

    # weighted BCE with logits
    return F.binary_cross_entropy_with_logits(
        au_logits, au_targets,
        weight=k,
        pos_weight=pw,
        reduction='mean'
    )

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/dfew7/16_16.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--gpu', default=[0, 1], type=int,help='GPU id to use.')
    parser.add_argument('--output', type=str, default="DFEWAS2")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def main(config):
    # load train and valid dataset
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

    # load pretrained model
    model, _ = AU_clip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
                          device="cpu", jit=False,
                          T=config.DATA.NUM_FRAMES,
                          droppath=config.MODEL.DROP_PATH_RATE,
                          use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                          use_cache=config.MODEL.FIX_TEXT,
                          logger=logger,
                          N=config.DATA.NUM_DIVIDE,
                          cfg=config
                          )
    model = model.cuda()

    # training data augmentation
    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        criterion_soft = SoftTargetCrossEntropy()
        mixup_fn = FixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
                                    smoothing=config.AUG.LABEL_SMOOTH,
                                    mixup_alpha=config.AUG.MIXUP,
                                    fmix_alpha=config.AUG.CUTMIX,
                                    switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
        criterion_soft = SoftTargetCrossEntropy()

    else:
        criterion = nn.CrossEntropyLoss()
        criterion_soft = SoftTargetCrossEntropy()

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
                                                      find_unused_parameters=True)

    start_epoch = 0
    max_war_global, max_war_local, max_war_fuse = 0.0, 0.0, 0.0

    # Track best epoch information (all from the same epoch with best WAR)
    best_epoch = 0
    best_war_global = 0.0
    best_uar_global = 0.0
    best_cls_acc_global = None
    best_war_local = 0.0
    best_uar_local = 0.0
    best_cls_acc_local = None
    best_war_fuse = 0.0
    best_uar_fuse = 0.0
    best_cls_acc_fuse = None
    # retrain
    if config.TRAIN.AUTO_RESUME:
        resume_file_path = auto_resume_helper(config.OUTPUT)
        if resume_file_path:
            config.defrost()
            config.MODEL.RESUME = resume_file_path
            config.freeze()
            logger.info(f'auto resuming from {resume_file_path}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    if config.MODEL.RESUME:
        start_epoch, max_war_global, best_state = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)
        max_war_local = best_state.get('max_war_local', 0.)
        max_war_fuse = best_state.get('max_war_fuse', 0.)
        best_epoch = best_state.get('best_epoch', 0)
        best_uar_global = best_state.get('best_uar_global', 0.)
        best_uar_local = best_state.get('best_uar_local', 0.)
        best_uar_fuse = best_state.get('best_uar_fuse', 0.)

    # textual prompt
    text_labels = generate_text(train_data)

    # model test
    if config.TEST.ONLY_TEST:
        results = validate(val_loader, text_labels, model, config)
        acc1, acc1_local, acc1_fuse = results[0], results[1], results[2]
        uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: WAR={acc1:.1f}% UAR={uar_global:.1f}%")
        return
    print("Start training")
    #model train and valid
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft)

        # Global, local and fuse classification accuracy
        results = validate(val_loader, text_labels, model, config)
        acc_global, acc_local, acc_fuse = results[0], results[1], results[2]
        uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
        war_global, war_local, war_fuse = results[6], results[7], results[8]
        cls_acc_global, cls_acc_local, cls_acc_fuse = results[9], results[10], results[11]

        # Get emotion class names
        if config.DATA.DATASET.lower() == 'FERV39K':
            emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
        else:
            emotion_names = [f'Class_{i}' for i in range(len(cls_acc_global))]

        # Log detailed results for current epoch
        logger.info(f"=" * 80)
        logger.info(f"Epoch [{epoch}/{config.TRAIN.EPOCHS - 1}] Validation Results:")
        logger.info(f"-" * 80)

        # Global results
        logger.info(f"Global Branch:")
        logger.info(f"  WAR: {war_global:.2f}%  |  UAR: {uar_global:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, cls_acc_global):
            logger.info(f"    {name:10s}: {acc*100:.2f}%")

        # Local results
        logger.info(f"-" * 80)
        logger.info(f"Local Branch:")
        logger.info(f"  WAR: {war_local:.2f}%  |  UAR: {uar_local:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, cls_acc_local):
            logger.info(f"    {name:10s}: {acc*100:.2f}%")

        # Fuse results
        logger.info(f"-" * 80)
        logger.info(f"Fuse Branch:")
        logger.info(f"  WAR: {war_fuse:.2f}%  |  UAR: {uar_fuse:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, cls_acc_fuse):
            logger.info(f"    {name:10s}: {acc*100:.2f}%")
        logger.info(f"=" * 80)

        is_best = war_global > max_war_global  # Use WAR (acc_global) as the criterion for best model

        # Update best epoch info - only update when finding a new best WAR
        if is_best:
            best_epoch = epoch
            # Update all metrics from the SAME epoch (the one with best WAR)
            best_war_global = war_global
            best_uar_global = uar_global
            best_cls_acc_global = cls_acc_global.copy()
            best_war_local = war_local
            best_uar_local = uar_local
            best_cls_acc_local = cls_acc_local.copy()
            best_war_fuse = war_fuse
            best_uar_fuse = uar_fuse
            best_cls_acc_fuse = cls_acc_fuse.copy()

            # Update max values - these are from the best epoch
            max_war_global = war_global
            max_war_local = war_local
            max_war_fuse = war_fuse

            logger.info(f">>> New best model found at epoch {epoch}! WAR: {war_global:.2f}% UAR: {uar_global:.2f}% <<<")

        logger.info(f'Current Best Epoch: {best_epoch}')
        logger.info(f'Best Global - WAR: {max_war_global:.2f}% | UAR: {best_uar_global:.2f}%')
        logger.info(f'Best Local  - WAR: {max_war_local:.2f}% | UAR: {best_uar_local:.2f}%')
        logger.info(f'Best Fuse   - WAR: {max_war_fuse:.2f}% | UAR: {best_uar_fuse:.2f}%')
        # save model
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_war_global, optimizer, lr_scheduler, logger, config.OUTPUT,
                         is_best, best_epoch=best_epoch,
                         max_war_local=max_war_local, max_war_fuse=max_war_fuse,
                         best_uar_global=best_uar_global, best_uar_local=best_uar_local, best_uar_fuse=best_uar_fuse)

    # Print best epoch summary
    logger.info(f"\n" + "=" * 80)
    logger.info(f"TRAINING COMPLETED - BEST MODEL SUMMARY")
    logger.info(f"=" * 80)
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"-" * 80)

    # Get emotion class names
    if config.DATA.DATASET.lower() == 'FERV39K':
        emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
    else:
        emotion_names = [f'Class_{i}' for i in range(len(best_cls_acc_global)) if best_cls_acc_global is not None]

    if best_cls_acc_global is not None:
        logger.info(f"Global Branch (Best):")
        logger.info(f"  WAR: {best_war_global:.2f}%  |  UAR: {best_uar_global:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, best_cls_acc_global):
            logger.info(f"    {name:10s}: {acc*100:.2f}%")

        logger.info(f"-" * 80)
        logger.info(f"Local Branch (Best):")
        logger.info(f"  WAR: {best_war_local:.2f}%  |  UAR: {best_uar_local:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, best_cls_acc_local):
            logger.info(f"    {name:10s}: {acc*100:.2f}%")

        logger.info(f"-" * 80)
        logger.info(f"Fuse Branch (Best):")
        logger.info(f"  WAR: {best_war_fuse:.2f}%  |  UAR: {best_uar_fuse:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, best_cls_acc_fuse):
            logger.info(f"    {name:10s}: {acc*100:.2f}%")
    logger.info(f"=" * 80 + "\n")

    # validation after training
    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    results = validate(val_loader, text_labels, model, config)
    acc = results[0]
    uar_global = results[3]
    logger.info(f"Final Accuracy: WAR={acc:.2f}% UAR={uar_global:.2f}%")


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft):
    au_criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()
    scaler = GradScaler()
    texts = text_labels.cuda(non_blocking=True)

    for idx, batch_data in enumerate(train_loader):
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        au_labels = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
        label_id = label_id.reshape(-1)
        images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        au_targets = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
        original_label_id = label_id.clone() if config.AU.ENABLED and mixup_fn is not None else None

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)
            # Apply the same mixup to AU labels using stored parameters
            if config.AU.ENABLED and hasattr(mixup_fn, 'last_lam') and hasattr(mixup_fn, 'last_rand_index'):
                lam = mixup_fn.last_lam
                rand_index = mixup_fn.last_rand_index
                # Mix AU labels with the same lambda and permutation
                au_targets = lam * au_targets + (1 - lam) * au_targets[rand_index, :]

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)
        with autocast():
            output_global, output_local, feat, au_logits = model(images, texts)
        output_global = output_global.float()
        output_local = output_local.float()
        au_logits = au_logits.float()
        if epoch > 13:
            total_loss = criterion(output_global, label_id)
            total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
        else:
            pre_output_local = torch.sum(
                output_local.view(output_local.shape[0], config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
            global_loss = criterion(output_global, label_id)
            local_loss = criterion(pre_output_local, label_id)
            kl_loss = criterion_soft(pre_output_local.clamp(-50, 50).softmax(dim=-1), output_global.clamp(-50, 50).softmax(dim=-1))

            # Check and zero out NaN losses
            if torch.isnan(local_loss) or torch.isinf(local_loss):
                logger.warning(f'[Epoch {epoch} Step {idx}] local_loss nan/inf, zeroing. global={global_loss.item():.4f} pre_output_local range=[{pre_output_local.min().item():.4f},{pre_output_local.max().item():.4f}]')
                local_loss = torch.zeros(1, device=output_global.device)
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                logger.warning(f'[Epoch {epoch} Step {idx}] kl_loss nan/inf, zeroing. global={global_loss.item():.4f}')
                kl_loss = torch.zeros(1, device=output_global.device)

            total_loss = global_loss + 1.0 * (local_loss + kl_loss)
            total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        # Compute AU loss regardless of mixup status
        # When mixup is enabled, AU labels are also mixed with the same parameters
        if config.AU.ENABLED:
            # Use original emotion labels before mixup for AU loss computation
            # because compute_au_loss_stage_b needs hard emotion class labels for knowledge weighting
            labels_for_au = original_label_id if original_label_id is not None else label_id
            au_loss = compute_au_loss_stage_b(
                au_logits, au_targets, labels_for_au,
                posw_option=config.AU.POSW_OPTION
            )
            if torch.isnan(au_loss) or torch.isinf(au_loss):
                logger.warning(f'[Epoch {epoch} Step {idx}] au_loss is {au_loss.item():.4f}, au_logits range=[{au_logits.min().item():.4f}, {au_logits.max().item():.4f}], au_targets range=[{au_targets.min().item():.4f}, {au_targets.max().item():.4f}], skipping au_loss')
            else:
                #total_loss = total_loss + config.AU.LAMBDA * au_loss
                total_loss = total_loss * (1 - config.AU.LAMBDA) + config.AU.LAMBDA * au_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f'[Epoch {epoch} Step {idx}] total_loss is {total_loss.item():.4f}, output_global range=[{output_global.min().item():.4f}, {output_global.max().item():.4f}], skipping update')
            optimizer.zero_grad()
            continue

        if config.TRAIN.OPT_LEVEL != 'O0':
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                if config.TRAIN.OPT_LEVEL != 'O0':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            if config.TRAIN.OPT_LEVEL != 'O0':
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(val_loader, text_labels, model, config):
    model.eval()

    acc_global_meter, acc_local_meter, acc_fuse_meter = AverageMeter(), AverageMeter(), AverageMeter()

    probility = []
    video_pre_global = []
    video_pre_local = []
    video_pre_fuse = []
    video_label = []
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()

            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            tot_similarity_local = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):
                image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                with autocast():
                    output, output_local, feat, _ = model(image_input, text_inputs)
                if idx < 1:
                    feature = feat
                else:
                    feature = torch.cat((feature, feat), dim=0)

                pre_output_global = output.view(b, -1)
                pre_output_local = torch.sum(output_local.view(b, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES),dim=1).squeeze(dim=-1)

                similarity = pre_output_global.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity

                similarity_local = pre_output_local.view(b, -1).softmax(dim=-1)
                tot_similarity_local += similarity_local.view(b, -1)

            probility.extend(tot_similarity.data.cpu().numpy().copy())
            values_global, indices_global = tot_similarity.topk(1, dim=-1)

            values_local, indices_local = tot_similarity_local.topk(1, dim=-1)

            fuse_similarity = tot_similarity + tot_similarity_local
            values_fuse, indices_fuse = fuse_similarity.topk(1, dim=-1)

            acc_global = 0
            acc_local = 0
            acc_fuse = 0
            for i in range(b):
                video_pre_global.append(indices_global[i].data.cpu().numpy().copy())
                video_pre_local.append(indices_local[i].data.cpu().numpy().copy())
                video_pre_fuse.append(indices_fuse[i].data.cpu().numpy().copy())
                video_label.append(label_id[i].data.cpu().numpy().copy())
                if indices_global[i] == label_id[i]:
                    acc_global += 1
                if indices_local[i] == label_id[i]:
                    acc_local += 1
                if indices_fuse[i] == label_id[i]:
                    acc_fuse += 1

            acc_global_meter.update(float(acc_global) / b * 100, b)
            acc_local_meter.update(float(acc_local) / b * 100, b)
            acc_fuse_meter.update(float(acc_fuse) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc_global_meter.avg:.3f}\t'
                    f'Acc@1: {acc_local_meter.avg:.3f}\t'
                    f'Acc@1: {acc_fuse_meter.avg:.3f}\t'
                )
    # confusion matrix
    cf = confusion_matrix(video_label, video_pre_global)
    np.set_printoptions(precision=4)
    normalized_cm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100

    cls_cnt = normalized_cm.sum(axis=1)
    cls_hit = np.diag(normalized_cm)
    # print(cf)
    cls_acc = cls_hit / cls_cnt
    cls_acc = np.around(cls_acc, 4)
    cm = np.array(normalized_cm)


    logger.info(f'Global - Class-wise Accuracy: {cls_acc}')
    upper = np.mean(np.max(cf, axis=1) / cls_cnt)
    logger.info(f'Global - Upper bound: {upper}')
    logger.info('Global - Evaluation is finished')
    logger.info(f'Global - Class Accuracy (UAR): {np.mean(cls_acc) * 100:.2f}%')

    cf_local = confusion_matrix(video_label, video_pre_local).astype(float)
    cls_cnt_local = cf_local.sum(axis=1)
    cls_hit_local = np.diag(cf_local)
    # print(cf)
    cls_acc_local = cls_hit_local / cls_cnt_local
    cls_acc_local = np.around(cls_acc_local, 4)
    logger.info(f'Local - Class-wise Accuracy: {cls_acc_local}')
    upper = np.mean(np.max(cf_local, axis=1) / cls_cnt_local)
    logger.info(f'Local - Upper bound: {upper}')
    logger.info('Local - Evaluation is finished')
    logger.info(f'Local - Class Accuracy (UAR): {np.mean(cls_acc_local) * 100:.2f}%')

    cf_fuse = confusion_matrix(video_label, video_pre_fuse).astype(float)
    cls_cnt_fuse = cf_fuse.sum(axis=1)
    cls_hit_fuse = np.diag(cf_fuse)
    # print(cf)
    cls_acc_fuse = cls_hit_fuse / cls_cnt_fuse
    cls_acc_fuse = np.around(cls_acc_fuse, 4)
    logger.info(f'Fuse - Class-wise Accuracy: {cls_acc_fuse}')
    upper = np.mean(np.max(cf_fuse, axis=1) / cls_cnt_fuse)
    logger.info(f'Fuse - Upper bound: {upper}')
    logger.info('Fuse - Evaluation is finished')
    logger.info(f'Fuse - Class Accuracy (UAR): {np.mean(cls_acc_fuse) * 100:.2f}%')

    acc_global_meter.sync()
    acc_local_meter.sync()
    acc_fuse_meter.sync()
    logger.info(f' * Acc@1 {acc_global_meter.avg:.3f} Acc_loca@1 {acc_local_meter.avg:.3f}  Acc_fuse@1 {acc_fuse_meter.avg:.3f}')

    # Calculate UAR (Unweighted Average Recall) and WAR (Weighted Average Recall)
    uar_global = np.mean(cls_acc) * 100  # UAR for global
    uar_local = np.mean(cls_acc_local) * 100  # UAR for local
    uar_fuse = np.mean(cls_acc_fuse) * 100  # UAR for fuse
    war_global = acc_global_meter.avg  # WAR is the same as overall accuracy
    war_local = acc_local_meter.avg
    war_fuse = acc_fuse_meter.avg

    return (acc_global_meter.avg, acc_local_meter.avg, acc_fuse_meter.avg,
            uar_global, uar_local, uar_fuse, war_global, war_local, war_fuse,
            cls_acc, cls_acc_local, cls_acc_fuse,
            video_label, video_pre_global, video_pre_local, video_pre_fuse)


if __name__ == '__main__':
    args, config = parse_option()

    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK']) 
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
        local_rank = args.local_rank  

    
    torch.cuda.set_device(local_rank)

   
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    dist.barrier()

    
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

   
    output_dir = Path(config.OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)

   
    logger = create_logger(output_dir=output_dir, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"Working directory: {output_dir}")

    
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, output_dir)

   
    main(config)


























































































