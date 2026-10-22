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
K_TABLE_MAFW = {
    0: [0.37458275378581574, 0.37673577978763106, 0.41551698634677164, 0.20406588498862172,
        0.7927884197377446, 0.8938573401735845, 0.26152800062149906, 0.8999413926399066,
        0.9306407335956202, 0.8244182955003968, 0.2344366715203141, 0.35584905526535116,
        0.2637127706449251, 0.1980659294341865, 0.8749731256118369, 0.5415473658691405,
        0.4089852072091214, 0.29454183039387216],
    1: [0.3563166809248465, 0.22025123080620432, 0.7883122756737404, 0.18552393229679323, 0.4330479071201498, 0.6512463687571445, 0.19562608246203084, 0.57228053429373, 0.3919835645137292, 0.5539327475091499, 0.26436050797140115, 0.3859719011102806, 0.2624203556065227, 0.19099832378533962, 0.5784889747902296, 0.5465085543388307, 0.4089852072091214, 0.31238729511297314],  # sad
    2: [0.29634994040489937, 0.26249346997682327, 0.5349197805247116, 0.211357886947774, 0.24956316484226185, 0.48303878792569305, 0.1750459267403643, 0.4222786033590969, 0.2595297603319664, 0.4038707629862569, 0.2255822802606922, 0.3300116586908106, 0.21700897117154114, 0.18584396273490125, 0.4406741647399928, 0.4892561535112399, 0.4089852072091214, 0.24616267045793816],  # neutral
    3: [0.2565426020973117, 0.22590006587965014, 0.7487900008174104, 0.24260938453624706, 0.3297348731445176, 0.6253054547733351, 0.2331741603449138, 0.6002659975024218, 0.25865288585343615, 0.4042681727124247, 0.25714836786909934, 0.4073109215604269, 0.22487756342746285, 0.1940789790636582, 0.7184070119769908, 0.6300831522957303, 0.4089852072091214, 0.22674904330132986],  # angry
    4: [0.45840639969725916, 0.4348135864186588, 0.5807746998255159, 0.32444326825812997, 0.27370636575780566, 0.49552874169247974, 0.22029419679136286, 0.45675173141757053, 0.30048086399990936, 0.35356016873960344, 0.2340917014067623, 0.35356016873960344, 0.22418778976044848, 0.18774040869030503, 0.6204770513319532, 0.6347324688342945, 0.4089852072091214, 0.2326942881893769],  # surprise
    5: [0.3669101379226728, 0.23849898584920362, 0.8556853679596857, 0.2093396686826401, 0.47998741884320045, 0.9116481111605065, 0.47437419451562296, 0.8176370658093514, 0.3690013619929339, 0.6122010138157353, 0.37313028939087656, 0.43845703949339593, 0.2712229960450541, 0.22790281312128532, 0.7303843653361992, 0.5736046807175323, 0.4089852072091214, 0.3157383637072392],  # disgust
    6: [0.4668184913800416, 0.3236830048705496, 0.7092147921039609, 0.3237692334141919, 0.284442322105034, 0.4833791428919572, 0.19268179747966238, 0.49211626418097015, 0.29857493916455136, 0.40402139627383793, 0.3114334189399665, 0.42236998344528387, 0.2762397282204001, 0.20086024967771943, 0.6377967064381651, 0.6643128457350147, 0.4089852072091214, 0.22004117429592432],  # fear
}
K_SCALE = 5.0

POSW_GLOBAL_MAFW = [
    61.279402753029004, 37.4999965909094, 0.960648143700703, 1.7340219384504716,
    3.1438355856767557, 1.9126547323751395, 150.0, 1.47371494466288,
    3.5932754490967955, 1.6255424573741943, 23.622091649878392, 10.698894732074733,
    91.06519759452226, 7.453093663610905, 33.153223132804584, 150.0,
    150.0, 65.17186481689612
]

POSW_DISTINCT_MAFW = {
    0: [50.190452290260815, 36.06895307967135, 2.218562807827461, 2.644067706980756, 0.7976588495374775, 0.9907407223936904, 150.0, 0.44295301418855015, 0.6362252566784589, 0.46059781982883397, 32.593739814456306, 11.499998662790853, 81.69224485211934, 6.570422072505488, 28.054046471879328, 150.0, 150.0, 66.18745863283836],  # happy
    1: [42.57139816328703, 37.124976796889506, 0.3555555476543212, 2.3333332058287866, 3.295774415790534, 1.4399999424000023, 150.0, 1.6406925696669883, 6.349396825373876, 1.8240739896262042, 20.78570686224755, 9.892855376275826, 150.0, 10.960782164552516, 31.105246786712218, 150.0, 150.0, 42.57139816328703],  # sad
    2: [150.0, 48.47823979206966, 1.3415637584040379, 1.6100917061905573, 8.17741869536946, 3.4108525809747063, 150.0, 2.6357826633935253, 10.048542713733717, 1.910485884642305, 34.56248919922212, 18.288132493536864, 93.83325513895406, 5.4293782243289135, 48.47823979206966, 150.0, 150.0, 86.5383949704654],  # neutral
    3: [46.2221708642546, 41.4999585000415, 0.6409266161804396, 1.335164761804134, 7.4999985000002996, 2.5416664548611285, 150.0, 1.7243588638231497, 22.611098549389695, 3.381442950366706, 15.99999360000256, 6.870369098079796, 105.24973687565782, 9.89743335963247, 34.416637986135015, 150.0, 150.0, 83.999832000336],  # angry
    4: [54.222161975375585, 28.23527750866029, 0.7137930788347214, 0.6790540311130395, 18.115377647931673, 4.020201614121049, 150.0, 4.344085554399403, 23.849988075005964, 6.202897651753963, 21.590899276863965, 8.557690661982564, 61.12492359384551, 12.078944189751528, 30.062481210949244, 123.24969187577032, 150.0, 61.12492359384551],  # surprise
    5: [60.999878000244, 33.44440728399191, 0.46226412913848447, 3.1891887582177354, 4.08196654393991, 1.4799998816000095, 150.0, 1.4799998816000095, 8.393936850322167, 2.9240502627784477, 21.142842040827116, 7.611108996914167, 150.0, 11.399995440001824, 33.44440728399191, 150.0, 150.0, 43.28565244906793],  # disgust
    6: [58.999803333988886, 28.999951666747226, 0.395348806562108, 1.1686745579910172, 7.571424965988111, 1.903225499479758, 150.0, 2.673468842149216, 21.499973125033595, 6.199997520000992, 8.473679750694869, 8.473679750694869, 58.999803333988886, 5.206894756243187, 21.499973125033595, 43.999890000275, 150.0, 150.0]   # fear
}


MAJOR_CLASSES_MAFW = {0, 1, 2, 3}

def compute_au_loss_stage_b(au_logits, au_targets, labels, posw_option='global'):
    """
    au_logits:  [B,18] float
    au_targets: [B,18] float {0,1}
    labels:     [B] long, emotion class ids (0..C-1)
    """
    device = au_logits.device
    B = labels.shape[0]
    # pos_weight: [B,18]
    if config.DATA.DATASET.upper() == 'MAFW':
        # knowledge weight: [B,18]
        k = torch.stack(
            [torch.tensor(K_TABLE_MAFW[int(c)], device=device, dtype=torch.float32) for c in labels],
            dim=0
        ) * K_SCALE
        if posw_option == 'global':
            pw = torch.tensor(POSW_GLOBAL_MAFW, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
        elif posw_option == 'distinct':
            pw = torch.stack(
                [torch.tensor(POSW_DISTINCT_MAFW[int(c)], device=device, dtype=torch.float32) for c in labels],
                dim=0
            )
        elif posw_option == 'minor':
            pw = torch.stack(
                [torch.tensor(POSW_DISTINCT_MAFW[int(c)], device=device, dtype=torch.float32) for c in labels],
                dim=0
            )
            # overwrite major classes with 1s
            mask_major = torch.tensor([int(c) in MAJOR_CLASSES_MAFW for c in labels], device=device, dtype=torch.bool)
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
    max_uar_global, max_uar_local, max_uar_fuse = 0.0, 0.0, 0.0

    # Track best WAR epoch information
    best_war_epoch = 0
    best_war_global = 0.0
    best_war_uar_global = 0.0  # UAR at the epoch with best WAR
    best_war_cls_acc_global = None
    best_war_local = 0.0
    best_war_uar_local = 0.0
    best_war_cls_acc_local = None
    best_war_fuse = 0.0
    best_war_uar_fuse = 0.0
    best_war_cls_acc_fuse = None

    # Track best UAR epoch information (separate tracking)
    best_uar_epoch = 0
    best_uar_global = 0.0
    best_uar_war_global = 0.0  # WAR at the epoch with best UAR
    best_uar_cls_acc_global = None
    best_uar_local = 0.0
    best_uar_war_local = 0.0
    best_uar_cls_acc_local = None
    best_uar_fuse = 0.0
    best_uar_war_fuse = 0.0
    best_uar_cls_acc_fuse = None
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
        max_uar_global = best_state.get('max_uar_global', 0.)
        max_uar_local = best_state.get('max_uar_local', 0.)
        max_uar_fuse = best_state.get('max_uar_fuse', 0.)

        # Best WAR epoch info
        best_war_epoch = best_state.get('best_war_epoch', 0)
        best_war_uar_global = best_state.get('best_war_uar_global', 0.)
        best_war_uar_local = best_state.get('best_war_uar_local', 0.)
        best_war_uar_fuse = best_state.get('best_war_uar_fuse', 0.)

        # Best UAR epoch info
        best_uar_epoch = best_state.get('best_uar_epoch', 0)
        best_uar_global = best_state.get('best_uar_global', 0.)
        best_uar_local = best_state.get('best_uar_local', 0.)
        best_uar_fuse = best_state.get('best_uar_fuse', 0.)
        best_uar_war_global = best_state.get('best_uar_war_global', 0.)
        best_uar_war_local = best_state.get('best_uar_war_local', 0.)
        best_uar_war_fuse = best_state.get('best_uar_war_fuse', 0.)

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
        if config.DATA.DATASET.upper() == 'MAFW':
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

        # Track best WAR model
        is_best_war = war_global > max_war_global
        if is_best_war:
            best_war_epoch = epoch
            best_war_global = war_global
            best_war_uar_global = uar_global
            best_war_cls_acc_global = cls_acc_global.copy()
            best_war_local = war_local
            best_war_uar_local = uar_local
            best_war_cls_acc_local = cls_acc_local.copy()
            best_war_fuse = war_fuse
            best_war_uar_fuse = uar_fuse
            best_war_cls_acc_fuse = cls_acc_fuse.copy()

            max_war_global = war_global
            max_war_local = war_local
            max_war_fuse = war_fuse

            logger.info(f">>> New best WAR model at epoch {epoch}! WAR: {war_global:.2f}% UAR: {uar_global:.2f}% <<<")

        # Track best UAR model (separate from WAR)
        is_best_uar = uar_global > max_uar_global
        if is_best_uar:
            best_uar_epoch = epoch
            best_uar_global = uar_global
            best_uar_war_global = war_global
            best_uar_cls_acc_global = cls_acc_global.copy()
            best_uar_local = uar_local
            best_uar_war_local = war_local
            best_uar_cls_acc_local = cls_acc_local.copy()
            best_uar_fuse = uar_fuse
            best_uar_war_fuse = war_fuse
            best_uar_cls_acc_fuse = cls_acc_fuse.copy()

            max_uar_global = uar_global
            max_uar_local = uar_local
            max_uar_fuse = uar_fuse

            logger.info(f">>> New best UAR model at epoch {epoch}! UAR: {uar_global:.2f}% WAR: {war_global:.2f}% <<<")

        logger.info(f'Best WAR Epoch: {best_war_epoch} | WAR: {max_war_global:.2f}% | UAR: {best_war_uar_global:.2f}%')
        logger.info(f'Best UAR Epoch: {best_uar_epoch} | UAR: {max_uar_global:.2f}% | WAR: {best_uar_war_global:.2f}%')

        # save model - save both best WAR and best UAR models
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            # Save best WAR model
            if is_best_war:
                epoch_saving(config, epoch, model.module, max_war_global, optimizer, lr_scheduler, logger, config.OUTPUT,
                             True, best_epoch=best_war_epoch,
                             max_war_local=max_war_local, max_war_fuse=max_war_fuse,
                             best_uar_global=best_war_uar_global, best_uar_local=best_war_uar_local, best_uar_fuse=best_war_uar_fuse,
                             checkpoint_name='best_war.pth')

            # Save best UAR model
            if is_best_uar:
                epoch_saving(config, epoch, model.module, max_uar_global, optimizer, lr_scheduler, logger, config.OUTPUT,
                             True, best_epoch=best_uar_epoch,
                             max_war_local=best_uar_war_local, max_war_fuse=best_uar_war_fuse,
                             best_uar_global=best_uar_global, best_uar_local=best_uar_local, best_uar_fuse=best_uar_fuse,
                             checkpoint_name='best_uar.pth')

            # Regular checkpoint save (not best model)
            if not is_best_war and not is_best_uar and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                epoch_saving(config, epoch, model.module, max_war_global, optimizer, lr_scheduler, logger, config.OUTPUT,
                             False, best_epoch=best_war_epoch,
                             max_war_local=max_war_local, max_war_fuse=max_war_fuse,
                             best_uar_global=best_war_uar_global, best_uar_local=best_war_uar_local, best_uar_fuse=best_war_uar_fuse)

    # Print best epoch summary
    logger.info(f"\n" + "=" * 80)
    logger.info(f"TRAINING COMPLETED - BEST MODELS SUMMARY")
    logger.info(f"=" * 80)
    logger.info(f"Best WAR Epoch: {best_war_epoch}")
    logger.info(f"-" * 80)

    # Get emotion class names
    if config.DATA.DATASET.upper() == 'MAFW':
        emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
    else:
        emotion_names = [f'Class_{i}' for i in range(len(best_war_cls_acc_global)) if best_war_cls_acc_global is not None]

    # Print best WAR model results
    if best_war_cls_acc_global is not None:
        logger.info(f"Best WAR Model (Epoch {best_war_epoch}):")
        logger.info(f"  Global Branch: WAR={best_war_global:.2f}%  |  UAR={best_war_uar_global:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, best_war_cls_acc_global):
            logger.info(f"    {name:10s}: {acc*100:.2f}%")

        logger.info(f"-" * 80)
        logger.info(f"  Local Branch: WAR={best_war_local:.2f}%  |  UAR={best_war_uar_local:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, best_war_cls_acc_local):
            logger.info(f"    {name:10s}: {acc*100:.2f}%")

        logger.info(f"-" * 80)
        logger.info(f"  Fuse Branch: WAR={best_war_fuse:.2f}%  |  UAR={best_war_uar_fuse:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, best_war_cls_acc_fuse):
            logger.info(f"    {name:10s}: {acc*100:.2f}%")

    logger.info(f"=" * 80)
    logger.info(f"Best UAR Epoch: {best_uar_epoch}")
    logger.info(f"-" * 80)

    # Print best UAR model results
    if best_uar_cls_acc_global is not None:
        logger.info(f"Best UAR Model (Epoch {best_uar_epoch}):")
        logger.info(f"  Global Branch: UAR={best_uar_global:.2f}%  |  WAR={best_uar_war_global:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, best_uar_cls_acc_global):
            logger.info(f"    {name:10s}: {acc*100:.2f}%")

        logger.info(f"-" * 80)
        logger.info(f"  Local Branch: UAR={best_uar_local:.2f}%  |  WAR={best_uar_war_local:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, best_uar_cls_acc_local):
            logger.info(f"    {name:10s}: {acc*100:.2f}%")

        logger.info(f"-" * 80)
        logger.info(f"  Fuse Branch: UAR={best_uar_fuse:.2f}%  |  WAR={best_uar_war_fuse:.2f}%")
        logger.info(f"  Per-class Accuracy:")
        for name, acc in zip(emotion_names, best_uar_cls_acc_fuse):
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
            if epoch>13:
                total_loss = criterion(output_global, label_id)
                total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
            else:
                # Debug: print actual shape before reshape
                if idx == 0:
                    print(f"DEBUG: output_local.shape = {output_local.shape}, output_local.dim() = {output_local.dim()}")

                # Check if output_local is already 3D [B, T, C] or 2D [B*T, C]
                if output_local.dim() == 3:
                    # Already [B, T, C], use directly
                    pre_output_local = torch.sum(output_local, dim=1).squeeze(dim=-1)
                else:
                    # [B*T, C], need to reshape to [B, T, C]
                    b = config.TRAIN.BATCH_SIZE
                    num_clips = output_local.shape[0] // b
                    pre_output_local = torch.sum(
                        output_local.view(b, num_clips, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)

                # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
                total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
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
                #total_loss = total_loss + config.AU.LAMBDA * au_loss
                total_loss = total_loss * (1 - config.AU.LAMBDA) + config.AU.LAMBDA * au_loss

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            scaler.step(optimizer)
            scaler.update()
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

                # Check if output_local is already 3D [B, T, C] or 2D [B*T, C]
                if output_local.dim() == 3:
                    # Already [B, T, C], use directly
                    pre_output_local = torch.sum(output_local, dim=1).squeeze(dim=-1)
                else:
                    # [B*T, C], need to reshape to [B, T, C]
                    num_clips = output_local.shape[0] // b
                    pre_output_local = torch.sum(output_local.view(b, num_clips, config.DATA.NUM_CLASSES),dim=1).squeeze(dim=-1)

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
