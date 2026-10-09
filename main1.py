# # # # import os
# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.backends.cudnn as cudnn
# # # # import torch.distributed as dist
# # # # import argparse
# # # # import datetime
# # # # import shutil
# # # # import time
# # # # import numpy as np
# # # # import random
# # # # from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# # # # from pathlib import Path
# # # # from sklearn.metrics import confusion_matrix
# # # # from sklearn.manifold import TSNE
# # # # from torch.cuda.amp import autocast
# # # # from torch.cuda.amp import GradScaler
# # # # from utils.optimizer import build_optimizer, build_scheduler
# # # # from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, plot_confusion_matrix
# # # # from utils.logger import create_logger
# # # # from datasets.build import build_dataloader
# # # # from datasets.blending import FixMixupBlending
# # # # from utils.config import get_config
# # # # from models import AU_clip
# # # # import torch.nn.functional as F
# # # # K_TABLE_DFEW = {
# # # #     0: [0.37458275378581574, 0.37673577978763106, 0.41551698634677164, 0.20406588498862172,
# # # #         0.7927884197377446, 0.8938573401735845, 0.26152800062149906, 0.8999413926399066,
# # # #         0.9306407335956202, 0.8244182955003968, 0.2344366715203141, 0.35584905526535116,
# # # #         0.2637127706449251, 0.1980659294341865, 0.8749731256118369, 0.5415473658691405,
# # # #         0.4089852072091214, 0.29454183039387216],
# # # #     1: [0.3563166809248465, 0.22025123080620432, 0.7883122756737404, 0.18552393229679323, 0.4330479071201498, 0.6512463687571445, 0.19562608246203084, 0.57228053429373, 0.3919835645137292, 0.5539327475091499, 0.26436050797140115, 0.3859719011102806, 0.2624203556065227, 0.19099832378533962, 0.5784889747902296, 0.5465085543388307, 0.4089852072091214, 0.31238729511297314],  # sad
# # # #     2: [0.29634994040489937, 0.26249346997682327, 0.5349197805247116, 0.211357886947774, 0.24956316484226185, 0.48303878792569305, 0.1750459267403643, 0.4222786033590969, 0.2595297603319664, 0.4038707629862569, 0.2255822802606922, 0.3300116586908106, 0.21700897117154114, 0.18584396273490125, 0.4406741647399928, 0.4892561535112399, 0.4089852072091214, 0.24616267045793816],  # neutral
# # # #     3: [0.2565426020973117, 0.22590006587965014, 0.7487900008174104, 0.24260938453624706, 0.3297348731445176, 0.6253054547733351, 0.2331741603449138, 0.6002659975024218, 0.25865288585343615, 0.4042681727124247, 0.25714836786909934, 0.4073109215604269, 0.22487756342746285, 0.1940789790636582, 0.7184070119769908, 0.6300831522957303, 0.4089852072091214, 0.22674904330132986],  # angry
# # # #     4: [0.45840639969725916, 0.4348135864186588, 0.5807746998255159, 0.32444326825812997, 0.27370636575780566, 0.49552874169247974, 0.22029419679136286, 0.45675173141757053, 0.30048086399990936, 0.35356016873960344, 0.2340917014067623, 0.35356016873960344, 0.22418778976044848, 0.18774040869030503, 0.6204770513319532, 0.6347324688342945, 0.4089852072091214, 0.2326942881893769],  # surprise
# # # #     5: [0.3669101379226728, 0.23849898584920362, 0.8556853679596857, 0.2093396686826401, 0.47998741884320045, 0.9116481111605065, 0.47437419451562296, 0.8176370658093514, 0.3690013619929339, 0.6122010138157353, 0.37313028939087656, 0.43845703949339593, 0.2712229960450541, 0.22790281312128532, 0.7303843653361992, 0.5736046807175323, 0.4089852072091214, 0.3157383637072392],  # disgust
# # # #     6: [0.4668184913800416, 0.3236830048705496, 0.7092147921039609, 0.3237692334141919, 0.284442322105034, 0.4833791428919572, 0.19268179747966238, 0.49211626418097015, 0.29857493916455136, 0.40402139627383793, 0.3114334189399665, 0.42236998344528387, 0.2762397282204001, 0.20086024967771943, 0.6377967064381651, 0.6643128457350147, 0.4089852072091214, 0.22004117429592432],  # fear
# # # # }
# # # # K_SCALE = 5.0

# # # # POSW_GLOBAL_DFEW = [8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674,
# # # #                    4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794,
# # # #                    7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384,
# # # #                    12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372,
# # # #                    101.28938906752411, 8.483380533611566]

# # # # POSW_DISTINCT_DFEW = {
# # # #     0: [4.648862512363996, 4.162923411588553, 1.83239825175909, 2.5131103421760863, 1.0496164371270917, 1.191954326288771, 15.430099793221252, 0.636697444899202, 1.0033104960263086, 0.7217365088935785, 3.69328950409615, 2.862698681095705, 6.9568094740508535, 4.616744014506562, 2.920370688175734, 5.462006293978289, 57.59313882654697, 4.170958066889254],  # happy
# # # #     1: [5.6823671940967, 4.992658194508461, 0.5149984185907146, 2.6989371862870613, 3.1921004145555045, 1.6714365386873313, 19.07732186190161, 2.1056834274599465, 7.311081685767773, 1.9735191296108734, 5.039584577609662, 3.51384996900186, 8.076543333000897, 6.52494935714581, 3.628397792864953, 5.6926866933852995, 70.26898981989036, 5.224216933388045],  # sad
# # # #     2: [6.4774986002239645, 5.6010812480692, 0.8710279064472912, 1.9167337801498792, 7.8117860530331145, 2.7351547887496284, 21.302160526041124, 3.19245786489297, 20.999073406774425, 2.574808023689626, 5.714757086292502, 3.906024704963953, 8.191199242945629, 5.206488904380156, 4.249791165053314, 7.363091976516634, 81.4053220208253, 4.963300960035722],  # neutral
# # # #     3: [4.928812812224508, 4.114906832298137, 0.9234234234234234, 1.4543961558346765, 4.781224255883091, 2.435997871208089, 12.75189571440743, 1.44547134935305, 13.252185430463577, 3.1313061506565307, 3.579413266753674, 2.578767654819184, 5.92967542503864, 5.292631578947368, 2.6113572291582763, 4.433813627794237, 55.85311729482212, 4.549840112780662],  # angry
# # # #     4: [7.116157728166966, 6.34866790582404, 0.9600900658968373, 0.761682850299846, 17.648977987421382, 5.163029358274876, 21.278938718008924, 6.183435536376713, 35.41059094397544, 6.815336463223788, 6.358355951919348, 4.08091030789826, 9.571078431372548, 7.167857450288371, 5.090243902439024, 8.104394549990404, 94.26706827309236, 6.122504128509233],  # surprise
# # # #     5: [4.229214780600462, 4.09392575928009, 0.7474435655026047, 2.1834797891036906, 5.054144385026738, 1.6915304606240713, 10.307116104868914, 2.2381122631390777, 10.731865284974093, 3.210599721059972, 4.137266023823029, 3.012848914488259, 5.983037779491133, 6.148382004735596, 3.6374807987711213, 5.387165021156559, 27.391849529780565, 3.2964895635673623],  # disgust
# # # #     6: [5.96072648535643, 5.154494891980657, 0.7569813845450734, 1.5509251975236857, 7.060588375159415, 2.779909786630176, 17.84796563052818, 2.8554517069539505, 15.333362533397574, 3.2621352565348083, 4.923954312221004, 3.5370230679384855, 7.644713355124371, 5.762544656620061, 3.9785987023043443, 6.670773851153989, 57.87385538364383, 5.229860670252932],  # fear
# # # # }

# # # # # minor strategy: define which classes are "major"
# # # # MAJOR_CLASSES_DFEW = {0, 1, 2, 3}

# # # # def compute_au_loss_stage_b(au_logits, au_targets, labels, posw_option='global'):
# # # #     """
# # # #     au_logits:  [B,18] float
# # # #     au_targets: [B,18] float {0,1}
# # # #     labels:     [B] long, emotion class ids (0..C-1)
# # # #     """
# # # #     device = au_logits.device
# # # #     B = labels.shape[0]
# # # #     # pos_weight: [B,18]
# # # #     if config.DATA.DATASET=='DFEW':
# # # #         # knowledge weight: [B,18]
# # # #         k = torch.stack(
# # # #             [torch.tensor(K_TABLE_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
# # # #             dim=0
# # # #         ) * K_SCALE
# # # #         if posw_option == 'global':
# # # #             pw = torch.tensor(POSW_GLOBAL_DFEW, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
# # # #         elif posw_option == 'distinct':
# # # #             pw = torch.stack(
# # # #                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
# # # #                 dim=0
# # # #             )
# # # #         elif posw_option == 'minor':
# # # #             pw = torch.stack(
# # # #                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
# # # #                 dim=0
# # # #             )
# # # #             # overwrite major classes with 1s
# # # #             mask_major = torch.tensor([int(c) in MAJOR_CLASSES_DFEW for c in labels], device=device, dtype=torch.bool)
# # # #             if mask_major.any():
# # # #                 pw = pw.clone()
# # # #                 pw[mask_major] = 1.0
# # # #         else:
# # # #             raise ValueError(f'Unknown posw_option: {posw_option}')

# # # #     # weighted BCE with logits
# # # #     return F.binary_cross_entropy_with_logits(
# # # #         au_logits, au_targets,
# # # #         weight=k,
# # # #         pos_weight=pw,
# # # #         reduction='mean'
# # # #     )

# # # # def parse_option():
# # # #     parser = argparse.ArgumentParser()
# # # #     parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/dfew7/16_16.yaml')
# # # #     parser.add_argument(
# # # #         "--opts",
# # # #         help="Modify config options by adding 'KEY VALUE' pairs. ",
# # # #         default=None,
# # # #         nargs='+',
# # # #     )
# # # #     parser.add_argument('--gpu', default=[0, 1], type=int,help='GPU id to use.')
# # # #     parser.add_argument('--output', type=str, default="DFEWAS2")
# # # #     parser.add_argument('--resume', type=str)
# # # #     parser.add_argument('--pretrained', type=str)
# # # #     parser.add_argument('--only_test', action='store_true')
# # # #     parser.add_argument('--batch-size', type=int)
# # # #     parser.add_argument('--accumulation-steps', type=int)
# # # #     parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
# # # #     args = parser.parse_args()
# # # #     config = get_config(args)
# # # #     return args, config


# # # # def main(config):
# # # #     # load train and valid dataset
# # # #     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

# # # #     # load pretrained model
# # # #     model, _ = AU_clip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
# # # #                           device="cpu", jit=False,
# # # #                           T=config.DATA.NUM_FRAMES,
# # # #                           droppath=config.MODEL.DROP_PATH_RATE,
# # # #                           use_checkpoint=config.TRAIN.USE_CHECKPOINT,
# # # #                           use_cache=config.MODEL.FIX_TEXT,
# # # #                           logger=logger,
# # # #                           N=config.DATA.NUM_DIVIDE,
# # # #                           cfg=config
# # # #                           )
# # # #     model = model.cuda()

# # # #     # training data augmentation
# # # #     mixup_fn = None
# # # #     if config.AUG.MIXUP > 0:
# # # #         criterion = SoftTargetCrossEntropy()
# # # #         criterion_soft = SoftTargetCrossEntropy()
# # # #         mixup_fn = FixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
# # # #                                     smoothing=config.AUG.LABEL_SMOOTH,
# # # #                                     mixup_alpha=config.AUG.MIXUP,
# # # #                                     fmix_alpha=config.AUG.CUTMIX,
# # # #                                     switch_prob=config.AUG.MIXUP_SWITCH_PROB)
# # # #     elif config.AUG.LABEL_SMOOTH > 0:
# # # #         criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
# # # #         criterion_soft = SoftTargetCrossEntropy()

# # # #     else:
# # # #         criterion = nn.CrossEntropyLoss()
# # # #         criterion_soft = SoftTargetCrossEntropy()

# # # #     optimizer = build_optimizer(config, model)
# # # #     lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
# # # #     model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
# # # #                                                       find_unused_parameters=True)

# # # #     start_epoch, max_acc_global, max_acc_local, max_acc_fuse = 0, 0.0, 0.0, 0.0
# # # #     # retrain
# # # #     if config.TRAIN.AUTO_RESUME:
# # # #         resume_file_path = auto_resume_helper(config.OUTPUT)
# # # #         if resume_file_path:
# # # #             config.defrost()
# # # #             config.MODEL.RESUME = resume_file_path
# # # #             config.freeze()
# # # #             logger.info(f'auto resuming from {resume_file_path}')
# # # #         else:
# # # #             logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
# # # #     if config.MODEL.RESUME:
# # # #         start_epoch, max_acc_global = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

# # # #     # textual prompt
# # # #     text_labels = generate_text(train_data)

# # # #     # model test
# # # #     if config.TEST.ONLY_TEST:
# # # #         acc1, acc1_local, acc1_fuse = validate(val_loader, text_labels, model, config)
# # # #         logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
# # # #         return
# # # #     print("Start training")
# # # #     #model train and valid
# # # #     for epoch in range(start_epoch, config.TRAIN.EPOCHS):
# # # #         train_loader.sampler.set_epoch(epoch)
# # # #         train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft)

# # # #         # Global, local and fuse classification accuracy
# # # #         acc_global, acc_local, acc_fuse = validate(val_loader, text_labels, model, config)
# # # #         logger.info(
# # # #             f"Accuracy of the network on the {len(val_data)} test videos: {acc_global:.2f}% {acc_local:.2f}% {acc_fuse:.2f}%")
# # # #         is_best = acc_global > max_acc_global
# # # #         max_acc_global = max(max_acc_global, acc_global)
# # # #         max_acc_local = max(max_acc_local, acc_local)
# # # #         max_acc_fuse = max(max_acc_fuse, acc_fuse)
# # # #         logger.info(f'Max accuracy: {max_acc_global:.2f}%  {max_acc_local:.2f}% {max_acc_fuse:.2f}%')
# # # #         # save model
# # # #         if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
# # # #             epoch_saving(config, epoch, model.module, max_acc_global, optimizer, lr_scheduler, logger, config.OUTPUT,
# # # #                          is_best)
# # # #     # validation after training
# # # #     config.defrost()
# # # #     config.TEST.NUM_CLIP = 4
# # # #     config.TEST.NUM_CROP = 3
# # # #     config.freeze()
# # # #     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
# # # #     acc = validate(val_loader, text_labels, model, config)
# # # #     logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc:.1f}%")


# # # # def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft):
# # # #     au_criterion = torch.nn.BCEWithLogitsLoss()
# # # #     model.train()
# # # #     optimizer.zero_grad()

# # # #     num_steps = len(train_loader)
# # # #     batch_time = AverageMeter()
# # # #     tot_loss_meter = AverageMeter()

# # # #     start = time.time()
# # # #     end = time.time()
# # # #     scaler = GradScaler()
# # # #     texts = text_labels.cuda(non_blocking=True)

# # # #     for idx, batch_data in enumerate(train_loader):
# # # #         images = batch_data["imgs"].cuda(non_blocking=True)
# # # #         label_id = batch_data["label"].cuda(non_blocking=True)
# # # #         au_labels = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
# # # #         label_id = label_id.reshape(-1)
# # # #         images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

# # # #         if mixup_fn is not None:
# # # #             images, label_id = mixup_fn(images, label_id)

# # # #         if texts.shape[0] == 1:
# # # #             texts = texts.view(1, -1)
# # # #         with autocast():
# # # #             output_global, output_local, feat, au_logits = model(images, texts)
# # # #             if epoch>13:
# # # #                 total_loss = criterion(output_global, label_id)
# # # #                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
# # # #             else:
# # # #                 pre_output_local = torch.sum(
# # # #                     output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
# # # #                 # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
# # # #                 total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
# # # #                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
# # # #             if config.AU.ENABLED and mixup_fn is None:
# # # #                 au_targets = batch_data["au"].cuda(non_blocking=True).float()
# # # #                 au_loss = compute_au_loss_stage_b(
# # # #                     au_logits, au_targets, label_id,
# # # #                     posw_option=config.AU.POSW_OPTION
# # # #                 )
# # # #                 total_loss = total_loss + config.AU.LAMBDA * au_loss

# # # #         if config.TRAIN.ACCUMULATION_STEPS == 1:
# # # #             optimizer.zero_grad()
# # # #         if config.TRAIN.OPT_LEVEL != 'O0':
# # # #             scaler.scale(total_loss).backward()
# # # #             scaler.step(optimizer)
# # # #             scaler.update()
# # # #         else:
# # # #             total_loss.backward()
# # # #             optimizer.step()

# # # #         if config.TRAIN.ACCUMULATION_STEPS > 1:
# # # #             if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
# # # #                 scaler.step(optimizer)
# # # #                 scaler.update()
# # # #                 optimizer.zero_grad()
# # # #                 lr_scheduler.step_update(epoch * num_steps + idx)
# # # #         else:
# # # #             scaler.step(optimizer)
# # # #             scaler.update()
# # # #             lr_scheduler.step_update(epoch * num_steps + idx)

# # # #         torch.cuda.synchronize()

# # # #         tot_loss_meter.update(total_loss.item(), len(label_id))
# # # #         batch_time.update(time.time() - end)
# # # #         end = time.time()

# # # #         if idx % config.PRINT_FREQ == 0:
# # # #             lr = optimizer.param_groups[0]['lr']
# # # #             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
# # # #             etas = batch_time.avg * (num_steps - idx)
# # # #             logger.info(
# # # #                 f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
# # # #                 f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
# # # #                 f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
# # # #                 f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
# # # #                 f'mem {memory_used:.0f}MB')
# # # #     epoch_time = time.time() - start
# # # #     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# # # # @torch.no_grad()
# # # # def validate(val_loader, text_labels, model, config):
# # # #     model.eval()

# # # #     acc_global_meter, acc_local_meter, acc_fuse_meter = AverageMeter(), AverageMeter(), AverageMeter()

# # # #     probility = []
# # # #     video_pre_global = []
# # # #     video_pre_local = []
# # # #     video_pre_fuse = []
# # # #     video_label = []
# # # #     with torch.no_grad():
# # # #         text_inputs = text_labels.cuda()
# # # #         logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
# # # #         for idx, batch_data in enumerate(val_loader):
# # # #             _image = batch_data["imgs"]
# # # #             label_id = batch_data["label"]
# # # #             label_id = label_id.reshape(-1)

# # # #             b, tn, c, h, w = _image.size()

# # # #             t = config.DATA.NUM_FRAMES
# # # #             n = tn // t
# # # #             _image = _image.view(b, n, t, c, h, w)
# # # #             tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
# # # #             tot_similarity_local = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
# # # #             for i in range(n):
# # # #                 image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
# # # #                 label_id = label_id.cuda(non_blocking=True)
# # # #                 image_input = image.cuda(non_blocking=True)

# # # #                 if config.TRAIN.OPT_LEVEL == 'O2':
# # # #                     image_input = image_input.half()
# # # #                 with autocast():
# # # #                     output, output_local, feat, _ = model(image_input, text_inputs)
# # # #                 if idx < 1:
# # # #                     feature = feat
# # # #                 else:
# # # #                     feature = torch.cat((feature, feat), dim=0)

# # # #                 pre_output_global = output.view(b, -1)
# # # #                 pre_output_local = torch.sum(output_local.view(b, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES),dim=1).squeeze(dim=-1)

# # # #                 similarity = pre_output_global.view(b, -1).softmax(dim=-1)
# # # #                 tot_similarity += similarity

# # # #                 similarity_local = pre_output_local.view(b, -1).softmax(dim=-1)
# # # #                 tot_similarity_local += similarity_local.view(b, -1)

# # # #             probility.extend(tot_similarity.data.cpu().numpy().copy())
# # # #             values_global, indices_global = tot_similarity.topk(1, dim=-1)

# # # #             values_local, indices_local = tot_similarity_local.topk(1, dim=-1)

# # # #             fuse_similarity = tot_similarity + tot_similarity_local
# # # #             values_fuse, indices_fuse = fuse_similarity.topk(1, dim=-1)

# # # #             acc_global = 0
# # # #             acc_local = 0
# # # #             acc_fuse = 0
# # # #             for i in range(b):
# # # #                 video_pre_global.append(indices_global[i].data.cpu().numpy().copy())
# # # #                 video_pre_local.append(indices_local[i].data.cpu().numpy().copy())
# # # #                 video_pre_fuse.append(indices_fuse[i].data.cpu().numpy().copy())
# # # #                 video_label.append(label_id[i].data.cpu().numpy().copy())
# # # #                 if indices_global[i] == label_id[i]:
# # # #                     acc_global += 1
# # # #                 if indices_local[i] == label_id[i]:
# # # #                     acc_local += 1
# # # #                 if indices_fuse[i] == label_id[i]:
# # # #                     acc_fuse += 1

# # # #             acc_global_meter.update(float(acc_global) / b * 100, b)
# # # #             acc_local_meter.update(float(acc_local) / b * 100, b)
# # # #             acc_fuse_meter.update(float(acc_fuse) / b * 100, b)
# # # #             if idx % config.PRINT_FREQ == 0:
# # # #                 logger.info(
# # # #                     f'Test: [{idx}/{len(val_loader)}]\t'
# # # #                     f'Acc@1: {acc_global_meter.avg:.3f}\t'
# # # #                     f'Acc@1: {acc_local_meter.avg:.3f}\t'
# # # #                     f'Acc@1: {acc_fuse_meter.avg:.3f}\t'
# # # #                 )
# # # #     # confusion matrix
# # # #     cf = confusion_matrix(video_label, video_pre_global)
# # # #     np.set_printoptions(precision=4)
# # # #     normalized_cm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
# # # #     normalized_cm = normalized_cm * 100

# # # #     cls_cnt = normalized_cm.sum(axis=1)
# # # #     cls_hit = np.diag(normalized_cm)
# # # #     # print(cf)
# # # #     cls_acc = cls_hit / cls_cnt
# # # #     cls_acc = np.around(cls_acc, 4)
# # # #     cm = np.array(normalized_cm)
# # # #     # save_path = 'AU-CLIP/results'
# # # #     # if not os.path.exists(save_path):
# # # #     #     os.makedirs(save_path)
# # # #     # labels_name = ['hap', 'sad', 'neu', 'ang', 'sur', 'dis', 'fea']
# # # #     # plot_confusion_matrix(cm, labels_name, 'AUCLIP', cls_acc)
# # # #     #
# # # #     # #t-SNE
# # # #     # col = ['orange', 'purple', 'g', 'r', 'darkblue', 'chocolate', 'c']
# # # #     # x_embed = TSNE(n_components=2, perplexity=100, n_iter=10000).fit_transform(feature.data.cpu())
# # # #     # label = np.array(video_label)
# # # #     # plt.figure(figsize=(6, 6))
# # # #     # for i in range(7):
# # # #     #     idxs = np.where(label == i)[0]
# # # #     #     plt.scatter(x_embed[idxs, 0], x_embed[idxs, 1], color=col[i], s=6, label=labels_name[i])
# # # #     # plt.legend(loc='upper left')
# # # #     # plt.xticks(fontsize=13)
# # # #     # plt.yticks(fontsize=13)
# # # #     # plt.savefig(os.path.join(save_path, 'AUCLIP_TSNE.jpg'), format='jpg')
# # # #     # plt.show()

# # # #     print(cls_acc)
# # # #     upper = np.mean(np.max(cf, axis=1) / cls_cnt)
# # # #     print('upper bound: {}'.format(upper))

# # # #     print('-----Evaluation is finished------')
# # # #     print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

# # # #     cf_local = confusion_matrix(video_label, video_pre_local).astype(float)
# # # #     cls_cnt_local = cf_local.sum(axis=1)
# # # #     cls_hit_local = np.diag(cf_local)
# # # #     # print(cf)
# # # #     cls_acc_local = cls_hit_local / cls_cnt_local
# # # #     cls_acc_local = np.around(cls_acc_local, 4)
# # # #     print(cls_acc_local)
# # # #     upper = np.mean(np.max(cf_local, axis=1) / cls_cnt_local)
# # # #     print('upper bound: {}'.format(upper))

# # # #     print('-----Evaluation is finished------')
# # # #     print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc_local) * 100))

# # # #     cf_fuse = confusion_matrix(video_label, video_pre_fuse).astype(float)
# # # #     cls_cnt_fuse = cf_fuse.sum(axis=1)
# # # #     cls_hit_fuse = np.diag(cf_fuse)
# # # #     # print(cf)
# # # #     cls_acc_fuse = cls_hit_fuse / cls_cnt_fuse
# # # #     cls_acc_fuse = np.around(cls_acc_fuse, 4)
# # # #     print(cls_acc_fuse)
# # # #     upper = np.mean(np.max(cf_fuse, axis=1) / cls_cnt_fuse)
# # # #     print('upper bound: {}'.format(upper))

# # # #     print('-----Evaluation is finished------')
# # # #     print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc_fuse) * 100))

# # # #     acc_global_meter.sync()
# # # #     acc_local_meter.sync()
# # # #     acc_fuse_meter.sync()
# # # #     logger.info(f' * Acc@1 {acc_global_meter.avg:.3f} Acc_loca@1 {acc_local_meter.avg:.3f}  Acc_fuse@1 {acc_fuse_meter.avg:.3f}')

# # # #     return acc_global_meter.avg, acc_local_meter.avg, acc_fuse_meter.avg


# # # # if __name__ == '__main__':
# # # #     args, config = parse_option()

# # # #     # 初始化分布式环境
# # # #     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
# # # #         rank = int(os.environ["RANK"])
# # # #         world_size = int(os.environ['WORLD_SIZE'])
# # # #         local_rank = int(os.environ['LOCAL_RANK'])  # 使用环境变量中的 LOCAL_RANK
# # # #         print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
# # # #     else:
# # # #         rank = -1
# # # #         world_size = -1
# # # #         local_rank = args.local_rank  # 如果未设置环境变量，则使用命令行参数

# # # #     # 设置当前 GPU 设备
# # # #     torch.cuda.set_device(local_rank)

# # # #     # 初始化分布式进程组
# # # #     dist.init_process_group(
# # # #         backend='nccl',
# # # #         init_method='env://',
# # # #         world_size=world_size,
# # # #         rank=rank
# # # #     )

# # # #     # 确保所有进程同步
# # # #     dist.barrier()

# # # #     # 设置随机种子
# # # #     seed = config.SEED + dist.get_rank()
# # # #     torch.manual_seed(seed)
# # # #     np.random.seed(seed)
# # # #     random.seed(seed)
# # # #     cudnn.benchmark = True

# # # #     # 创建输出目录
# # # #     output_dir = Path(config.OUTPUT)
# # # #     output_dir.mkdir(parents=True, exist_ok=True)

# # # #     # 创建日志记录器
# # # #     logger = create_logger(output_dir=output_dir, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
# # # #     logger.info(f"Working directory: {output_dir}")

# # # #     # 保存配置文件（仅主进程执行）
# # # #     if dist.get_rank() == 0:
# # # #         logger.info(config)
# # # #         shutil.copy(args.config, output_dir)

# # # #     # 启动主训练逻辑
# # # #     main(config)



# # # import os
# # # import torch
# # # import torch.nn as nn
# # # import torch.backends.cudnn as cudnn
# # # import torch.distributed as dist
# # # import argparse
# # # import datetime
# # # import shutil
# # # import time
# # # import numpy as np
# # # import random
# # # from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# # # from pathlib import Path
# # # from sklearn.metrics import confusion_matrix
# # # from sklearn.manifold import TSNE
# # # from torch.cuda.amp import autocast
# # # from torch.cuda.amp import GradScaler
# # # from utils.optimizer import build_optimizer, build_scheduler
# # # from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, plot_confusion_matrix
# # # from utils.logger import create_logger
# # # from datasets.build import build_dataloader
# # # from datasets.blending import FixMixupBlending
# # # from utils.config import get_config
# # # from models import AU_clip
# # # import torch.nn.functional as F
# # # K_TABLE_DFEW = {
# # #     0: [0.37458275378581574, 0.37673577978763106, 0.41551698634677164, 0.20406588498862172,
# # #         0.7927884197377446, 0.8938573401735845, 0.26152800062149906, 0.8999413926399066,
# # #         0.9306407335956202, 0.8244182955003968, 0.2344366715203141, 0.35584905526535116,
# # #         0.2637127706449251, 0.1980659294341865, 0.8749731256118369, 0.5415473658691405,
# # #         0.4089852072091214, 0.29454183039387216],
# # #     1: [0.3563166809248465, 0.22025123080620432, 0.7883122756737404, 0.18552393229679323, 0.4330479071201498, 0.6512463687571445, 0.19562608246203084, 0.57228053429373, 0.3919835645137292, 0.5539327475091499, 0.26436050797140115, 0.3859719011102806, 0.2624203556065227, 0.19099832378533962, 0.5784889747902296, 0.5465085543388307, 0.4089852072091214, 0.31238729511297314],  # sad
# # #     2: [0.29634994040489937, 0.26249346997682327, 0.5349197805247116, 0.211357886947774, 0.24956316484226185, 0.48303878792569305, 0.1750459267403643, 0.4222786033590969, 0.2595297603319664, 0.4038707629862569, 0.2255822802606922, 0.3300116586908106, 0.21700897117154114, 0.18584396273490125, 0.4406741647399928, 0.4892561535112399, 0.4089852072091214, 0.24616267045793816],  # neutral
# # #     3: [0.2565426020973117, 0.22590006587965014, 0.7487900008174104, 0.24260938453624706, 0.3297348731445176, 0.6253054547733351, 0.2331741603449138, 0.6002659975024218, 0.25865288585343615, 0.4042681727124247, 0.25714836786909934, 0.4073109215604269, 0.22487756342746285, 0.1940789790636582, 0.7184070119769908, 0.6300831522957303, 0.4089852072091214, 0.22674904330132986],  # angry
# # #     4: [0.45840639969725916, 0.4348135864186588, 0.5807746998255159, 0.32444326825812997, 0.27370636575780566, 0.49552874169247974, 0.22029419679136286, 0.45675173141757053, 0.30048086399990936, 0.35356016873960344, 0.2340917014067623, 0.35356016873960344, 0.22418778976044848, 0.18774040869030503, 0.6204770513319532, 0.6347324688342945, 0.4089852072091214, 0.2326942881893769],  # surprise
# # #     5: [0.3669101379226728, 0.23849898584920362, 0.8556853679596857, 0.2093396686826401, 0.47998741884320045, 0.9116481111605065, 0.47437419451562296, 0.8176370658093514, 0.3690013619929339, 0.6122010138157353, 0.37313028939087656, 0.43845703949339593, 0.2712229960450541, 0.22790281312128532, 0.7303843653361992, 0.5736046807175323, 0.4089852072091214, 0.3157383637072392],  # disgust
# # #     6: [0.4668184913800416, 0.3236830048705496, 0.7092147921039609, 0.3237692334141919, 0.284442322105034, 0.4833791428919572, 0.19268179747966238, 0.49211626418097015, 0.29857493916455136, 0.40402139627383793, 0.3114334189399665, 0.42236998344528387, 0.2762397282204001, 0.20086024967771943, 0.6377967064381651, 0.6643128457350147, 0.4089852072091214, 0.22004117429592432],  # fear
# # # }
# # # K_SCALE = 5.0

# # # POSW_GLOBAL_DFEW = [8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674,
# # #                    4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794,
# # #                    7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384,
# # #                    12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372,
# # #                    101.28938906752411, 8.483380533611566]

# # # POSW_DISTINCT_DFEW = {
# # #     0: [4.648862512363996, 4.162923411588553, 1.83239825175909, 2.5131103421760863, 1.0496164371270917, 1.191954326288771, 15.430099793221252, 0.636697444899202, 1.0033104960263086, 0.7217365088935785, 3.69328950409615, 2.862698681095705, 6.9568094740508535, 4.616744014506562, 2.920370688175734, 5.462006293978289, 57.59313882654697, 4.170958066889254],  # happy
# # #     1: [5.6823671940967, 4.992658194508461, 0.5149984185907146, 2.6989371862870613, 3.1921004145555045, 1.6714365386873313, 19.07732186190161, 2.1056834274599465, 7.311081685767773, 1.9735191296108734, 5.039584577609662, 3.51384996900186, 8.076543333000897, 6.52494935714581, 3.628397792864953, 5.6926866933852995, 70.26898981989036, 5.224216933388045],  # sad
# # #     2: [6.4774986002239645, 5.6010812480692, 0.8710279064472912, 1.9167337801498792, 7.8117860530331145, 2.7351547887496284, 21.302160526041124, 3.19245786489297, 20.999073406774425, 2.574808023689626, 5.714757086292502, 3.906024704963953, 8.191199242945629, 5.206488904380156, 4.249791165053314, 7.363091976516634, 81.4053220208253, 4.963300960035722],  # neutral
# # #     3: [4.928812812224508, 4.114906832298137, 0.9234234234234234, 1.4543961558346765, 4.781224255883091, 2.435997871208089, 12.75189571440743, 1.44547134935305, 13.252185430463577, 3.1313061506565307, 3.579413266753674, 2.578767654819184, 5.92967542503864, 5.292631578947368, 2.6113572291582763, 4.433813627794237, 55.85311729482212, 4.549840112780662],  # angry
# # #     4: [7.116157728166966, 6.34866790582404, 0.9600900658968373, 0.761682850299846, 17.648977987421382, 5.163029358274876, 21.278938718008924, 6.183435536376713, 35.41059094397544, 6.815336463223788, 6.358355951919348, 4.08091030789826, 9.571078431372548, 7.167857450288371, 5.090243902439024, 8.104394549990404, 94.26706827309236, 6.122504128509233],  # surprise
# # #     5: [4.229214780600462, 4.09392575928009, 0.7474435655026047, 2.1834797891036906, 5.054144385026738, 1.6915304606240713, 10.307116104868914, 2.2381122631390777, 10.731865284974093, 3.210599721059972, 4.137266023823029, 3.012848914488259, 5.983037779491133, 6.148382004735596, 3.6374807987711213, 5.387165021156559, 27.391849529780565, 3.2964895635673623],  # disgust
# # #     6: [5.96072648535643, 5.154494891980657, 0.7569813845450734, 1.5509251975236857, 7.060588375159415, 2.779909786630176, 17.84796563052818, 2.8554517069539505, 15.333362533397574, 3.2621352565348083, 4.923954312221004, 3.5370230679384855, 7.644713355124371, 5.762544656620061, 3.9785987023043443, 6.670773851153989, 57.87385538364383, 5.229860670252932],  # fear
# # # }

# # # # minor strategy: define which classes are "major"
# # # MAJOR_CLASSES_DFEW = {0, 1, 2, 3}

# # # def compute_au_loss_stage_b(au_logits, au_targets, labels, posw_option='global'):
# # #     """
# # #     au_logits:  [B,18] float
# # #     au_targets: [B,18] float {0,1}
# # #     labels:     [B] long, emotion class ids (0..C-1)
# # #     """
# # #     device = au_logits.device
# # #     B = labels.shape[0]
# # #     # pos_weight: [B,18]
# # #     if config.DATA.DATASET=='DFEW':
# # #         # knowledge weight: [B,18]
# # #         k = torch.stack(
# # #             [torch.tensor(K_TABLE_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
# # #             dim=0
# # #         ) * K_SCALE
# # #         if posw_option == 'global':
# # #             pw = torch.tensor(POSW_GLOBAL_DFEW, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
# # #         elif posw_option == 'distinct':
# # #             pw = torch.stack(
# # #                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
# # #                 dim=0
# # #             )
# # #         elif posw_option == 'minor':
# # #             pw = torch.stack(
# # #                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
# # #                 dim=0
# # #             )
# # #             # overwrite major classes with 1s
# # #             mask_major = torch.tensor([int(c) in MAJOR_CLASSES_DFEW for c in labels], device=device, dtype=torch.bool)
# # #             if mask_major.any():
# # #                 pw = pw.clone()
# # #                 pw[mask_major] = 1.0
# # #         else:
# # #             raise ValueError(f'Unknown posw_option: {posw_option}')

# # #     # weighted BCE with logits
# # #     return F.binary_cross_entropy_with_logits(
# # #         au_logits, au_targets,
# # #         weight=k,
# # #         pos_weight=pw,
# # #         reduction='mean'
# # #     )

# # # def parse_option():
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/dfew7/16_16.yaml')
# # #     parser.add_argument(
# # #         "--opts",
# # #         help="Modify config options by adding 'KEY VALUE' pairs. ",
# # #         default=None,
# # #         nargs='+',
# # #     )
# # #     parser.add_argument('--gpu', default=[0, 1], type=int,help='GPU id to use.')
# # #     parser.add_argument('--output', type=str, default="DFEWAS2")
# # #     parser.add_argument('--resume', type=str)
# # #     parser.add_argument('--pretrained', type=str)
# # #     parser.add_argument('--only_test', action='store_true')
# # #     parser.add_argument('--batch-size', type=int)
# # #     parser.add_argument('--accumulation-steps', type=int)
# # #     parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
# # #     args = parser.parse_args()
# # #     config = get_config(args)
# # #     return args, config


# # # def main(config):
# # #     # load train and valid dataset
# # #     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

# # #     # load pretrained model
# # #     model, _ = AU_clip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
# # #                           device="cpu", jit=False,
# # #                           T=config.DATA.NUM_FRAMES,
# # #                           droppath=config.MODEL.DROP_PATH_RATE,
# # #                           use_checkpoint=config.TRAIN.USE_CHECKPOINT,
# # #                           use_cache=config.MODEL.FIX_TEXT,
# # #                           logger=logger,
# # #                           N=config.DATA.NUM_DIVIDE,
# # #                           cfg=config
# # #                           )
# # #     model = model.cuda()

# # #     # training data augmentation
# # #     mixup_fn = None
# # #     if config.AUG.MIXUP > 0:
# # #         criterion = SoftTargetCrossEntropy()
# # #         criterion_soft = SoftTargetCrossEntropy()
# # #         mixup_fn = FixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
# # #                                     smoothing=config.AUG.LABEL_SMOOTH,
# # #                                     mixup_alpha=config.AUG.MIXUP,
# # #                                     fmix_alpha=config.AUG.CUTMIX,
# # #                                     switch_prob=config.AUG.MIXUP_SWITCH_PROB)
# # #     elif config.AUG.LABEL_SMOOTH > 0:
# # #         criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
# # #         criterion_soft = SoftTargetCrossEntropy()

# # #     else:
# # #         criterion = nn.CrossEntropyLoss()
# # #         criterion_soft = SoftTargetCrossEntropy()

# # #     optimizer = build_optimizer(config, model)
# # #     lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
# # #     model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
# # #                                                       find_unused_parameters=True)

# # #     start_epoch, max_acc_global, max_acc_local, max_acc_fuse = 0, 0.0, 0.0, 0.0
# # #     max_uar_global, max_uar_local, max_uar_fuse = 0.0, 0.0, 0.0
# # #     # retrain
# # #     if config.TRAIN.AUTO_RESUME:
# # #         resume_file_path = auto_resume_helper(config.OUTPUT)
# # #         if resume_file_path:
# # #             config.defrost()
# # #             config.MODEL.RESUME = resume_file_path
# # #             config.freeze()
# # #             logger.info(f'auto resuming from {resume_file_path}')
# # #         else:
# # #             logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
# # #     if config.MODEL.RESUME:
# # #         start_epoch, max_acc_global = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

# # #     # textual prompt
# # #     text_labels = generate_text(train_data)

# # #     # model test
# # #     if config.TEST.ONLY_TEST:
# # #         results = validate(val_loader, text_labels, model, config)
# # #         acc1, acc1_local, acc1_fuse = results[0], results[1], results[2]
# # #         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
# # #         logger.info(f"Accuracy of the network on the {len(val_data)} test videos: WAR={acc1:.1f}% UAR={uar_global:.1f}%")
# # #         return
# # #     print("Start training")
# # #     #model train and valid
# # #     for epoch in range(start_epoch, config.TRAIN.EPOCHS):
# # #         train_loader.sampler.set_epoch(epoch)
# # #         train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft)

# # #         # Global, local and fuse classification accuracy
# # #         results = validate(val_loader, text_labels, model, config)
# # #         acc_global, acc_local, acc_fuse = results[0], results[1], results[2]
# # #         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
# # #         war_global, war_local, war_fuse = results[6], results[7], results[8]

# # #         logger.info(
# # #             f"Accuracy of the network on the {len(val_data)} test videos:\n"
# # #             f"  WAR: {acc_global:.2f}% {acc_local:.2f}% {acc_fuse:.2f}%\n"
# # #             f"  UAR: {uar_global:.2f}% {uar_local:.2f}% {uar_fuse:.2f}%")

# # #         is_best = uar_global > max_uar_global  # Use UAR as the criterion for best model
# # #         max_acc_global = max(max_acc_global, acc_global)
# # #         max_acc_local = max(max_acc_local, acc_local)
# # #         max_acc_fuse = max(max_acc_fuse, acc_fuse)
# # #         max_uar_global = max(max_uar_global, uar_global)
# # #         max_uar_local = max(max_uar_local, uar_local)
# # #         max_uar_fuse = max(max_uar_fuse, uar_fuse)

# # #         logger.info(f'Max WAR: {max_acc_global:.2f}% {max_acc_local:.2f}% {max_acc_fuse:.2f}%')
# # #         logger.info(f'Max UAR: {max_uar_global:.2f}% {max_uar_local:.2f}% {max_uar_fuse:.2f}%')
# # #         # save model
# # #         if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
# # #             epoch_saving(config, epoch, model.module, max_acc_global, optimizer, lr_scheduler, logger, config.OUTPUT,
# # #                          is_best)
# # #     # validation after training
# # #     logger.info("=" * 50)
# # #     logger.info("Training completed. Loading best model for final evaluation...")

# # #     # Load best model
# # #     best_model_path = os.path.join(config.OUTPUT, 'best.pth')
# # #     if os.path.exists(best_model_path):
# # #         logger.info(f"Loading best model from {best_model_path}")
# # #         checkpoint = torch.load(best_model_path, map_location='cpu')
# # #         model.module.load_state_dict(checkpoint['model'])
# # #         logger.info(f"Best model loaded successfully (epoch {checkpoint['epoch']})")
# # #     else:
# # #         logger.info("Best model checkpoint not found, using current model")

# # #     config.defrost()
# # #     config.TEST.NUM_CLIP = 4
# # #     config.TEST.NUM_CROP = 3
# # #     config.freeze()
# # #     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
# # #     results = validate(val_loader, text_labels, model, config)
# # #     acc = results[0]
# # #     uar_global = results[3]
# # #     logger.info(f"Final Accuracy: WAR={acc:.2f}% UAR={uar_global:.2f}%")


# # # def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft):
# # #     au_criterion = torch.nn.BCEWithLogitsLoss()
# # #     model.train()
# # #     optimizer.zero_grad()

# # #     num_steps = len(train_loader)
# # #     batch_time = AverageMeter()
# # #     tot_loss_meter = AverageMeter()

# # #     start = time.time()
# # #     end = time.time()
# # #     scaler = GradScaler()
# # #     texts = text_labels.cuda(non_blocking=True)

# # #     for idx, batch_data in enumerate(train_loader):
# # #         images = batch_data["imgs"].cuda(non_blocking=True)
# # #         label_id = batch_data["label"].cuda(non_blocking=True)
# # #         au_labels = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
# # #         label_id = label_id.reshape(-1)
# # #         images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

# # #         if mixup_fn is not None:
# # #             images, label_id = mixup_fn(images, label_id)

# # #         if texts.shape[0] == 1:
# # #             texts = texts.view(1, -1)
# # #         with autocast():
# # #             output_global, output_local, feat, au_logits = model(images, texts)
# # #             if epoch>13:
# # #                 total_loss = criterion(output_global, label_id)
# # #                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
# # #             else:
# # #                 pre_output_local = torch.sum(
# # #                     output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
# # #                 # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
# # #                 total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
# # #                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
# # #             if config.AU.ENABLED and mixup_fn is None:
# # #                 au_targets = batch_data["au"].cuda(non_blocking=True).float()
# # #                 au_loss = compute_au_loss_stage_b(
# # #                     au_logits, au_targets, label_id,
# # #                     posw_option=config.AU.POSW_OPTION
# # #                 )
# # #                 total_loss = total_loss + config.AU.LAMBDA * au_loss
# # #                 # total_loss = (1-config.AU.LAMBDA) * total_loss + config.AU.LAMBDA * au_loss

# # #         if config.TRAIN.ACCUMULATION_STEPS == 1:
# # #             optimizer.zero_grad()
# # #         if config.TRAIN.OPT_LEVEL != 'O0':
# # #             scaler.scale(total_loss).backward()
# # #             scaler.step(optimizer)
# # #             scaler.update()
# # #         else:
# # #             total_loss.backward()
# # #             optimizer.step()

# # #         if config.TRAIN.ACCUMULATION_STEPS > 1:
# # #             if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
# # #                 scaler.step(optimizer)
# # #                 scaler.update()
# # #                 optimizer.zero_grad()
# # #                 lr_scheduler.step_update(epoch * num_steps + idx)
# # #         else:
# # #             scaler.step(optimizer)
# # #             scaler.update()
# # #             lr_scheduler.step_update(epoch * num_steps + idx)

# # #         torch.cuda.synchronize()

# # #         tot_loss_meter.update(total_loss.item(), len(label_id))
# # #         batch_time.update(time.time() - end)
# # #         end = time.time()

# # #         if idx % config.PRINT_FREQ == 0:
# # #             lr = optimizer.param_groups[0]['lr']
# # #             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
# # #             etas = batch_time.avg * (num_steps - idx)
# # #             logger.info(
# # #                 f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
# # #                 f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
# # #                 f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
# # #                 f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
# # #                 f'mem {memory_used:.0f}MB')
# # #     epoch_time = time.time() - start
# # #     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# # # @torch.no_grad()
# # # def validate(val_loader, text_labels, model, config):
# # #     model.eval()

# # #     acc_global_meter, acc_local_meter, acc_fuse_meter = AverageMeter(), AverageMeter(), AverageMeter()

# # #     probility = []
# # #     video_pre_global = []
# # #     video_pre_local = []
# # #     video_pre_fuse = []
# # #     video_label = []
# # #     with torch.no_grad():
# # #         text_inputs = text_labels.cuda()
# # #         logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
# # #         for idx, batch_data in enumerate(val_loader):
# # #             _image = batch_data["imgs"]
# # #             label_id = batch_data["label"]
# # #             label_id = label_id.reshape(-1)

# # #             b, tn, c, h, w = _image.size()

# # #             t = config.DATA.NUM_FRAMES
# # #             n = tn // t
# # #             _image = _image.view(b, n, t, c, h, w)
# # #             tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
# # #             tot_similarity_local = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
# # #             for i in range(n):
# # #                 image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
# # #                 label_id = label_id.cuda(non_blocking=True)
# # #                 image_input = image.cuda(non_blocking=True)

# # #                 if config.TRAIN.OPT_LEVEL == 'O2':
# # #                     image_input = image_input.half()
# # #                 with autocast():
# # #                     output, output_local, feat, _ = model(image_input, text_inputs)
# # #                 if idx < 1:
# # #                     feature = feat
# # #                 else:
# # #                     feature = torch.cat((feature, feat), dim=0)

# # #                 pre_output_global = output.view(b, -1)
# # #                 pre_output_local = torch.sum(output_local.view(b, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES),dim=1).squeeze(dim=-1)

# # #                 similarity = pre_output_global.view(b, -1).softmax(dim=-1)
# # #                 tot_similarity += similarity

# # #                 similarity_local = pre_output_local.view(b, -1).softmax(dim=-1)
# # #                 tot_similarity_local += similarity_local.view(b, -1)

# # #             probility.extend(tot_similarity.data.cpu().numpy().copy())
# # #             values_global, indices_global = tot_similarity.topk(1, dim=-1)

# # #             values_local, indices_local = tot_similarity_local.topk(1, dim=-1)

# # #             fuse_similarity = tot_similarity + tot_similarity_local
# # #             values_fuse, indices_fuse = fuse_similarity.topk(1, dim=-1)

# # #             acc_global = 0
# # #             acc_local = 0
# # #             acc_fuse = 0
# # #             for i in range(b):
# # #                 video_pre_global.append(indices_global[i].data.cpu().numpy().copy())
# # #                 video_pre_local.append(indices_local[i].data.cpu().numpy().copy())
# # #                 video_pre_fuse.append(indices_fuse[i].data.cpu().numpy().copy())
# # #                 video_label.append(label_id[i].data.cpu().numpy().copy())
# # #                 if indices_global[i] == label_id[i]:
# # #                     acc_global += 1
# # #                 if indices_local[i] == label_id[i]:
# # #                     acc_local += 1
# # #                 if indices_fuse[i] == label_id[i]:
# # #                     acc_fuse += 1

# # #             acc_global_meter.update(float(acc_global) / b * 100, b)
# # #             acc_local_meter.update(float(acc_local) / b * 100, b)
# # #             acc_fuse_meter.update(float(acc_fuse) / b * 100, b)
# # #             if idx % config.PRINT_FREQ == 0:
# # #                 logger.info(
# # #                     f'Test: [{idx}/{len(val_loader)}]\t'
# # #                     f'Acc@1: {acc_global_meter.avg:.3f}\t'
# # #                     f'Acc@1: {acc_local_meter.avg:.3f}\t'
# # #                     f'Acc@1: {acc_fuse_meter.avg:.3f}\t'
# # #                 )
# # #     # confusion matrix
# # #     cf = confusion_matrix(video_label, video_pre_global)
# # #     np.set_printoptions(precision=4)
# # #     normalized_cm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
# # #     normalized_cm = normalized_cm * 100

# # #     cls_cnt = normalized_cm.sum(axis=1)
# # #     cls_hit = np.diag(normalized_cm)
# # #     # print(cf)
# # #     cls_acc = cls_hit / cls_cnt
# # #     cls_acc = np.around(cls_acc, 4)
# # #     cm = np.array(normalized_cm)
# # #     # save_path = 'AU-CLIP/results'
# # #     # if not os.path.exists(save_path):
# # #     #     os.makedirs(save_path)
# # #     # labels_name = ['hap', 'sad', 'neu', 'ang', 'sur', 'dis', 'fea']
# # #     # plot_confusion_matrix(cm, labels_name, 'AUCLIP', cls_acc)
# # #     #
# # #     # #t-SNE
# # #     # col = ['orange', 'purple', 'g', 'r', 'darkblue', 'chocolate', 'c']
# # #     # x_embed = TSNE(n_components=2, perplexity=100, n_iter=10000).fit_transform(feature.data.cpu())
# # #     # label = np.array(video_label)
# # #     # plt.figure(figsize=(6, 6))
# # #     # for i in range(7):
# # #     #     idxs = np.where(label == i)[0]
# # #     #     plt.scatter(x_embed[idxs, 0], x_embed[idxs, 1], color=col[i], s=6, label=labels_name[i])
# # #     # plt.legend(loc='upper left')
# # #     # plt.xticks(fontsize=13)
# # #     # plt.yticks(fontsize=13)
# # #     # plt.savefig(os.path.join(save_path, 'AUCLIP_TSNE.jpg'), format='jpg')
# # #     # plt.show()

# # #     logger.info(f'Global - Class-wise Accuracy: {cls_acc}')
# # #     upper = np.mean(np.max(cf, axis=1) / cls_cnt)
# # #     logger.info(f'Global - Upper bound: {upper}')
# # #     logger.info('Global - Evaluation is finished')
# # #     logger.info(f'Global - Class Accuracy (UAR): {np.mean(cls_acc) * 100:.2f}%')

# # #     cf_local = confusion_matrix(video_label, video_pre_local).astype(float)
# # #     cls_cnt_local = cf_local.sum(axis=1)
# # #     cls_hit_local = np.diag(cf_local)
# # #     # print(cf)
# # #     cls_acc_local = cls_hit_local / cls_cnt_local
# # #     cls_acc_local = np.around(cls_acc_local, 4)
# # #     logger.info(f'Local - Class-wise Accuracy: {cls_acc_local}')
# # #     upper = np.mean(np.max(cf_local, axis=1) / cls_cnt_local)
# # #     logger.info(f'Local - Upper bound: {upper}')
# # #     logger.info('Local - Evaluation is finished')
# # #     logger.info(f'Local - Class Accuracy (UAR): {np.mean(cls_acc_local) * 100:.2f}%')

# # #     cf_fuse = confusion_matrix(video_label, video_pre_fuse).astype(float)
# # #     cls_cnt_fuse = cf_fuse.sum(axis=1)
# # #     cls_hit_fuse = np.diag(cf_fuse)
# # #     # print(cf)
# # #     cls_acc_fuse = cls_hit_fuse / cls_cnt_fuse
# # #     cls_acc_fuse = np.around(cls_acc_fuse, 4)
# # #     logger.info(f'Fuse - Class-wise Accuracy: {cls_acc_fuse}')
# # #     upper = np.mean(np.max(cf_fuse, axis=1) / cls_cnt_fuse)
# # #     logger.info(f'Fuse - Upper bound: {upper}')
# # #     logger.info('Fuse - Evaluation is finished')
# # #     logger.info(f'Fuse - Class Accuracy (UAR): {np.mean(cls_acc_fuse) * 100:.2f}%')

# # #     acc_global_meter.sync()
# # #     acc_local_meter.sync()
# # #     acc_fuse_meter.sync()
# # #     logger.info(f' * Acc@1 {acc_global_meter.avg:.3f} Acc_loca@1 {acc_local_meter.avg:.3f}  Acc_fuse@1 {acc_fuse_meter.avg:.3f}')

# # #     # Calculate UAR (Unweighted Average Recall) and WAR (Weighted Average Recall)
# # #     uar_global = np.mean(cls_acc) * 100  # UAR for global
# # #     uar_local = np.mean(cls_acc_local) * 100  # UAR for local
# # #     uar_fuse = np.mean(cls_acc_fuse) * 100  # UAR for fuse
# # #     war_global = acc_global_meter.avg  # WAR is the same as overall accuracy
# # #     war_local = acc_local_meter.avg
# # #     war_fuse = acc_fuse_meter.avg

# # #     return (acc_global_meter.avg, acc_local_meter.avg, acc_fuse_meter.avg,
# # #             uar_global, uar_local, uar_fuse, war_global, war_local, war_fuse,
# # #             video_label, video_pre_global, video_pre_local, video_pre_fuse)


# # # if __name__ == '__main__':
# # #     args, config = parse_option()

# # #     # 初始化分布式环境
# # #     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
# # #         rank = int(os.environ["RANK"])
# # #         world_size = int(os.environ['WORLD_SIZE'])
# # #         local_rank = int(os.environ['LOCAL_RANK'])  # 使用环境变量中的 LOCAL_RANK
# # #         print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
# # #     else:
# # #         rank = -1
# # #         world_size = -1
# # #         local_rank = args.local_rank  # 如果未设置环境变量，则使用命令行参数

# # #     # 设置当前 GPU 设备
# # #     torch.cuda.set_device(local_rank)

# # #     # 初始化分布式进程组
# # #     dist.init_process_group(
# # #         backend='nccl',
# # #         init_method='env://',
# # #         world_size=world_size,
# # #         rank=rank
# # #     )

# # #     # 确保所有进程同步
# # #     dist.barrier()

# # #     # 设置随机种子
# # #     seed = config.SEED + dist.get_rank()
# # #     torch.manual_seed(seed)
# # #     np.random.seed(seed)
# # #     random.seed(seed)
# # #     cudnn.benchmark = True

# # #     # 创建输出目录
# # #     output_dir = Path(config.OUTPUT)
# # #     output_dir.mkdir(parents=True, exist_ok=True)

# # #     # 创建日志记录器
# # #     logger = create_logger(output_dir=output_dir, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
# # #     logger.info(f"Working directory: {output_dir}")

# # #     # 保存配置文件（仅主进程执行）
# # #     if dist.get_rank() == 0:
# # #         logger.info(config)
# # #         shutil.copy(args.config, output_dir)

# # #     # 启动主训练逻辑
# # #     main(config)


# # import os
# # import torch
# # import torch.nn as nn
# # import torch.backends.cudnn as cudnn
# # import torch.distributed as dist
# # import argparse
# # import datetime
# # import shutil
# # import time
# # import numpy as np
# # import random
# # from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# # from pathlib import Path
# # from sklearn.metrics import confusion_matrix
# # from sklearn.manifold import TSNE
# # from torch.cuda.amp import autocast
# # from torch.cuda.amp import GradScaler
# # from utils.optimizer import build_optimizer, build_scheduler
# # from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, plot_confusion_matrix
# # from utils.logger import create_logger
# # from datasets.build import build_dataloader
# # from datasets.blending import FixMixupBlending
# # from utils.config import get_config
# # from models import AU_clip
# # import torch.nn.functional as F
# # K_TABLE_DFEW = {
# #     0: [0.37458275378581574, 0.37673577978763106, 0.41551698634677164, 0.20406588498862172,
# #         0.7927884197377446, 0.8938573401735845, 0.26152800062149906, 0.8999413926399066,
# #         0.9306407335956202, 0.8244182955003968, 0.2344366715203141, 0.35584905526535116,
# #         0.2637127706449251, 0.1980659294341865, 0.8749731256118369, 0.5415473658691405,
# #         0.4089852072091214, 0.29454183039387216],
# #     1: [0.3563166809248465, 0.22025123080620432, 0.7883122756737404, 0.18552393229679323, 0.4330479071201498, 0.6512463687571445, 0.19562608246203084, 0.57228053429373, 0.3919835645137292, 0.5539327475091499, 0.26436050797140115, 0.3859719011102806, 0.2624203556065227, 0.19099832378533962, 0.5784889747902296, 0.5465085543388307, 0.4089852072091214, 0.31238729511297314],  # sad
# #     2: [0.29634994040489937, 0.26249346997682327, 0.5349197805247116, 0.211357886947774, 0.24956316484226185, 0.48303878792569305, 0.1750459267403643, 0.4222786033590969, 0.2595297603319664, 0.4038707629862569, 0.2255822802606922, 0.3300116586908106, 0.21700897117154114, 0.18584396273490125, 0.4406741647399928, 0.4892561535112399, 0.4089852072091214, 0.24616267045793816],  # neutral
# #     3: [0.2565426020973117, 0.22590006587965014, 0.7487900008174104, 0.24260938453624706, 0.3297348731445176, 0.6253054547733351, 0.2331741603449138, 0.6002659975024218, 0.25865288585343615, 0.4042681727124247, 0.25714836786909934, 0.4073109215604269, 0.22487756342746285, 0.1940789790636582, 0.7184070119769908, 0.6300831522957303, 0.4089852072091214, 0.22674904330132986],  # angry
# #     4: [0.45840639969725916, 0.4348135864186588, 0.5807746998255159, 0.32444326825812997, 0.27370636575780566, 0.49552874169247974, 0.22029419679136286, 0.45675173141757053, 0.30048086399990936, 0.35356016873960344, 0.2340917014067623, 0.35356016873960344, 0.22418778976044848, 0.18774040869030503, 0.6204770513319532, 0.6347324688342945, 0.4089852072091214, 0.2326942881893769],  # surprise
# #     5: [0.3669101379226728, 0.23849898584920362, 0.8556853679596857, 0.2093396686826401, 0.47998741884320045, 0.9116481111605065, 0.47437419451562296, 0.8176370658093514, 0.3690013619929339, 0.6122010138157353, 0.37313028939087656, 0.43845703949339593, 0.2712229960450541, 0.22790281312128532, 0.7303843653361992, 0.5736046807175323, 0.4089852072091214, 0.3157383637072392],  # disgust
# #     6: [0.4668184913800416, 0.3236830048705496, 0.7092147921039609, 0.3237692334141919, 0.284442322105034, 0.4833791428919572, 0.19268179747966238, 0.49211626418097015, 0.29857493916455136, 0.40402139627383793, 0.3114334189399665, 0.42236998344528387, 0.2762397282204001, 0.20086024967771943, 0.6377967064381651, 0.6643128457350147, 0.4089852072091214, 0.22004117429592432],  # fear
# # }
# # K_SCALE = 5.0

# # POSW_GLOBAL_DFEW = [8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674,
# #                    4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794,
# #                    7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384,
# #                    12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372,
# #                    101.28938906752411, 8.483380533611566]

# # POSW_DISTINCT_DFEW = {
# #     0: [4.648862512363996, 4.162923411588553, 1.83239825175909, 2.5131103421760863, 1.0496164371270917, 1.191954326288771, 15.430099793221252, 0.636697444899202, 1.0033104960263086, 0.7217365088935785, 3.69328950409615, 2.862698681095705, 6.9568094740508535, 4.616744014506562, 2.920370688175734, 5.462006293978289, 57.59313882654697, 4.170958066889254],  # happy
# #     1: [5.6823671940967, 4.992658194508461, 0.5149984185907146, 2.6989371862870613, 3.1921004145555045, 1.6714365386873313, 19.07732186190161, 2.1056834274599465, 7.311081685767773, 1.9735191296108734, 5.039584577609662, 3.51384996900186, 8.076543333000897, 6.52494935714581, 3.628397792864953, 5.6926866933852995, 70.26898981989036, 5.224216933388045],  # sad
# #     2: [6.4774986002239645, 5.6010812480692, 0.8710279064472912, 1.9167337801498792, 7.8117860530331145, 2.7351547887496284, 21.302160526041124, 3.19245786489297, 20.999073406774425, 2.574808023689626, 5.714757086292502, 3.906024704963953, 8.191199242945629, 5.206488904380156, 4.249791165053314, 7.363091976516634, 81.4053220208253, 4.963300960035722],  # neutral
# #     3: [4.928812812224508, 4.114906832298137, 0.9234234234234234, 1.4543961558346765, 4.781224255883091, 2.435997871208089, 12.75189571440743, 1.44547134935305, 13.252185430463577, 3.1313061506565307, 3.579413266753674, 2.578767654819184, 5.92967542503864, 5.292631578947368, 2.6113572291582763, 4.433813627794237, 55.85311729482212, 4.549840112780662],  # angry
# #     4: [7.116157728166966, 6.34866790582404, 0.9600900658968373, 0.761682850299846, 17.648977987421382, 5.163029358274876, 21.278938718008924, 6.183435536376713, 35.41059094397544, 6.815336463223788, 6.358355951919348, 4.08091030789826, 9.571078431372548, 7.167857450288371, 5.090243902439024, 8.104394549990404, 94.26706827309236, 6.122504128509233],  # surprise
# #     5: [4.229214780600462, 4.09392575928009, 0.7474435655026047, 2.1834797891036906, 5.054144385026738, 1.6915304606240713, 10.307116104868914, 2.2381122631390777, 10.731865284974093, 3.210599721059972, 4.137266023823029, 3.012848914488259, 5.983037779491133, 6.148382004735596, 3.6374807987711213, 5.387165021156559, 27.391849529780565, 3.2964895635673623],  # disgust
# #     6: [5.96072648535643, 5.154494891980657, 0.7569813845450734, 1.5509251975236857, 7.060588375159415, 2.779909786630176, 17.84796563052818, 2.8554517069539505, 15.333362533397574, 3.2621352565348083, 4.923954312221004, 3.5370230679384855, 7.644713355124371, 5.762544656620061, 3.9785987023043443, 6.670773851153989, 57.87385538364383, 5.229860670252932],  # fear
# # }

# # # minor strategy: define which classes are "major"
# # MAJOR_CLASSES_DFEW = {0, 1, 2, 3}

# # def compute_au_loss_stage_b(au_logits, au_targets, labels, posw_option='global'):
# #     """
# #     au_logits:  [B,18] float
# #     au_targets: [B,18] float {0,1}
# #     labels:     [B] long, emotion class ids (0..C-1)
# #     """
# #     device = au_logits.device
# #     B = labels.shape[0]
# #     # pos_weight: [B,18]
# #     if config.DATA.DATASET=='DFEW':
# #         # knowledge weight: [B,18]
# #         k = torch.stack(
# #             [torch.tensor(K_TABLE_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
# #             dim=0
# #         ) * K_SCALE
# #         if posw_option == 'global':
# #             pw = torch.tensor(POSW_GLOBAL_DFEW, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
# #         elif posw_option == 'distinct':
# #             pw = torch.stack(
# #                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
# #                 dim=0
# #             )
# #         elif posw_option == 'minor':
# #             pw = torch.stack(
# #                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
# #                 dim=0
# #             )
# #             # overwrite major classes with 1s
# #             mask_major = torch.tensor([int(c) in MAJOR_CLASSES_DFEW for c in labels], device=device, dtype=torch.bool)
# #             if mask_major.any():
# #                 pw = pw.clone()
# #                 pw[mask_major] = 1.0
# #         else:
# #             raise ValueError(f'Unknown posw_option: {posw_option}')

# #     # weighted BCE with logits
# #     return F.binary_cross_entropy_with_logits(
# #         au_logits, au_targets,
# #         weight=k,
# #         pos_weight=pw,
# #         reduction='mean'
# #     )

# # def parse_option():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/dfew7/16_16.yaml')
# #     parser.add_argument(
# #         "--opts",
# #         help="Modify config options by adding 'KEY VALUE' pairs. ",
# #         default=None,
# #         nargs='+',
# #     )
# #     parser.add_argument('--gpu', default=[0, 1], type=int,help='GPU id to use.')
# #     parser.add_argument('--output', type=str, default="DFEWAS2")
# #     parser.add_argument('--resume', type=str)
# #     parser.add_argument('--pretrained', type=str)
# #     parser.add_argument('--only_test', action='store_true')
# #     parser.add_argument('--batch-size', type=int)
# #     parser.add_argument('--accumulation-steps', type=int)
# #     parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
# #     args = parser.parse_args()
# #     config = get_config(args)
# #     return args, config


# # def main(config):
# #     # load train and valid dataset
# #     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

# #     # load pretrained model
# #     model, _ = AU_clip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
# #                           device="cpu", jit=False,
# #                           T=config.DATA.NUM_FRAMES,
# #                           droppath=config.MODEL.DROP_PATH_RATE,
# #                           use_checkpoint=config.TRAIN.USE_CHECKPOINT,
# #                           use_cache=config.MODEL.FIX_TEXT,
# #                           logger=logger,
# #                           N=config.DATA.NUM_DIVIDE,
# #                           cfg=config
# #                           )
# #     model = model.cuda()

# #     # training data augmentation
# #     mixup_fn = None
# #     if config.AUG.MIXUP > 0:
# #         criterion = SoftTargetCrossEntropy()
# #         criterion_soft = SoftTargetCrossEntropy()
# #         mixup_fn = FixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
# #                                     smoothing=config.AUG.LABEL_SMOOTH,
# #                                     mixup_alpha=config.AUG.MIXUP,
# #                                     fmix_alpha=config.AUG.CUTMIX,
# #                                     switch_prob=config.AUG.MIXUP_SWITCH_PROB)
# #     elif config.AUG.LABEL_SMOOTH > 0:
# #         criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
# #         criterion_soft = SoftTargetCrossEntropy()

# #     else:
# #         criterion = nn.CrossEntropyLoss()
# #         criterion_soft = SoftTargetCrossEntropy()

# #     optimizer = build_optimizer(config, model)
# #     lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
# #     model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
# #                                                       find_unused_parameters=True)

# #     start_epoch = 0
# #     max_war_global, max_war_local, max_war_fuse = 0.0, 0.0, 0.0

# #     # Track best epoch information (all from the same epoch with best WAR)
# #     best_epoch = 0
# #     best_war_global = 0.0
# #     best_uar_global = 0.0
# #     best_cls_acc_global = None
# #     best_war_local = 0.0
# #     best_uar_local = 0.0
# #     best_cls_acc_local = None
# #     best_war_fuse = 0.0
# #     best_uar_fuse = 0.0
# #     best_cls_acc_fuse = None
# #     # retrain
# #     if config.TRAIN.AUTO_RESUME:
# #         resume_file_path = auto_resume_helper(config.OUTPUT)
# #         if resume_file_path:
# #             config.defrost()
# #             config.MODEL.RESUME = resume_file_path
# #             config.freeze()
# #             logger.info(f'auto resuming from {resume_file_path}')
# #         else:
# #             logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
# #     if config.MODEL.RESUME:
# #         start_epoch, _ = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

# #     # textual prompt
# #     text_labels = generate_text(train_data)

# #     # model test
# #     if config.TEST.ONLY_TEST:
# #         results = validate(val_loader, text_labels, model, config)
# #         acc1, acc1_local, acc1_fuse = results[0], results[1], results[2]
# #         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
# #         logger.info(f"Accuracy of the network on the {len(val_data)} test videos: WAR={acc1:.1f}% UAR={uar_global:.1f}%")
# #         return
# #     print("Start training")
# #     #model train and valid
# #     for epoch in range(start_epoch, config.TRAIN.EPOCHS):
# #         train_loader.sampler.set_epoch(epoch)
# #         train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft)

# #         # Global, local and fuse classification accuracy
# #         results = validate(val_loader, text_labels, model, config)
# #         acc_global, acc_local, acc_fuse = results[0], results[1], results[2]
# #         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
# #         war_global, war_local, war_fuse = results[6], results[7], results[8]
# #         cls_acc_global, cls_acc_local, cls_acc_fuse = results[9], results[10], results[11]

# #         # Get emotion class names
# #         if config.DATA.DATASET.lower() == 'dfew':
# #             emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
# #         else:
# #             emotion_names = [f'Class_{i}' for i in range(len(cls_acc_global))]

# #         # Log detailed results for current epoch
# #         logger.info(f"=" * 80)
# #         logger.info(f"Epoch [{epoch}/{config.TRAIN.EPOCHS - 1}] Validation Results:")
# #         logger.info(f"-" * 80)

# #         # Global results
# #         logger.info(f"Global Branch:")
# #         logger.info(f"  WAR: {war_global:.2f}%  |  UAR: {uar_global:.2f}%")
# #         logger.info(f"  Per-class Accuracy:")
# #         for name, acc in zip(emotion_names, cls_acc_global):
# #             logger.info(f"    {name:10s}: {acc*100:.2f}%")

# #         # Local results
# #         logger.info(f"-" * 80)
# #         logger.info(f"Local Branch:")
# #         logger.info(f"  WAR: {war_local:.2f}%  |  UAR: {uar_local:.2f}%")
# #         logger.info(f"  Per-class Accuracy:")
# #         for name, acc in zip(emotion_names, cls_acc_local):
# #             logger.info(f"    {name:10s}: {acc*100:.2f}%")

# #         # Fuse results
# #         logger.info(f"-" * 80)
# #         logger.info(f"Fuse Branch:")
# #         logger.info(f"  WAR: {war_fuse:.2f}%  |  UAR: {uar_fuse:.2f}%")
# #         logger.info(f"  Per-class Accuracy:")
# #         for name, acc in zip(emotion_names, cls_acc_fuse):
# #             logger.info(f"    {name:10s}: {acc*100:.2f}%")
# #         logger.info(f"=" * 80)

# #         is_best = war_global > max_war_global  # Use WAR (acc_global) as the criterion for best model

# #         # Update best epoch info - only update when finding a new best WAR
# #         if is_best:
# #             best_epoch = epoch
# #             # Update all metrics from the SAME epoch (the one with best WAR)
# #             best_war_global = war_global
# #             best_uar_global = uar_global
# #             best_cls_acc_global = cls_acc_global.copy()
# #             best_war_local = war_local
# #             best_uar_local = uar_local
# #             best_cls_acc_local = cls_acc_local.copy()
# #             best_war_fuse = war_fuse
# #             best_uar_fuse = uar_fuse
# #             best_cls_acc_fuse = cls_acc_fuse.copy()

# #             # Update max values - these are from the best epoch
# #             max_war_global = war_global
# #             max_war_local = war_local
# #             max_war_fuse = war_fuse

# #             logger.info(f">>> New best model found at epoch {epoch}! WAR: {war_global:.2f}% UAR: {uar_global:.2f}% <<<")

# #         logger.info(f'Current Best Epoch: {best_epoch}')
# #         logger.info(f'Best Global - WAR: {max_war_global:.2f}% | UAR: {best_uar_global:.2f}%')
# #         logger.info(f'Best Local  - WAR: {max_war_local:.2f}% | UAR: {best_uar_local:.2f}%')
# #         logger.info(f'Best Fuse   - WAR: {max_war_fuse:.2f}% | UAR: {best_uar_fuse:.2f}%')
# #         # save model
# #         if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
# #             epoch_saving(config, epoch, model.module, max_war_global, optimizer, lr_scheduler, logger, config.OUTPUT,
# #                          is_best)

# #     # Print best epoch summary
# #     logger.info(f"\n" + "=" * 80)
# #     logger.info(f"TRAINING COMPLETED - BEST MODEL SUMMARY")
# #     logger.info(f"=" * 80)
# #     logger.info(f"Best Epoch: {best_epoch}")
# #     logger.info(f"-" * 80)

# #     # Get emotion class names
# #     if config.DATA.DATASET.lower() == 'dfew':
# #         emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
# #     else:
# #         emotion_names = [f'Class_{i}' for i in range(len(best_cls_acc_global)) if best_cls_acc_global is not None]

# #     if best_cls_acc_global is not None:
# #         logger.info(f"Global Branch (Best):")
# #         logger.info(f"  WAR: {best_war_global:.2f}%  |  UAR: {best_uar_global:.2f}%")
# #         logger.info(f"  Per-class Accuracy:")
# #         for name, acc in zip(emotion_names, best_cls_acc_global):
# #             logger.info(f"    {name:10s}: {acc*100:.2f}%")

# #         logger.info(f"-" * 80)
# #         logger.info(f"Local Branch (Best):")
# #         logger.info(f"  WAR: {best_war_local:.2f}%  |  UAR: {best_uar_local:.2f}%")
# #         logger.info(f"  Per-class Accuracy:")
# #         for name, acc in zip(emotion_names, best_cls_acc_local):
# #             logger.info(f"    {name:10s}: {acc*100:.2f}%")

# #         logger.info(f"-" * 80)
# #         logger.info(f"Fuse Branch (Best):")
# #         logger.info(f"  WAR: {best_war_fuse:.2f}%  |  UAR: {best_uar_fuse:.2f}%")
# #         logger.info(f"  Per-class Accuracy:")
# #         for name, acc in zip(emotion_names, best_cls_acc_fuse):
# #             logger.info(f"    {name:10s}: {acc*100:.2f}%")
# #     logger.info(f"=" * 80 + "\n")

# #     # validation after training
# #     config.defrost()
# #     config.TEST.NUM_CLIP = 4
# #     config.TEST.NUM_CROP = 3
# #     config.freeze()
# #     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
# #     results = validate(val_loader, text_labels, model, config)
# #     acc = results[0]
# #     uar_global = results[3]
# #     logger.info(f"Final Accuracy: WAR={acc:.2f}% UAR={uar_global:.2f}%")


# # def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft):
# #     au_criterion = torch.nn.BCEWithLogitsLoss()
# #     model.train()
# #     optimizer.zero_grad()

# #     num_steps = len(train_loader)
# #     batch_time = AverageMeter()
# #     tot_loss_meter = AverageMeter()

# #     start = time.time()
# #     end = time.time()
# #     scaler = GradScaler()
# #     texts = text_labels.cuda(non_blocking=True)

# #     for idx, batch_data in enumerate(train_loader):
# #         images = batch_data["imgs"].cuda(non_blocking=True)
# #         label_id = batch_data["label"].cuda(non_blocking=True)
# #         au_labels = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
# #         label_id = label_id.reshape(-1)
# #         images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

# #         if mixup_fn is not None:
# #             images, label_id = mixup_fn(images, label_id)

# #         if texts.shape[0] == 1:
# #             texts = texts.view(1, -1)
# #         with autocast():
# #             output_global, output_local, feat, au_logits = model(images, texts)
# #             if epoch>13:
# #                 total_loss = criterion(output_global, label_id)
# #                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
# #             else:
# #                 pre_output_local = torch.sum(
# #                     output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
# #                 # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
# #                 total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
# #                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
# #             if config.AU.ENABLED and mixup_fn is None:
# #                 au_targets = batch_data["au"].cuda(non_blocking=True).float()
# #                 au_loss = compute_au_loss_stage_b(
# #                     au_logits, au_targets, label_id,
# #                     posw_option=config.AU.POSW_OPTION
# #                 )
# #                 total_loss = total_loss + config.AU.LAMBDA * au_loss
# #                 #total_loss = (1-config.AU.LAMBDA) * total_loss + config.AU.LAMBDA * au_loss

# #         if config.TRAIN.ACCUMULATION_STEPS == 1:
# #             optimizer.zero_grad()
# #         if config.TRAIN.OPT_LEVEL != 'O0':
# #             scaler.scale(total_loss).backward()
# #             scaler.step(optimizer)
# #             scaler.update()
# #         else:
# #             total_loss.backward()
# #             optimizer.step()

# #         if config.TRAIN.ACCUMULATION_STEPS > 1:
# #             if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
# #                 scaler.step(optimizer)
# #                 scaler.update()
# #                 optimizer.zero_grad()
# #                 lr_scheduler.step_update(epoch * num_steps + idx)
# #         else:
# #             scaler.step(optimizer)
# #             scaler.update()
# #             lr_scheduler.step_update(epoch * num_steps + idx)

# #         torch.cuda.synchronize()

# #         tot_loss_meter.update(total_loss.item(), len(label_id))
# #         batch_time.update(time.time() - end)
# #         end = time.time()

# #         if idx % config.PRINT_FREQ == 0:
# #             lr = optimizer.param_groups[0]['lr']
# #             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
# #             etas = batch_time.avg * (num_steps - idx)
# #             logger.info(
# #                 f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
# #                 f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
# #                 f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
# #                 f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
# #                 f'mem {memory_used:.0f}MB')
# #     epoch_time = time.time() - start
# #     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# # @torch.no_grad()
# # def validate(val_loader, text_labels, model, config):
# #     model.eval()

# #     acc_global_meter, acc_local_meter, acc_fuse_meter = AverageMeter(), AverageMeter(), AverageMeter()

# #     probility = []
# #     video_pre_global = []
# #     video_pre_local = []
# #     video_pre_fuse = []
# #     video_label = []
# #     with torch.no_grad():
# #         text_inputs = text_labels.cuda()
# #         logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
# #         for idx, batch_data in enumerate(val_loader):
# #             _image = batch_data["imgs"]
# #             label_id = batch_data["label"]
# #             label_id = label_id.reshape(-1)

# #             b, tn, c, h, w = _image.size()

# #             t = config.DATA.NUM_FRAMES
# #             n = tn // t
# #             _image = _image.view(b, n, t, c, h, w)
# #             tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
# #             tot_similarity_local = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
# #             for i in range(n):
# #                 image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
# #                 label_id = label_id.cuda(non_blocking=True)
# #                 image_input = image.cuda(non_blocking=True)

# #                 if config.TRAIN.OPT_LEVEL == 'O2':
# #                     image_input = image_input.half()
# #                 with autocast():
# #                     output, output_local, feat, _ = model(image_input, text_inputs)
# #                 if idx < 1:
# #                     feature = feat
# #                 else:
# #                     feature = torch.cat((feature, feat), dim=0)

# #                 pre_output_global = output.view(b, -1)
# #                 pre_output_local = torch.sum(output_local.view(b, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES),dim=1).squeeze(dim=-1)

# #                 similarity = pre_output_global.view(b, -1).softmax(dim=-1)
# #                 tot_similarity += similarity

# #                 similarity_local = pre_output_local.view(b, -1).softmax(dim=-1)
# #                 tot_similarity_local += similarity_local.view(b, -1)

# #             probility.extend(tot_similarity.data.cpu().numpy().copy())
# #             values_global, indices_global = tot_similarity.topk(1, dim=-1)

# #             values_local, indices_local = tot_similarity_local.topk(1, dim=-1)

# #             fuse_similarity = tot_similarity + tot_similarity_local
# #             values_fuse, indices_fuse = fuse_similarity.topk(1, dim=-1)

# #             acc_global = 0
# #             acc_local = 0
# #             acc_fuse = 0
# #             for i in range(b):
# #                 video_pre_global.append(indices_global[i].data.cpu().numpy().copy())
# #                 video_pre_local.append(indices_local[i].data.cpu().numpy().copy())
# #                 video_pre_fuse.append(indices_fuse[i].data.cpu().numpy().copy())
# #                 video_label.append(label_id[i].data.cpu().numpy().copy())
# #                 if indices_global[i] == label_id[i]:
# #                     acc_global += 1
# #                 if indices_local[i] == label_id[i]:
# #                     acc_local += 1
# #                 if indices_fuse[i] == label_id[i]:
# #                     acc_fuse += 1

# #             acc_global_meter.update(float(acc_global) / b * 100, b)
# #             acc_local_meter.update(float(acc_local) / b * 100, b)
# #             acc_fuse_meter.update(float(acc_fuse) / b * 100, b)
# #             if idx % config.PRINT_FREQ == 0:
# #                 logger.info(
# #                     f'Test: [{idx}/{len(val_loader)}]\t'
# #                     f'Acc@1: {acc_global_meter.avg:.3f}\t'
# #                     f'Acc@1: {acc_local_meter.avg:.3f}\t'
# #                     f'Acc@1: {acc_fuse_meter.avg:.3f}\t'
# #                 )
# #     # confusion matrix
# #     cf = confusion_matrix(video_label, video_pre_global)
# #     np.set_printoptions(precision=4)
# #     normalized_cm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
# #     normalized_cm = normalized_cm * 100

# #     cls_cnt = normalized_cm.sum(axis=1)
# #     cls_hit = np.diag(normalized_cm)
# #     # print(cf)
# #     cls_acc = cls_hit / cls_cnt
# #     cls_acc = np.around(cls_acc, 4)
# #     cm = np.array(normalized_cm)
# #     # save_path = 'AU-CLIP/results'
# #     # if not os.path.exists(save_path):
# #     #     os.makedirs(save_path)
# #     # labels_name = ['hap', 'sad', 'neu', 'ang', 'sur', 'dis', 'fea']
# #     # plot_confusion_matrix(cm, labels_name, 'AUCLIP', cls_acc)
# #     #
# #     # #t-SNE
# #     # col = ['orange', 'purple', 'g', 'r', 'darkblue', 'chocolate', 'c']
# #     # x_embed = TSNE(n_components=2, perplexity=100, n_iter=10000).fit_transform(feature.data.cpu())
# #     # label = np.array(video_label)
# #     # plt.figure(figsize=(6, 6))
# #     # for i in range(7):
# #     #     idxs = np.where(label == i)[0]
# #     #     plt.scatter(x_embed[idxs, 0], x_embed[idxs, 1], color=col[i], s=6, label=labels_name[i])
# #     # plt.legend(loc='upper left')
# #     # plt.xticks(fontsize=13)
# #     # plt.yticks(fontsize=13)
# #     # plt.savefig(os.path.join(save_path, 'AUCLIP_TSNE.jpg'), format='jpg')
# #     # plt.show()

# #     logger.info(f'Global - Class-wise Accuracy: {cls_acc}')
# #     upper = np.mean(np.max(cf, axis=1) / cls_cnt)
# #     logger.info(f'Global - Upper bound: {upper}')
# #     logger.info('Global - Evaluation is finished')
# #     logger.info(f'Global - Class Accuracy (UAR): {np.mean(cls_acc) * 100:.2f}%')

# #     cf_local = confusion_matrix(video_label, video_pre_local).astype(float)
# #     cls_cnt_local = cf_local.sum(axis=1)
# #     cls_hit_local = np.diag(cf_local)
# #     # print(cf)
# #     cls_acc_local = cls_hit_local / cls_cnt_local
# #     cls_acc_local = np.around(cls_acc_local, 4)
# #     logger.info(f'Local - Class-wise Accuracy: {cls_acc_local}')
# #     upper = np.mean(np.max(cf_local, axis=1) / cls_cnt_local)
# #     logger.info(f'Local - Upper bound: {upper}')
# #     logger.info('Local - Evaluation is finished')
# #     logger.info(f'Local - Class Accuracy (UAR): {np.mean(cls_acc_local) * 100:.2f}%')

# #     cf_fuse = confusion_matrix(video_label, video_pre_fuse).astype(float)
# #     cls_cnt_fuse = cf_fuse.sum(axis=1)
# #     cls_hit_fuse = np.diag(cf_fuse)
# #     # print(cf)
# #     cls_acc_fuse = cls_hit_fuse / cls_cnt_fuse
# #     cls_acc_fuse = np.around(cls_acc_fuse, 4)
# #     logger.info(f'Fuse - Class-wise Accuracy: {cls_acc_fuse}')
# #     upper = np.mean(np.max(cf_fuse, axis=1) / cls_cnt_fuse)
# #     logger.info(f'Fuse - Upper bound: {upper}')
# #     logger.info('Fuse - Evaluation is finished')
# #     logger.info(f'Fuse - Class Accuracy (UAR): {np.mean(cls_acc_fuse) * 100:.2f}%')

# #     acc_global_meter.sync()
# #     acc_local_meter.sync()
# #     acc_fuse_meter.sync()
# #     logger.info(f' * Acc@1 {acc_global_meter.avg:.3f} Acc_loca@1 {acc_local_meter.avg:.3f}  Acc_fuse@1 {acc_fuse_meter.avg:.3f}')

# #     # Calculate UAR (Unweighted Average Recall) and WAR (Weighted Average Recall)
# #     uar_global = np.mean(cls_acc) * 100  # UAR for global
# #     uar_local = np.mean(cls_acc_local) * 100  # UAR for local
# #     uar_fuse = np.mean(cls_acc_fuse) * 100  # UAR for fuse
# #     war_global = acc_global_meter.avg  # WAR is the same as overall accuracy
# #     war_local = acc_local_meter.avg
# #     war_fuse = acc_fuse_meter.avg

# #     return (acc_global_meter.avg, acc_local_meter.avg, acc_fuse_meter.avg,
# #             uar_global, uar_local, uar_fuse, war_global, war_local, war_fuse,
# #             cls_acc, cls_acc_local, cls_acc_fuse,
# #             video_label, video_pre_global, video_pre_local, video_pre_fuse)


# # if __name__ == '__main__':
# #     args, config = parse_option()

# #     # 初始化分布式环境
# #     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
# #         rank = int(os.environ["RANK"])
# #         world_size = int(os.environ['WORLD_SIZE'])
# #         local_rank = int(os.environ['LOCAL_RANK'])  # 使用环境变量中的 LOCAL_RANK
# #         print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
# #     else:
# #         rank = -1
# #         world_size = -1
# #         local_rank = args.local_rank  # 如果未设置环境变量，则使用命令行参数

# #     # 设置当前 GPU 设备
# #     torch.cuda.set_device(local_rank)

# #     # 初始化分布式进程组
# #     dist.init_process_group(
# #         backend='nccl',
# #         init_method='env://',
# #         world_size=world_size,
# #         rank=rank
# #     )

# #     # 确保所有进程同步
# #     dist.barrier()

# #     # 设置随机种子
# #     seed = config.SEED + dist.get_rank()
# #     torch.manual_seed(seed)
# #     np.random.seed(seed)
# #     random.seed(seed)
# #     cudnn.benchmark = True

# #     # 创建输出目录
# #     output_dir = Path(config.OUTPUT)
# #     output_dir.mkdir(parents=True, exist_ok=True)

# #     # 创建日志记录器
# #     logger = create_logger(output_dir=output_dir, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
# #     logger.info(f"Working directory: {output_dir}")

# #     # 保存配置文件（仅主进程执行）
# #     if dist.get_rank() == 0:
# #         logger.info(config)
# #         shutil.copy(args.config, output_dir)

# #     # 启动主训练逻辑
# #     main(config)



# import os
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import argparse
# import datetime
# import shutil
# import time
# import numpy as np
# import random
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from pathlib import Path
# from sklearn.metrics import confusion_matrix
# from sklearn.manifold import TSNE
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
# from utils.optimizer import build_optimizer, build_scheduler
# from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, plot_confusion_matrix
# from utils.logger import create_logger
# from datasets.build import build_dataloader
# from datasets.blending import FixMixupBlending
# from utils.config import get_config
# from models import AU_clip
# import torch.nn.functional as F
# from train_with_au_mixup import train_one_epoch_with_au_mixup, mixup_data_with_au

# K_TABLE_DFEW = {
#     0: [0.37458275378581574, 0.37673577978763106, 0.41551698634677164, 0.20406588498862172,
#         0.7927884197377446, 0.8938573401735845, 0.26152800062149906, 0.8999413926399066,
#         0.9306407335956202, 0.8244182955003968, 0.2344366715203141, 0.35584905526535116,
#         0.2637127706449251, 0.1980659294341865, 0.8749731256118369, 0.5415473658691405,
#         0.4089852072091214, 0.29454183039387216],
#     1: [0.3563166809248465, 0.22025123080620432, 0.7883122756737404, 0.18552393229679323, 0.4330479071201498, 0.6512463687571445, 0.19562608246203084, 0.57228053429373, 0.3919835645137292, 0.5539327475091499, 0.26436050797140115, 0.3859719011102806, 0.2624203556065227, 0.19099832378533962, 0.5784889747902296, 0.5465085543388307, 0.4089852072091214, 0.31238729511297314],  # sad
#     2: [0.29634994040489937, 0.26249346997682327, 0.5349197805247116, 0.211357886947774, 0.24956316484226185, 0.48303878792569305, 0.1750459267403643, 0.4222786033590969, 0.2595297603319664, 0.4038707629862569, 0.2255822802606922, 0.3300116586908106, 0.21700897117154114, 0.18584396273490125, 0.4406741647399928, 0.4892561535112399, 0.4089852072091214, 0.24616267045793816],  # neutral
#     3: [0.2565426020973117, 0.22590006587965014, 0.7487900008174104, 0.24260938453624706, 0.3297348731445176, 0.6253054547733351, 0.2331741603449138, 0.6002659975024218, 0.25865288585343615, 0.4042681727124247, 0.25714836786909934, 0.4073109215604269, 0.22487756342746285, 0.1940789790636582, 0.7184070119769908, 0.6300831522957303, 0.4089852072091214, 0.22674904330132986],  # angry
#     4: [0.45840639969725916, 0.4348135864186588, 0.5807746998255159, 0.32444326825812997, 0.27370636575780566, 0.49552874169247974, 0.22029419679136286, 0.45675173141757053, 0.30048086399990936, 0.35356016873960344, 0.2340917014067623, 0.35356016873960344, 0.22418778976044848, 0.18774040869030503, 0.6204770513319532, 0.6347324688342945, 0.4089852072091214, 0.2326942881893769],  # surprise
#     5: [0.3669101379226728, 0.23849898584920362, 0.8556853679596857, 0.2093396686826401, 0.47998741884320045, 0.9116481111605065, 0.47437419451562296, 0.8176370658093514, 0.3690013619929339, 0.6122010138157353, 0.37313028939087656, 0.43845703949339593, 0.2712229960450541, 0.22790281312128532, 0.7303843653361992, 0.5736046807175323, 0.4089852072091214, 0.3157383637072392],  # disgust
#     6: [0.4668184913800416, 0.3236830048705496, 0.7092147921039609, 0.3237692334141919, 0.284442322105034, 0.4833791428919572, 0.19268179747966238, 0.49211626418097015, 0.29857493916455136, 0.40402139627383793, 0.3114334189399665, 0.42236998344528387, 0.2762397282204001, 0.20086024967771943, 0.6377967064381651, 0.6643128457350147, 0.4089852072091214, 0.22004117429592432],  # fear
# }
# K_SCALE = 5.0

# POSW_GLOBAL_DFEW = [8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674,
#                    4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794,
#                    7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384,
#                    12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372,
#                    101.28938906752411, 8.483380533611566]

# POSW_DISTINCT_DFEW = {
#     0: [4.648862512363996, 4.162923411588553, 1.83239825175909, 2.5131103421760863, 1.0496164371270917, 1.191954326288771, 15.430099793221252, 0.636697444899202, 1.0033104960263086, 0.7217365088935785, 3.69328950409615, 2.862698681095705, 6.9568094740508535, 4.616744014506562, 2.920370688175734, 5.462006293978289, 57.59313882654697, 4.170958066889254],  # happy
#     1: [5.6823671940967, 4.992658194508461, 0.5149984185907146, 2.6989371862870613, 3.1921004145555045, 1.6714365386873313, 19.07732186190161, 2.1056834274599465, 7.311081685767773, 1.9735191296108734, 5.039584577609662, 3.51384996900186, 8.076543333000897, 6.52494935714581, 3.628397792864953, 5.6926866933852995, 70.26898981989036, 5.224216933388045],  # sad
#     2: [6.4774986002239645, 5.6010812480692, 0.8710279064472912, 1.9167337801498792, 7.8117860530331145, 2.7351547887496284, 21.302160526041124, 3.19245786489297, 20.999073406774425, 2.574808023689626, 5.714757086292502, 3.906024704963953, 8.191199242945629, 5.206488904380156, 4.249791165053314, 7.363091976516634, 81.4053220208253, 4.963300960035722],  # neutral
#     3: [4.928812812224508, 4.114906832298137, 0.9234234234234234, 1.4543961558346765, 4.781224255883091, 2.435997871208089, 12.75189571440743, 1.44547134935305, 13.252185430463577, 3.1313061506565307, 3.579413266753674, 2.578767654819184, 5.92967542503864, 5.292631578947368, 2.6113572291582763, 4.433813627794237, 55.85311729482212, 4.549840112780662],  # angry
#     4: [7.116157728166966, 6.34866790582404, 0.9600900658968373, 0.761682850299846, 17.648977987421382, 5.163029358274876, 21.278938718008924, 6.183435536376713, 35.41059094397544, 6.815336463223788, 6.358355951919348, 4.08091030789826, 9.571078431372548, 7.167857450288371, 5.090243902439024, 8.104394549990404, 94.26706827309236, 6.122504128509233],  # surprise
#     5: [4.229214780600462, 4.09392575928009, 0.7474435655026047, 2.1834797891036906, 5.054144385026738, 1.6915304606240713, 10.307116104868914, 2.2381122631390777, 10.731865284974093, 3.210599721059972, 4.137266023823029, 3.012848914488259, 5.983037779491133, 6.148382004735596, 3.6374807987711213, 5.387165021156559, 27.391849529780565, 3.2964895635673623],  # disgust
#     6: [5.96072648535643, 5.154494891980657, 0.7569813845450734, 1.5509251975236857, 7.060588375159415, 2.779909786630176, 17.84796563052818, 2.8554517069539505, 15.333362533397574, 3.2621352565348083, 4.923954312221004, 3.5370230679384855, 7.644713355124371, 5.762544656620061, 3.9785987023043443, 6.670773851153989, 57.87385538364383, 5.229860670252932],  # fear
# }

# # minor strategy: define which classes are "major"
# MAJOR_CLASSES_DFEW = {0, 1, 2, 3}

# def compute_au_loss_stage_b(au_logits, au_targets, labels, posw_option='global'):
#     """
#     au_logits:  [B,18] float
#     au_targets: [B,18] float {0,1}
#     labels:     [B] long, emotion class ids (0..C-1)
#     """
#     device = au_logits.device
#     B = labels.shape[0]
#     # pos_weight: [B,18]
#     if config.DATA.DATASET=='DFEW':
#         # knowledge weight: [B,18]
#         k = torch.stack(
#             [torch.tensor(K_TABLE_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#             dim=0
#         ) * K_SCALE
#         if posw_option == 'global':
#             pw = torch.tensor(POSW_GLOBAL_DFEW, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
#         elif posw_option == 'distinct':
#             pw = torch.stack(
#                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#                 dim=0
#             )
#         elif posw_option == 'minor':
#             pw = torch.stack(
#                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#                 dim=0
#             )
#             # overwrite major classes with 1s
#             mask_major = torch.tensor([int(c) in MAJOR_CLASSES_DFEW for c in labels], device=device, dtype=torch.bool)
#             if mask_major.any():
#                 pw = pw.clone()
#                 pw[mask_major] = 1.0
#         else:
#             raise ValueError(f'Unknown posw_option: {posw_option}')

#     # weighted BCE with logits
#     return F.binary_cross_entropy_with_logits(
#         au_logits, au_targets,
#         weight=k,
#         pos_weight=pw,
#         reduction='mean'
#     )

# def parse_option():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/dfew7/16_16.yaml')
#     parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )
#     parser.add_argument('--gpu', default=[0, 1], type=int,help='GPU id to use.')
#     parser.add_argument('--output', type=str, default="DFEWAS2")
#     parser.add_argument('--resume', type=str)
#     parser.add_argument('--pretrained', type=str)
#     parser.add_argument('--only_test', action='store_true')
#     parser.add_argument('--batch-size', type=int)
#     parser.add_argument('--accumulation-steps', type=int)
#     parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
#     args = parser.parse_args()
#     config = get_config(args)
#     return args, config


# def main(config):
#     # load train and valid dataset
#     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

#     # load pretrained model
#     model, _ = AU_clip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
#                           device="cpu", jit=False,
#                           T=config.DATA.NUM_FRAMES,
#                           droppath=config.MODEL.DROP_PATH_RATE,
#                           use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#                           use_cache=config.MODEL.FIX_TEXT,
#                           logger=logger,
#                           N=config.DATA.NUM_DIVIDE,
#                           cfg=config
#                           )
#     model = model.cuda()

#     # training data augmentation
#     mixup_fn = None
#     if config.AUG.MIXUP > 0:
#         criterion = SoftTargetCrossEntropy()
#         criterion_soft = SoftTargetCrossEntropy()
#         mixup_fn = FixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
#                                     smoothing=config.AUG.LABEL_SMOOTH,
#                                     mixup_alpha=config.AUG.MIXUP,
#                                     fmix_alpha=config.AUG.CUTMIX,
#                                     switch_prob=config.AUG.MIXUP_SWITCH_PROB)
#     elif config.AUG.LABEL_SMOOTH > 0:
#         criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
#         criterion_soft = SoftTargetCrossEntropy()

#     else:
#         criterion = nn.CrossEntropyLoss()
#         criterion_soft = SoftTargetCrossEntropy()

#     optimizer = build_optimizer(config, model)
#     lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
#     model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
#                                                       find_unused_parameters=True)

#     start_epoch = 0
#     max_war_global, max_war_local, max_war_fuse = 0.0, 0.0, 0.0

#     # Track best epoch information (all from the same epoch with best WAR)
#     best_epoch = 0
#     best_war_global = 0.0
#     best_uar_global = 0.0
#     best_cls_acc_global = None
#     best_war_local = 0.0
#     best_uar_local = 0.0
#     best_cls_acc_local = None
#     best_war_fuse = 0.0
#     best_uar_fuse = 0.0
#     best_cls_acc_fuse = None
#     # retrain
#     if config.TRAIN.AUTO_RESUME:
#         resume_file_path = auto_resume_helper(config.OUTPUT)
#         if resume_file_path:
#             config.defrost()
#             config.MODEL.RESUME = resume_file_path
#             config.freeze()
#             logger.info(f'auto resuming from {resume_file_path}')
#         else:
#             logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
#     if config.MODEL.RESUME:
#         start_epoch, _ = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

#     # textual prompt
#     text_labels = generate_text(train_data)

#     # model test
#     if config.TEST.ONLY_TEST:
#         results = validate(val_loader, text_labels, model, config)
#         acc1, acc1_local, acc1_fuse = results[0], results[1], results[2]
#         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
#         logger.info(f"Accuracy of the network on the {len(val_data)} test videos: WAR={acc1:.1f}% UAR={uar_global:.1f}%")
#         return
#     print("Start training with AU-aware mixup")
#     #model train and valid
#     for epoch in range(start_epoch, config.TRAIN.EPOCHS):
#         train_loader.sampler.set_epoch(epoch)
#         # Use new AU-aware mixup training function
#         # au_loss_weight controls the balance: total_loss = (1-λ)*cls_loss + λ*au_loss
#         train_one_epoch_with_au_mixup(
#             epoch, model, criterion, optimizer, lr_scheduler,
#             train_loader, text_labels, config, mixup_fn, criterion_soft,
#             au_loss_weight=config.AU.LAMBDA if hasattr(config.AU, 'LAMBDA') else 0.5
#         )

#         # Global, local and fuse classification accuracy
#         results = validate(val_loader, text_labels, model, config)
#         acc_global, acc_local, acc_fuse = results[0], results[1], results[2]
#         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
#         war_global, war_local, war_fuse = results[6], results[7], results[8]
#         cls_acc_global, cls_acc_local, cls_acc_fuse = results[9], results[10], results[11]

#         # Get emotion class names
#         if config.DATA.DATASET.lower() == 'dfew':
#             emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
#         else:
#             emotion_names = [f'Class_{i}' for i in range(len(cls_acc_global))]

#         # Log detailed results for current epoch
#         logger.info(f"=" * 80)
#         logger.info(f"Epoch [{epoch}/{config.TRAIN.EPOCHS - 1}] Validation Results:")
#         logger.info(f"-" * 80)

#         # Global results
#         logger.info(f"Global Branch:")
#         logger.info(f"  WAR: {war_global:.2f}%  |  UAR: {uar_global:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_global):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         # Local results
#         logger.info(f"-" * 80)
#         logger.info(f"Local Branch:")
#         logger.info(f"  WAR: {war_local:.2f}%  |  UAR: {uar_local:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_local):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         # Fuse results
#         logger.info(f"-" * 80)
#         logger.info(f"Fuse Branch:")
#         logger.info(f"  WAR: {war_fuse:.2f}%  |  UAR: {uar_fuse:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_fuse):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")
#         logger.info(f"=" * 80)

#         is_best = war_global > max_war_global  # Use WAR (acc_global) as the criterion for best model

#         # Update best epoch info - only update when finding a new best WAR
#         if is_best:
#             best_epoch = epoch
#             # Update all metrics from the SAME epoch (the one with best WAR)
#             best_war_global = war_global
#             best_uar_global = uar_global
#             best_cls_acc_global = cls_acc_global.copy()
#             best_war_local = war_local
#             best_uar_local = uar_local
#             best_cls_acc_local = cls_acc_local.copy()
#             best_war_fuse = war_fuse
#             best_uar_fuse = uar_fuse
#             best_cls_acc_fuse = cls_acc_fuse.copy()

#             # Update max values - these are from the best epoch
#             max_war_global = war_global
#             max_war_local = war_local
#             max_war_fuse = war_fuse

#             logger.info(f">>> New best model found at epoch {epoch}! WAR: {war_global:.2f}% UAR: {uar_global:.2f}% <<<")

#         logger.info(f'Current Best Epoch: {best_epoch}')
#         logger.info(f'Best Global - WAR: {max_war_global:.2f}% | UAR: {best_uar_global:.2f}%')
#         logger.info(f'Best Local  - WAR: {max_war_local:.2f}% | UAR: {best_uar_local:.2f}%')
#         logger.info(f'Best Fuse   - WAR: {max_war_fuse:.2f}% | UAR: {best_uar_fuse:.2f}%')
#         # save model
#         if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
#             epoch_saving(config, epoch, model.module, max_war_global, optimizer, lr_scheduler, logger, config.OUTPUT,
#                          is_best)

#     # Print best epoch summary
#     logger.info(f"\n" + "=" * 80)
#     logger.info(f"TRAINING COMPLETED - BEST MODEL SUMMARY")
#     logger.info(f"=" * 80)
#     logger.info(f"Best Epoch: {best_epoch}")
#     logger.info(f"-" * 80)

#     # Get emotion class names
#     if config.DATA.DATASET.lower() == 'dfew':
#         emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
#     else:
#         emotion_names = [f'Class_{i}' for i in range(len(best_cls_acc_global)) if best_cls_acc_global is not None]

#     if best_cls_acc_global is not None:
#         logger.info(f"Global Branch (Best):")
#         logger.info(f"  WAR: {best_war_global:.2f}%  |  UAR: {best_uar_global:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_global):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         logger.info(f"-" * 80)
#         logger.info(f"Local Branch (Best):")
#         logger.info(f"  WAR: {best_war_local:.2f}%  |  UAR: {best_uar_local:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_local):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         logger.info(f"-" * 80)
#         logger.info(f"Fuse Branch (Best):")
#         logger.info(f"  WAR: {best_war_fuse:.2f}%  |  UAR: {best_uar_fuse:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_fuse):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")
#     logger.info(f"=" * 80 + "\n")

#     # validation after training
#     config.defrost()
#     config.TEST.NUM_CLIP = 4
#     config.TEST.NUM_CROP = 3
#     config.freeze()
#     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
#     results = validate(val_loader, text_labels, model, config)
#     acc = results[0]
#     uar_global = results[3]
#     logger.info(f"Final Accuracy: WAR={acc:.2f}% UAR={uar_global:.2f}%")


# def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft):
#     au_criterion = torch.nn.BCEWithLogitsLoss()
#     model.train()
#     optimizer.zero_grad()

#     num_steps = len(train_loader)
#     batch_time = AverageMeter()
#     tot_loss_meter = AverageMeter()

#     start = time.time()
#     end = time.time()
#     scaler = GradScaler()
#     texts = text_labels.cuda(non_blocking=True)

#     for idx, batch_data in enumerate(train_loader):
#         images = batch_data["imgs"].cuda(non_blocking=True)
#         label_id = batch_data["label"].cuda(non_blocking=True)
#         au_labels = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
#         label_id = label_id.reshape(-1)
#         images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

#         # Randomly decide whether to use mixup for this batch
#         # 50% chance to use mixup, 50% chance to use AU loss
#         import random
#         use_mixup_this_batch = (mixup_fn is not None) and (random.random() > 0.5)

#         if use_mixup_this_batch:
#             images, label_id = mixup_fn(images, label_id)

#         if texts.shape[0] == 1:
#             texts = texts.view(1, -1)
#         with autocast():
#             output_global, output_local, feat, au_logits = model(images, texts)
#             if epoch>13:
#                 total_loss = criterion(output_global, label_id)
#                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#             else:
#                 pre_output_local = torch.sum(
#                     output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
#                 # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
#                 total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
#                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#             if config.AU.ENABLED and mixup_fn is None:
#                 au_targets = batch_data["au"].cuda(non_blocking=True).float()
#                 au_loss = compute_au_loss_stage_b(
#                     au_logits, au_targets, label_id,
#                     posw_option=config.AU.POSW_OPTION
#                 )
#                 total_loss = total_loss + config.AU.LAMBDA * au_loss

#         if config.TRAIN.ACCUMULATION_STEPS == 1:
#             optimizer.zero_grad()
#         if config.TRAIN.OPT_LEVEL != 'O0':
#             scaler.scale(total_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             total_loss.backward()
#             optimizer.step()

#         if config.TRAIN.ACCUMULATION_STEPS > 1:
#             if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()
#                 lr_scheduler.step_update(epoch * num_steps + idx)
#         else:
#             scaler.step(optimizer)
#             scaler.update()
#             lr_scheduler.step_update(epoch * num_steps + idx)

#         torch.cuda.synchronize()

#         tot_loss_meter.update(total_loss.item(), len(label_id))
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if idx % config.PRINT_FREQ == 0:
#             lr = optimizer.param_groups[0]['lr']
#             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
#             etas = batch_time.avg * (num_steps - idx)
#             logger.info(
#                 f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
#                 f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
#                 f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
#                 f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
#                 f'mem {memory_used:.0f}MB')
#     epoch_time = time.time() - start
#     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# @torch.no_grad()
# def validate(val_loader, text_labels, model, config):
#     model.eval()

#     acc_global_meter, acc_local_meter, acc_fuse_meter = AverageMeter(), AverageMeter(), AverageMeter()

#     probility = []
#     video_pre_global = []
#     video_pre_local = []
#     video_pre_fuse = []
#     video_label = []
#     with torch.no_grad():
#         text_inputs = text_labels.cuda()
#         logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
#         for idx, batch_data in enumerate(val_loader):
#             _image = batch_data["imgs"]
#             label_id = batch_data["label"]
#             label_id = label_id.reshape(-1)

#             b, tn, c, h, w = _image.size()

#             t = config.DATA.NUM_FRAMES
#             n = tn // t
#             _image = _image.view(b, n, t, c, h, w)
#             tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             tot_similarity_local = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             for i in range(n):
#                 image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
#                 label_id = label_id.cuda(non_blocking=True)
#                 image_input = image.cuda(non_blocking=True)

#                 if config.TRAIN.OPT_LEVEL == 'O2':
#                     image_input = image_input.half()
#                 with autocast():
#                     output, output_local, feat, _ = model(image_input, text_inputs)
#                 if idx < 1:
#                     feature = feat
#                 else:
#                     feature = torch.cat((feature, feat), dim=0)

#                 pre_output_global = output.view(b, -1)
#                 pre_output_local = torch.sum(output_local.view(b, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES),dim=1).squeeze(dim=-1)

#                 similarity = pre_output_global.view(b, -1).softmax(dim=-1)
#                 tot_similarity += similarity

#                 similarity_local = pre_output_local.view(b, -1).softmax(dim=-1)
#                 tot_similarity_local += similarity_local.view(b, -1)

#             probility.extend(tot_similarity.data.cpu().numpy().copy())
#             values_global, indices_global = tot_similarity.topk(1, dim=-1)

#             values_local, indices_local = tot_similarity_local.topk(1, dim=-1)

#             fuse_similarity = tot_similarity + tot_similarity_local
#             values_fuse, indices_fuse = fuse_similarity.topk(1, dim=-1)

#             acc_global = 0
#             acc_local = 0
#             acc_fuse = 0
#             for i in range(b):
#                 video_pre_global.append(indices_global[i].data.cpu().numpy().copy())
#                 video_pre_local.append(indices_local[i].data.cpu().numpy().copy())
#                 video_pre_fuse.append(indices_fuse[i].data.cpu().numpy().copy())
#                 video_label.append(label_id[i].data.cpu().numpy().copy())
#                 if indices_global[i] == label_id[i]:
#                     acc_global += 1
#                 if indices_local[i] == label_id[i]:
#                     acc_local += 1
#                 if indices_fuse[i] == label_id[i]:
#                     acc_fuse += 1

#             acc_global_meter.update(float(acc_global) / b * 100, b)
#             acc_local_meter.update(float(acc_local) / b * 100, b)
#             acc_fuse_meter.update(float(acc_fuse) / b * 100, b)
#             if idx % config.PRINT_FREQ == 0:
#                 logger.info(
#                     f'Test: [{idx}/{len(val_loader)}]\t'
#                     f'Acc@1: {acc_global_meter.avg:.3f}\t'
#                     f'Acc@1: {acc_local_meter.avg:.3f}\t'
#                     f'Acc@1: {acc_fuse_meter.avg:.3f}\t'
#                 )
#     # confusion matrix
#     cf = confusion_matrix(video_label, video_pre_global)
#     np.set_printoptions(precision=4)
#     normalized_cm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
#     normalized_cm = normalized_cm * 100

#     cls_cnt = normalized_cm.sum(axis=1)
#     cls_hit = np.diag(normalized_cm)
#     # print(cf)
#     cls_acc = cls_hit / cls_cnt
#     cls_acc = np.around(cls_acc, 4)
#     cm = np.array(normalized_cm)
#     # save_path = 'AU-CLIP/results'
#     # if not os.path.exists(save_path):
#     #     os.makedirs(save_path)
#     # labels_name = ['hap', 'sad', 'neu', 'ang', 'sur', 'dis', 'fea']
#     # plot_confusion_matrix(cm, labels_name, 'AUCLIP', cls_acc)
#     #
#     # #t-SNE
#     # col = ['orange', 'purple', 'g', 'r', 'darkblue', 'chocolate', 'c']
#     # x_embed = TSNE(n_components=2, perplexity=100, n_iter=10000).fit_transform(feature.data.cpu())
#     # label = np.array(video_label)
#     # plt.figure(figsize=(6, 6))
#     # for i in range(7):
#     #     idxs = np.where(label == i)[0]
#     #     plt.scatter(x_embed[idxs, 0], x_embed[idxs, 1], color=col[i], s=6, label=labels_name[i])
#     # plt.legend(loc='upper left')
#     # plt.xticks(fontsize=13)
#     # plt.yticks(fontsize=13)
#     # plt.savefig(os.path.join(save_path, 'AUCLIP_TSNE.jpg'), format='jpg')
#     # plt.show()

#     logger.info(f'Global - Class-wise Accuracy: {cls_acc}')
#     upper = np.mean(np.max(cf, axis=1) / cls_cnt)
#     logger.info(f'Global - Upper bound: {upper}')
#     logger.info('Global - Evaluation is finished')
#     logger.info(f'Global - Class Accuracy (UAR): {np.mean(cls_acc) * 100:.2f}%')

#     cf_local = confusion_matrix(video_label, video_pre_local).astype(float)
#     cls_cnt_local = cf_local.sum(axis=1)
#     cls_hit_local = np.diag(cf_local)
#     # print(cf)
#     cls_acc_local = cls_hit_local / cls_cnt_local
#     cls_acc_local = np.around(cls_acc_local, 4)
#     logger.info(f'Local - Class-wise Accuracy: {cls_acc_local}')
#     upper = np.mean(np.max(cf_local, axis=1) / cls_cnt_local)
#     logger.info(f'Local - Upper bound: {upper}')
#     logger.info('Local - Evaluation is finished')
#     logger.info(f'Local - Class Accuracy (UAR): {np.mean(cls_acc_local) * 100:.2f}%')

#     cf_fuse = confusion_matrix(video_label, video_pre_fuse).astype(float)
#     cls_cnt_fuse = cf_fuse.sum(axis=1)
#     cls_hit_fuse = np.diag(cf_fuse)
#     # print(cf)
#     cls_acc_fuse = cls_hit_fuse / cls_cnt_fuse
#     cls_acc_fuse = np.around(cls_acc_fuse, 4)
#     logger.info(f'Fuse - Class-wise Accuracy: {cls_acc_fuse}')
#     upper = np.mean(np.max(cf_fuse, axis=1) / cls_cnt_fuse)
#     logger.info(f'Fuse - Upper bound: {upper}')
#     logger.info('Fuse - Evaluation is finished')
#     logger.info(f'Fuse - Class Accuracy (UAR): {np.mean(cls_acc_fuse) * 100:.2f}%')

#     acc_global_meter.sync()
#     acc_local_meter.sync()
#     acc_fuse_meter.sync()
#     logger.info(f' * Acc@1 {acc_global_meter.avg:.3f} Acc_loca@1 {acc_local_meter.avg:.3f}  Acc_fuse@1 {acc_fuse_meter.avg:.3f}')

#     # Calculate UAR (Unweighted Average Recall) and WAR (Weighted Average Recall)
#     uar_global = np.mean(cls_acc) * 100  # UAR for global
#     uar_local = np.mean(cls_acc_local) * 100  # UAR for local
#     uar_fuse = np.mean(cls_acc_fuse) * 100  # UAR for fuse
#     war_global = acc_global_meter.avg  # WAR is the same as overall accuracy
#     war_local = acc_local_meter.avg
#     war_fuse = acc_fuse_meter.avg

#     return (acc_global_meter.avg, acc_local_meter.avg, acc_fuse_meter.avg,
#             uar_global, uar_local, uar_fuse, war_global, war_local, war_fuse,
#             cls_acc, cls_acc_local, cls_acc_fuse,
#             video_label, video_pre_global, video_pre_local, video_pre_fuse)


# if __name__ == '__main__':
#     args, config = parse_option()

#     # 初始化分布式环境
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ['WORLD_SIZE'])
#         local_rank = int(os.environ['LOCAL_RANK'])  # 使用环境变量中的 LOCAL_RANK
#         print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
#     else:
#         rank = -1
#         world_size = -1
#         local_rank = args.local_rank  # 如果未设置环境变量，则使用命令行参数

#     # 设置当前 GPU 设备
#     torch.cuda.set_device(local_rank)

#     # 初始化分布式进程组
#     dist.init_process_group(
#         backend='nccl',
#         init_method='env://',
#         world_size=world_size,
#         rank=rank
#     )

#     # 确保所有进程同步
#     dist.barrier()

#     # 设置随机种子
#     seed = config.SEED + dist.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     cudnn.benchmark = True

#     # 创建输出目录
#     output_dir = Path(config.OUTPUT)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 创建日志记录器
#     logger = create_logger(output_dir=output_dir, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
#     logger.info(f"Working directory: {output_dir}")

#     # 保存配置文件（仅主进程执行）
#     if dist.get_rank() == 0:
#         logger.info(config)
#         shutil.copy(args.config, output_dir)

#     # 启动主训练逻辑
#     main(config)










# import os
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import argparse
# import datetime
# import shutil
# import time
# import numpy as np
# import random
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from pathlib import Path
# from sklearn.metrics import confusion_matrix
# from sklearn.manifold import TSNE
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
# from utils.optimizer import build_optimizer, build_scheduler
# from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, plot_confusion_matrix
# from utils.logger import create_logger
# from datasets.build import build_dataloader
# from datasets.blending import FixMixupBlending
# from utils.config import get_config
# from models import AU_clip
# import torch.nn.functional as F
# K_TABLE_DFEW = {
#     0: [0.37458275378581574, 0.37673577978763106, 0.41551698634677164, 0.20406588498862172,
#         0.7927884197377446, 0.8938573401735845, 0.26152800062149906, 0.8999413926399066,
#         0.9306407335956202, 0.8244182955003968, 0.2344366715203141, 0.35584905526535116,
#         0.2637127706449251, 0.1980659294341865, 0.8749731256118369, 0.5415473658691405,
#         0.4089852072091214, 0.29454183039387216],
#     1: [0.3563166809248465, 0.22025123080620432, 0.7883122756737404, 0.18552393229679323, 0.4330479071201498, 0.6512463687571445, 0.19562608246203084, 0.57228053429373, 0.3919835645137292, 0.5539327475091499, 0.26436050797140115, 0.3859719011102806, 0.2624203556065227, 0.19099832378533962, 0.5784889747902296, 0.5465085543388307, 0.4089852072091214, 0.31238729511297314],  # sad
#     2: [0.29634994040489937, 0.26249346997682327, 0.5349197805247116, 0.211357886947774, 0.24956316484226185, 0.48303878792569305, 0.1750459267403643, 0.4222786033590969, 0.2595297603319664, 0.4038707629862569, 0.2255822802606922, 0.3300116586908106, 0.21700897117154114, 0.18584396273490125, 0.4406741647399928, 0.4892561535112399, 0.4089852072091214, 0.24616267045793816],  # neutral
#     3: [0.2565426020973117, 0.22590006587965014, 0.7487900008174104, 0.24260938453624706, 0.3297348731445176, 0.6253054547733351, 0.2331741603449138, 0.6002659975024218, 0.25865288585343615, 0.4042681727124247, 0.25714836786909934, 0.4073109215604269, 0.22487756342746285, 0.1940789790636582, 0.7184070119769908, 0.6300831522957303, 0.4089852072091214, 0.22674904330132986],  # angry
#     4: [0.45840639969725916, 0.4348135864186588, 0.5807746998255159, 0.32444326825812997, 0.27370636575780566, 0.49552874169247974, 0.22029419679136286, 0.45675173141757053, 0.30048086399990936, 0.35356016873960344, 0.2340917014067623, 0.35356016873960344, 0.22418778976044848, 0.18774040869030503, 0.6204770513319532, 0.6347324688342945, 0.4089852072091214, 0.2326942881893769],  # surprise
#     5: [0.3669101379226728, 0.23849898584920362, 0.8556853679596857, 0.2093396686826401, 0.47998741884320045, 0.9116481111605065, 0.47437419451562296, 0.8176370658093514, 0.3690013619929339, 0.6122010138157353, 0.37313028939087656, 0.43845703949339593, 0.2712229960450541, 0.22790281312128532, 0.7303843653361992, 0.5736046807175323, 0.4089852072091214, 0.3157383637072392],  # disgust
#     6: [0.4668184913800416, 0.3236830048705496, 0.7092147921039609, 0.3237692334141919, 0.284442322105034, 0.4833791428919572, 0.19268179747966238, 0.49211626418097015, 0.29857493916455136, 0.40402139627383793, 0.3114334189399665, 0.42236998344528387, 0.2762397282204001, 0.20086024967771943, 0.6377967064381651, 0.6643128457350147, 0.4089852072091214, 0.22004117429592432],  # fear
# }
# K_SCALE = 5.0

# POSW_GLOBAL_DFEW = [8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674,
#                    4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794,
#                    7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384,
#                    12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372,
#                    101.28938906752411, 8.483380533611566]

# POSW_DISTINCT_DFEW = {
#     0: [4.648862512363996, 4.162923411588553, 1.83239825175909, 2.5131103421760863, 1.0496164371270917, 1.191954326288771, 15.430099793221252, 0.636697444899202, 1.0033104960263086, 0.7217365088935785, 3.69328950409615, 2.862698681095705, 6.9568094740508535, 4.616744014506562, 2.920370688175734, 5.462006293978289, 57.59313882654697, 4.170958066889254],  # happy
#     1: [5.6823671940967, 4.992658194508461, 0.5149984185907146, 2.6989371862870613, 3.1921004145555045, 1.6714365386873313, 19.07732186190161, 2.1056834274599465, 7.311081685767773, 1.9735191296108734, 5.039584577609662, 3.51384996900186, 8.076543333000897, 6.52494935714581, 3.628397792864953, 5.6926866933852995, 70.26898981989036, 5.224216933388045],  # sad
#     2: [6.4774986002239645, 5.6010812480692, 0.8710279064472912, 1.9167337801498792, 7.8117860530331145, 2.7351547887496284, 21.302160526041124, 3.19245786489297, 20.999073406774425, 2.574808023689626, 5.714757086292502, 3.906024704963953, 8.191199242945629, 5.206488904380156, 4.249791165053314, 7.363091976516634, 81.4053220208253, 4.963300960035722],  # neutral
#     3: [4.928812812224508, 4.114906832298137, 0.9234234234234234, 1.4543961558346765, 4.781224255883091, 2.435997871208089, 12.75189571440743, 1.44547134935305, 13.252185430463577, 3.1313061506565307, 3.579413266753674, 2.578767654819184, 5.92967542503864, 5.292631578947368, 2.6113572291582763, 4.433813627794237, 55.85311729482212, 4.549840112780662],  # angry
#     4: [7.116157728166966, 6.34866790582404, 0.9600900658968373, 0.761682850299846, 17.648977987421382, 5.163029358274876, 21.278938718008924, 6.183435536376713, 35.41059094397544, 6.815336463223788, 6.358355951919348, 4.08091030789826, 9.571078431372548, 7.167857450288371, 5.090243902439024, 8.104394549990404, 94.26706827309236, 6.122504128509233],  # surprise
#     5: [4.229214780600462, 4.09392575928009, 0.7474435655026047, 2.1834797891036906, 5.054144385026738, 1.6915304606240713, 10.307116104868914, 2.2381122631390777, 10.731865284974093, 3.210599721059972, 4.137266023823029, 3.012848914488259, 5.983037779491133, 6.148382004735596, 3.6374807987711213, 5.387165021156559, 27.391849529780565, 3.2964895635673623],  # disgust
#     6: [5.96072648535643, 5.154494891980657, 0.7569813845450734, 1.5509251975236857, 7.060588375159415, 2.779909786630176, 17.84796563052818, 2.8554517069539505, 15.333362533397574, 3.2621352565348083, 4.923954312221004, 3.5370230679384855, 7.644713355124371, 5.762544656620061, 3.9785987023043443, 6.670773851153989, 57.87385538364383, 5.229860670252932],  # fear
# }

# # minor strategy: define which classes are "major"
# MAJOR_CLASSES_DFEW = {0, 1, 2, 3}

# def compute_au_loss_stage_b(au_logits, au_targets, labels, posw_option='global'):
#     """
#     au_logits:  [B,18] float
#     au_targets: [B,18] float {0,1}
#     labels:     [B] long, emotion class ids (0..C-1)
#     """
#     device = au_logits.device
#     B = labels.shape[0]
#     # pos_weight: [B,18]
#     if config.DATA.DATASET=='DFEW':
#         # knowledge weight: [B,18]
#         k = torch.stack(
#             [torch.tensor(K_TABLE_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#             dim=0
#         ) * K_SCALE
#         if posw_option == 'global':
#             pw = torch.tensor(POSW_GLOBAL_DFEW, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
#         elif posw_option == 'distinct':
#             pw = torch.stack(
#                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#                 dim=0
#             )
#         elif posw_option == 'minor':
#             pw = torch.stack(
#                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#                 dim=0
#             )
#             # overwrite major classes with 1s
#             mask_major = torch.tensor([int(c) in MAJOR_CLASSES_DFEW for c in labels], device=device, dtype=torch.bool)
#             if mask_major.any():
#                 pw = pw.clone()
#                 pw[mask_major] = 1.0
#         else:
#             raise ValueError(f'Unknown posw_option: {posw_option}')

#     # weighted BCE with logits
#     return F.binary_cross_entropy_with_logits(
#         au_logits, au_targets,
#         weight=k,
#         pos_weight=pw,
#         reduction='mean'
#     )

# def parse_option():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/dfew7/16_16.yaml')
#     parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )
#     parser.add_argument('--gpu', default=[0, 1], type=int,help='GPU id to use.')
#     parser.add_argument('--output', type=str, default="DFEWAS2")
#     parser.add_argument('--resume', type=str)
#     parser.add_argument('--pretrained', type=str)
#     parser.add_argument('--only_test', action='store_true')
#     parser.add_argument('--batch-size', type=int)
#     parser.add_argument('--accumulation-steps', type=int)
#     parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
#     args = parser.parse_args()
#     config = get_config(args)
#     return args, config


# def main(config):
#     # load train and valid dataset
#     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

#     # load pretrained model
#     model, _ = AU_clip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
#                           device="cpu", jit=False,
#                           T=config.DATA.NUM_FRAMES,
#                           droppath=config.MODEL.DROP_PATH_RATE,
#                           use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#                           use_cache=config.MODEL.FIX_TEXT,
#                           logger=logger,
#                           N=config.DATA.NUM_DIVIDE,
#                           cfg=config
#                           )
#     model = model.cuda()

#     # training data augmentation
#     mixup_fn = None
#     if config.AUG.MIXUP > 0:
#         criterion = SoftTargetCrossEntropy()
#         criterion_soft = SoftTargetCrossEntropy()
#         mixup_fn = FixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
#                                     smoothing=config.AUG.LABEL_SMOOTH,
#                                     mixup_alpha=config.AUG.MIXUP,
#                                     fmix_alpha=config.AUG.CUTMIX,
#                                     switch_prob=config.AUG.MIXUP_SWITCH_PROB)
#     elif config.AUG.LABEL_SMOOTH > 0:
#         criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
#         criterion_soft = SoftTargetCrossEntropy()

#     else:
#         criterion = nn.CrossEntropyLoss()
#         criterion_soft = SoftTargetCrossEntropy()

#     optimizer = build_optimizer(config, model)
#     lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
#     model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
#                                                       find_unused_parameters=True)

#     start_epoch = 0
#     max_war_global, max_war_local, max_war_fuse = 0.0, 0.0, 0.0

#     # Track best epoch information (all from the same epoch with best WAR)
#     best_epoch = 0
#     best_war_global = 0.0
#     best_uar_global = 0.0
#     best_cls_acc_global = None
#     best_war_local = 0.0
#     best_uar_local = 0.0
#     best_cls_acc_local = None
#     best_war_fuse = 0.0
#     best_uar_fuse = 0.0
#     best_cls_acc_fuse = None
#     # retrain
#     if config.TRAIN.AUTO_RESUME:
#         resume_file_path = auto_resume_helper(config.OUTPUT)
#         if resume_file_path:
#             config.defrost()
#             config.MODEL.RESUME = resume_file_path
#             config.freeze()
#             logger.info(f'auto resuming from {resume_file_path}')
#         else:
#             logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
#     if config.MODEL.RESUME:
#         start_epoch, _ = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

#     # textual prompt
#     text_labels = generate_text(train_data)

#     # model test
#     if config.TEST.ONLY_TEST:
#         results = validate(val_loader, text_labels, model, config)
#         acc1, acc1_local, acc1_fuse = results[0], results[1], results[2]
#         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
#         logger.info(f"Accuracy of the network on the {len(val_data)} test videos: WAR={acc1:.1f}% UAR={uar_global:.1f}%")
#         return
#     print("Start training")
#     #model train and valid
#     for epoch in range(start_epoch, config.TRAIN.EPOCHS):
#         train_loader.sampler.set_epoch(epoch)
#         train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft)

#         # Global, local and fuse classification accuracy
#         results = validate(val_loader, text_labels, model, config)
#         acc_global, acc_local, acc_fuse = results[0], results[1], results[2]
#         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
#         war_global, war_local, war_fuse = results[6], results[7], results[8]
#         cls_acc_global, cls_acc_local, cls_acc_fuse = results[9], results[10], results[11]

#         # Get emotion class names
#         if config.DATA.DATASET.lower() == 'dfew':
#             emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
#         else:
#             emotion_names = [f'Class_{i}' for i in range(len(cls_acc_global))]

#         # Log detailed results for current epoch
#         logger.info(f"=" * 80)
#         logger.info(f"Epoch [{epoch}/{config.TRAIN.EPOCHS - 1}] Validation Results:")
#         logger.info(f"-" * 80)

#         # Global results
#         logger.info(f"Global Branch:")
#         logger.info(f"  WAR: {war_global:.2f}%  |  UAR: {uar_global:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_global):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         # Local results
#         logger.info(f"-" * 80)
#         logger.info(f"Local Branch:")
#         logger.info(f"  WAR: {war_local:.2f}%  |  UAR: {uar_local:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_local):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         # Fuse results
#         logger.info(f"-" * 80)
#         logger.info(f"Fuse Branch:")
#         logger.info(f"  WAR: {war_fuse:.2f}%  |  UAR: {uar_fuse:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_fuse):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")
#         logger.info(f"=" * 80)

#         is_best = war_global > max_war_global  # Use WAR (acc_global) as the criterion for best model

#         # Update best epoch info - only update when finding a new best WAR
#         if is_best:
#             best_epoch = epoch
#             # Update all metrics from the SAME epoch (the one with best WAR)
#             best_war_global = war_global
#             best_uar_global = uar_global
#             best_cls_acc_global = cls_acc_global.copy()
#             best_war_local = war_local
#             best_uar_local = uar_local
#             best_cls_acc_local = cls_acc_local.copy()
#             best_war_fuse = war_fuse
#             best_uar_fuse = uar_fuse
#             best_cls_acc_fuse = cls_acc_fuse.copy()

#             # Update max values - these are from the best epoch
#             max_war_global = war_global
#             max_war_local = war_local
#             max_war_fuse = war_fuse

#             logger.info(f">>> New best model found at epoch {epoch}! WAR: {war_global:.2f}% UAR: {uar_global:.2f}% <<<")

#         logger.info(f'Current Best Epoch: {best_epoch}')
#         logger.info(f'Best Global - WAR: {max_war_global:.2f}% | UAR: {best_uar_global:.2f}%')
#         logger.info(f'Best Local  - WAR: {max_war_local:.2f}% | UAR: {best_uar_local:.2f}%')
#         logger.info(f'Best Fuse   - WAR: {max_war_fuse:.2f}% | UAR: {best_uar_fuse:.2f}%')
#         # save model
#         if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
#             epoch_saving(config, epoch, model.module, max_war_global, optimizer, lr_scheduler, logger, config.OUTPUT,
#                          is_best)

#     # Print best epoch summary
#     logger.info(f"\n" + "=" * 80)
#     logger.info(f"TRAINING COMPLETED - BEST MODEL SUMMARY")
#     logger.info(f"=" * 80)
#     logger.info(f"Best Epoch: {best_epoch}")
#     logger.info(f"-" * 80)

#     # Get emotion class names
#     if config.DATA.DATASET.lower() == 'dfew':
#         emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
#     else:
#         emotion_names = [f'Class_{i}' for i in range(len(best_cls_acc_global)) if best_cls_acc_global is not None]

#     if best_cls_acc_global is not None:
#         logger.info(f"Global Branch (Best):")
#         logger.info(f"  WAR: {best_war_global:.2f}%  |  UAR: {best_uar_global:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_global):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         logger.info(f"-" * 80)
#         logger.info(f"Local Branch (Best):")
#         logger.info(f"  WAR: {best_war_local:.2f}%  |  UAR: {best_uar_local:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_local):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         logger.info(f"-" * 80)
#         logger.info(f"Fuse Branch (Best):")
#         logger.info(f"  WAR: {best_war_fuse:.2f}%  |  UAR: {best_uar_fuse:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_fuse):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")
#     logger.info(f"=" * 80 + "\n")

#     # validation after training
#     config.defrost()
#     config.TEST.NUM_CLIP = 4
#     config.TEST.NUM_CROP = 3
#     config.freeze()
#     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
#     results = validate(val_loader, text_labels, model, config)
#     acc = results[0]
#     uar_global = results[3]
#     logger.info(f"Final Accuracy: WAR={acc:.2f}% UAR={uar_global:.2f}%")


# def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft):
#     au_criterion = torch.nn.BCEWithLogitsLoss()
#     model.train()
#     optimizer.zero_grad()

#     num_steps = len(train_loader)
#     batch_time = AverageMeter()
#     tot_loss_meter = AverageMeter()

#     start = time.time()
#     end = time.time()
#     scaler = GradScaler()
#     texts = text_labels.cuda(non_blocking=True)

#     for idx, batch_data in enumerate(train_loader):
#         images = batch_data["imgs"].cuda(non_blocking=True)
#         label_id = batch_data["label"].cuda(non_blocking=True)
#         au_labels = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
#         label_id = label_id.reshape(-1)
#         images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

#         if mixup_fn is not None:
#             images, label_id = mixup_fn(images, label_id)

#         if texts.shape[0] == 1:
#             texts = texts.view(1, -1)
#         with autocast():
#             output_global, output_local, feat, au_logits = model(images, texts)
#             if epoch>13:
#                 total_loss = criterion(output_global, label_id)
#                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#             else:
#                 pre_output_local = torch.sum(
#                     output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
#                 # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
#                 total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
#                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#             if config.AU.ENABLED and mixup_fn is None:
#                 au_targets = batch_data["au"].cuda(non_blocking=True).float()
#                 au_loss = compute_au_loss_stage_b(
#                     au_logits, au_targets, label_id,
#                     posw_option=config.AU.POSW_OPTION
#                 )
#                 total_loss = total_loss + config.AU.LAMBDA * au_loss
#                 #total_loss = (1-config.AU.LAMBDA) * total_loss + config.AU.LAMBDA * au_loss

#         if config.TRAIN.ACCUMULATION_STEPS == 1:
#             optimizer.zero_grad()
#         if config.TRAIN.OPT_LEVEL != 'O0':
#             scaler.scale(total_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             total_loss.backward()
#             optimizer.step()

#         if config.TRAIN.ACCUMULATION_STEPS > 1:
#             if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()
#                 lr_scheduler.step_update(epoch * num_steps + idx)
#         else:
#             scaler.step(optimizer)
#             scaler.update()
#             lr_scheduler.step_update(epoch * num_steps + idx)

#         torch.cuda.synchronize()

#         tot_loss_meter.update(total_loss.item(), len(label_id))
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if idx % config.PRINT_FREQ == 0:
#             lr = optimizer.param_groups[0]['lr']
#             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
#             etas = batch_time.avg * (num_steps - idx)
#             logger.info(
#                 f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
#                 f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
#                 f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
#                 f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
#                 f'mem {memory_used:.0f}MB')
#     epoch_time = time.time() - start
#     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# @torch.no_grad()
# def validate(val_loader, text_labels, model, config):
#     model.eval()

#     acc_global_meter, acc_local_meter, acc_fuse_meter = AverageMeter(), AverageMeter(), AverageMeter()

#     probility = []
#     video_pre_global = []
#     video_pre_local = []
#     video_pre_fuse = []
#     video_label = []
#     with torch.no_grad():
#         text_inputs = text_labels.cuda()
#         logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
#         for idx, batch_data in enumerate(val_loader):
#             _image = batch_data["imgs"]
#             label_id = batch_data["label"]
#             label_id = label_id.reshape(-1)

#             b, tn, c, h, w = _image.size()

#             t = config.DATA.NUM_FRAMES
#             n = tn // t
#             _image = _image.view(b, n, t, c, h, w)
#             tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             tot_similarity_local = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             for i in range(n):
#                 image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
#                 label_id = label_id.cuda(non_blocking=True)
#                 image_input = image.cuda(non_blocking=True)

#                 if config.TRAIN.OPT_LEVEL == 'O2':
#                     image_input = image_input.half()
#                 with autocast():
#                     output, output_local, feat, _ = model(image_input, text_inputs)
#                 if idx < 1:
#                     feature = feat
#                 else:
#                     feature = torch.cat((feature, feat), dim=0)

#                 pre_output_global = output.view(b, -1)
#                 pre_output_local = torch.sum(output_local.view(b, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES),dim=1).squeeze(dim=-1)

#                 similarity = pre_output_global.view(b, -1).softmax(dim=-1)
#                 tot_similarity += similarity

#                 similarity_local = pre_output_local.view(b, -1).softmax(dim=-1)
#                 tot_similarity_local += similarity_local.view(b, -1)

#             probility.extend(tot_similarity.data.cpu().numpy().copy())
#             values_global, indices_global = tot_similarity.topk(1, dim=-1)

#             values_local, indices_local = tot_similarity_local.topk(1, dim=-1)

#             fuse_similarity = tot_similarity + tot_similarity_local
#             values_fuse, indices_fuse = fuse_similarity.topk(1, dim=-1)

#             acc_global = 0
#             acc_local = 0
#             acc_fuse = 0
#             for i in range(b):
#                 video_pre_global.append(indices_global[i].data.cpu().numpy().copy())
#                 video_pre_local.append(indices_local[i].data.cpu().numpy().copy())
#                 video_pre_fuse.append(indices_fuse[i].data.cpu().numpy().copy())
#                 video_label.append(label_id[i].data.cpu().numpy().copy())
#                 if indices_global[i] == label_id[i]:
#                     acc_global += 1
#                 if indices_local[i] == label_id[i]:
#                     acc_local += 1
#                 if indices_fuse[i] == label_id[i]:
#                     acc_fuse += 1

#             acc_global_meter.update(float(acc_global) / b * 100, b)
#             acc_local_meter.update(float(acc_local) / b * 100, b)
#             acc_fuse_meter.update(float(acc_fuse) / b * 100, b)
#             if idx % config.PRINT_FREQ == 0:
#                 logger.info(
#                     f'Test: [{idx}/{len(val_loader)}]\t'
#                     f'Acc@1: {acc_global_meter.avg:.3f}\t'
#                     f'Acc@1: {acc_local_meter.avg:.3f}\t'
#                     f'Acc@1: {acc_fuse_meter.avg:.3f}\t'
#                 )
#     # confusion matrix
#     cf = confusion_matrix(video_label, video_pre_global)
#     np.set_printoptions(precision=4)
#     normalized_cm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
#     normalized_cm = normalized_cm * 100

#     cls_cnt = normalized_cm.sum(axis=1)
#     cls_hit = np.diag(normalized_cm)
#     # print(cf)
#     cls_acc = cls_hit / cls_cnt
#     cls_acc = np.around(cls_acc, 4)
#     cm = np.array(normalized_cm)
#     # save_path = 'AU-CLIP/results'
#     # if not os.path.exists(save_path):
#     #     os.makedirs(save_path)
#     # labels_name = ['hap', 'sad', 'neu', 'ang', 'sur', 'dis', 'fea']
#     # plot_confusion_matrix(cm, labels_name, 'AUCLIP', cls_acc)
#     #
#     # #t-SNE
#     # col = ['orange', 'purple', 'g', 'r', 'darkblue', 'chocolate', 'c']
#     # x_embed = TSNE(n_components=2, perplexity=100, n_iter=10000).fit_transform(feature.data.cpu())
#     # label = np.array(video_label)
#     # plt.figure(figsize=(6, 6))
#     # for i in range(7):
#     #     idxs = np.where(label == i)[0]
#     #     plt.scatter(x_embed[idxs, 0], x_embed[idxs, 1], color=col[i], s=6, label=labels_name[i])
#     # plt.legend(loc='upper left')
#     # plt.xticks(fontsize=13)
#     # plt.yticks(fontsize=13)
#     # plt.savefig(os.path.join(save_path, 'AUCLIP_TSNE.jpg'), format='jpg')
#     # plt.show()

#     logger.info(f'Global - Class-wise Accuracy: {cls_acc}')
#     upper = np.mean(np.max(cf, axis=1) / cls_cnt)
#     logger.info(f'Global - Upper bound: {upper}')
#     logger.info('Global - Evaluation is finished')
#     logger.info(f'Global - Class Accuracy (UAR): {np.mean(cls_acc) * 100:.2f}%')

#     cf_local = confusion_matrix(video_label, video_pre_local).astype(float)
#     cls_cnt_local = cf_local.sum(axis=1)
#     cls_hit_local = np.diag(cf_local)
#     # print(cf)
#     cls_acc_local = cls_hit_local / cls_cnt_local
#     cls_acc_local = np.around(cls_acc_local, 4)
#     logger.info(f'Local - Class-wise Accuracy: {cls_acc_local}')
#     upper = np.mean(np.max(cf_local, axis=1) / cls_cnt_local)
#     logger.info(f'Local - Upper bound: {upper}')
#     logger.info('Local - Evaluation is finished')
#     logger.info(f'Local - Class Accuracy (UAR): {np.mean(cls_acc_local) * 100:.2f}%')

#     cf_fuse = confusion_matrix(video_label, video_pre_fuse).astype(float)
#     cls_cnt_fuse = cf_fuse.sum(axis=1)
#     cls_hit_fuse = np.diag(cf_fuse)
#     # print(cf)
#     cls_acc_fuse = cls_hit_fuse / cls_cnt_fuse
#     cls_acc_fuse = np.around(cls_acc_fuse, 4)
#     logger.info(f'Fuse - Class-wise Accuracy: {cls_acc_fuse}')
#     upper = np.mean(np.max(cf_fuse, axis=1) / cls_cnt_fuse)
#     logger.info(f'Fuse - Upper bound: {upper}')
#     logger.info('Fuse - Evaluation is finished')
#     logger.info(f'Fuse - Class Accuracy (UAR): {np.mean(cls_acc_fuse) * 100:.2f}%')

#     acc_global_meter.sync()
#     acc_local_meter.sync()
#     acc_fuse_meter.sync()
#     logger.info(f' * Acc@1 {acc_global_meter.avg:.3f} Acc_loca@1 {acc_local_meter.avg:.3f}  Acc_fuse@1 {acc_fuse_meter.avg:.3f}')

#     # Calculate UAR (Unweighted Average Recall) and WAR (Weighted Average Recall)
#     uar_global = np.mean(cls_acc) * 100  # UAR for global
#     uar_local = np.mean(cls_acc_local) * 100  # UAR for local
#     uar_fuse = np.mean(cls_acc_fuse) * 100  # UAR for fuse
#     war_global = acc_global_meter.avg  # WAR is the same as overall accuracy
#     war_local = acc_local_meter.avg
#     war_fuse = acc_fuse_meter.avg

#     return (acc_global_meter.avg, acc_local_meter.avg, acc_fuse_meter.avg,
#             uar_global, uar_local, uar_fuse, war_global, war_local, war_fuse,
#             cls_acc, cls_acc_local, cls_acc_fuse,
#             video_label, video_pre_global, video_pre_local, video_pre_fuse)


# if __name__ == '__main__':
#     args, config = parse_option()

#     # 初始化分布式环境
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ['WORLD_SIZE'])
#         local_rank = int(os.environ['LOCAL_RANK'])  # 使用环境变量中的 LOCAL_RANK
#         print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
#     else:
#         rank = -1
#         world_size = -1
#         local_rank = args.local_rank  # 如果未设置环境变量，则使用命令行参数

#     # 设置当前 GPU 设备
#     torch.cuda.set_device(local_rank)

#     # 初始化分布式进程组
#     dist.init_process_group(
#         backend='nccl',
#         init_method='env://',
#         world_size=world_size,
#         rank=rank
#     )

#     # 确保所有进程同步
#     dist.barrier()

#     # 设置随机种子
#     seed = config.SEED + dist.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     cudnn.benchmark = True

#     # 创建输出目录
#     output_dir = Path(config.OUTPUT)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 创建日志记录器
#     logger = create_logger(output_dir=output_dir, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
#     logger.info(f"Working directory: {output_dir}")

#     # 保存配置文件（仅主进程执行）
#     if dist.get_rank() == 0:
#         logger.info(config)
#         shutil.copy(args.config, output_dir)

#     # 启动主训练逻辑
#     main(config)
























































# import os
# import copy  # 【新增】导入 copy 模块用于深拷贝模型权重
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import argparse
# import datetime
# import shutil
# import time
# import numpy as np
# import random
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from pathlib import Path
# from sklearn.metrics import confusion_matrix
# from sklearn.manifold import TSNE
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
# from utils.optimizer import build_optimizer, build_scheduler
# from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, plot_confusion_matrix
# from utils.logger import create_logger
# from datasets.build import build_dataloader
# from datasets.blending import FixMixupBlending
# from utils.config import get_config
# from models import AU_clip
# import torch.nn.functional as F

# K_TABLE_DFEW = {
#     0: [0.37458275378581574, 0.37673577978763106, 0.41551698634677164, 0.20406588498862172,
#         0.7927884197377446, 0.8938573401735845, 0.26152800062149906, 0.8999413926399066,
#         0.9306407335956202, 0.8244182955003968, 0.2344366715203141, 0.35584905526535116,
#         0.2637127706449251, 0.1980659294341865, 0.8749731256118369, 0.5415473658691405,
#         0.4089852072091214, 0.29454183039387216],
#     1: [0.3563166809248465, 0.22025123080620432, 0.7883122756737404, 0.18552393229679323, 0.4330479071201498, 0.6512463687571445, 0.19562608246203084, 0.57228053429373, 0.3919835645137292, 0.5539327475091499, 0.26436050797140115, 0.3859719011102806, 0.2624203556065227, 0.19099832378533962, 0.5784889747902296, 0.5465085543388307, 0.4089852072091214, 0.31238729511297314],  # sad
#     2: [0.29634994040489937, 0.26249346997682327, 0.5349197805247116, 0.211357886947774, 0.24956316484226185, 0.48303878792569305, 0.1750459267403643, 0.4222786033590969, 0.2595297603319664, 0.4038707629862569, 0.2255822802606922, 0.3300116586908106, 0.21700897117154114, 0.18584396273490125, 0.4406741647399928, 0.4892561535112399, 0.4089852072091214, 0.24616267045793816],  # neutral
#     3: [0.2565426020973117, 0.22590006587965014, 0.7487900008174104, 0.24260938453624706, 0.3297348731445176, 0.6253054547733351, 0.2331741603449138, 0.6002659975024218, 0.25865288585343615, 0.4042681727124247, 0.25714836786909934, 0.4073109215604269, 0.22487756342746285, 0.1940789790636582, 0.7184070119769908, 0.6300831522957303, 0.4089852072091214, 0.22674904330132986],  # angry
#     4: [0.45840639969725916, 0.4348135864186588, 0.5807746998255159, 0.32444326825812997, 0.27370636575780566, 0.49552874169247974, 0.22029419679136286, 0.45675173141757053, 0.30048086399990936, 0.35356016873960344, 0.2340917014067623, 0.35356016873960344, 0.22418778976044848, 0.18774040869030503, 0.6204770513319532, 0.6347324688342945, 0.4089852072091214, 0.2326942881893769],  # surprise
#     5: [0.3669101379226728, 0.23849898584920362, 0.8556853679596857, 0.2093396686826401, 0.47998741884320045, 0.9116481111605065, 0.47437419451562296, 0.8176370658093514, 0.3690013619929339, 0.6122010138157353, 0.37313028939087656, 0.43845703949339593, 0.2712229960450541, 0.22790281312128532, 0.7303843653361992, 0.5736046807175323, 0.4089852072091214, 0.3157383637072392],  # disgust
#     6: [0.4668184913800416, 0.3236830048705496, 0.7092147921039609, 0.3237692334141919, 0.284442322105034, 0.4833791428919572, 0.19268179747966238, 0.49211626418097015, 0.29857493916455136, 0.40402139627383793, 0.3114334189399665, 0.42236998344528387, 0.2762397282204001, 0.20086024967771943, 0.6377967064381651, 0.6643128457350147, 0.4089852072091214, 0.22004117429592432],  # fear
# }
# K_SCALE = 5.0

# POSW_GLOBAL_DFEW = [8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674,
#                    4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794,
#                    7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384,
#                    12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372,
#                    101.28938906752411, 8.483380533611566]

# POSW_DISTINCT_DFEW = {
#     0: [4.648862512363996, 4.162923411588553, 1.83239825175909, 2.5131103421760863, 1.0496164371270917, 1.191954326288771, 15.430099793221252, 0.636697444899202, 1.0033104960263086, 0.7217365088935785, 3.69328950409615, 2.862698681095705, 6.9568094740508535, 4.616744014506562, 2.920370688175734, 5.462006293978289, 57.59313882654697, 4.170958066889254],  # happy
#     1: [5.6823671940967, 4.992658194508461, 0.5149984185907146, 2.6989371862870613, 3.1921004145555045, 1.6714365386873313, 19.07732186190161, 2.1056834274599465, 7.311081685767773, 1.9735191296108734, 5.039584577609662, 3.51384996900186, 8.076543333000897, 6.52494935714581, 3.628397792864953, 5.6926866933852995, 70.26898981989036, 5.224216933388045],  # sad
#     2: [6.4774986002239645, 5.6010812480692, 0.8710279064472912, 1.9167337801498792, 7.8117860530331145, 2.7351547887496284, 21.302160526041124, 3.19245786489297, 20.999073406774425, 2.574808023689626, 5.714757086292502, 3.906024704963953, 8.191199242945629, 5.206488904380156, 4.249791165053314, 7.363091976516634, 81.4053220208253, 4.963300960035722],  # neutral
#     3: [4.928812812224508, 4.114906832298137, 0.9234234234234234, 1.4543961558346765, 4.781224255883091, 2.435997871208089, 12.75189571440743, 1.44547134935305, 13.252185430463577, 3.1313061506565307, 3.579413266753674, 2.578767654819184, 5.92967542503864, 5.292631578947368, 2.6113572291582763, 4.433813627794237, 55.85311729482212, 4.549840112780662],  # angry
#     4: [7.116157728166966, 6.34866790582404, 0.9600900658968373, 0.761682850299846, 17.648977987421382, 5.163029358274876, 21.278938718008924, 6.183435536376713, 35.41059094397544, 6.815336463223788, 6.358355951919348, 4.08091030789826, 9.571078431372548, 7.167857450288371, 5.090243902439024, 8.104394549990404, 94.26706827309236, 6.122504128509233],  # surprise
#     5: [4.229214780600462, 4.09392575928009, 0.7474435655026047, 2.1834797891036906, 5.054144385026738, 1.6915304606240713, 10.307116104868914, 2.2381122631390777, 10.731865284974093, 3.210599721059972, 4.137266023823029, 3.012848914488259, 5.983037779491133, 6.148382004735596, 3.6374807987711213, 5.387165021156559, 27.391849529780565, 3.2964895635673623],  # disgust
#     6: [5.96072648535643, 5.154494891980657, 0.7569813845450734, 1.5509251975236857, 7.060588375159415, 2.779909786630176, 17.84796563052818, 2.8554517069539505, 15.333362533397574, 3.2621352565348083, 4.923954312221004, 3.5370230679384855, 7.644713355124371, 5.762544656620061, 3.9785987023043443, 6.670773851153989, 57.87385538364383, 5.229860670252932],  # fear
# }

# # minor strategy: define which classes are "major"
# MAJOR_CLASSES_DFEW = {0, 1, 2, 3}

# def compute_au_loss_stage_b(au_logits, au_targets, labels, posw_option='global'):
#     """
#     au_logits:  [B,18] float
#     au_targets: [B,18] float {0,1}
#     labels:     [B] long, emotion class ids (0..C-1)
#     """
#     device = au_logits.device
#     B = labels.shape[0]
#     # pos_weight: [B,18]
#     if config.DATA.DATASET=='DFEW':
#         # knowledge weight: [B,18]
#         k = torch.stack(
#             [torch.tensor(K_TABLE_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#             dim=0
#         ) * K_SCALE
#         if posw_option == 'global':
#             pw = torch.tensor(POSW_GLOBAL_DFEW, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
#         elif posw_option == 'distinct':
#             pw = torch.stack(
#                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#                 dim=0
#             )
#         elif posw_option == 'minor':
#             pw = torch.stack(
#                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#                 dim=0
#             )
#             # overwrite major classes with 1s
#             mask_major = torch.tensor([int(c) in MAJOR_CLASSES_DFEW for c in labels], device=device, dtype=torch.bool)
#             if mask_major.any():
#                 pw = pw.clone()
#                 pw[mask_major] = 1.0
#         else:
#             raise ValueError(f'Unknown posw_option: {posw_option}')

#     # weighted BCE with logits
#     return F.binary_cross_entropy_with_logits(
#         au_logits, au_targets,
#         weight=k,
#         pos_weight=pw,
#         reduction='mean'
#     )

# def parse_option():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/dfew7/16_16.yaml')
#     parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )
#     parser.add_argument('--gpu', default=[0, 1], type=int,help='GPU id to use.')
#     parser.add_argument('--output', type=str, default="DFEWAS2")
#     parser.add_argument('--resume', type=str)
#     parser.add_argument('--pretrained', type=str)
#     parser.add_argument('--only_test', action='store_true')
#     parser.add_argument('--batch-size', type=int)
#     parser.add_argument('--accumulation-steps', type=int)
#     parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
#     args = parser.parse_args()
#     config = get_config(args)
#     return args, config


# def main(config):
#     # load train and valid dataset
#     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

#     # load pretrained model
#     model, _ = AU_clip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
#                           device="cpu", jit=False,
#                           T=config.DATA.NUM_FRAMES,
#                           droppath=config.MODEL.DROP_PATH_RATE,
#                           use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#                           use_cache=config.MODEL.FIX_TEXT,
#                           logger=logger,
#                           N=config.DATA.NUM_DIVIDE,
#                           cfg=config
#                           )
#     model = model.cuda()

#     # training data augmentation
#     mixup_fn = None
#     if config.AUG.MIXUP > 0:
#         criterion = SoftTargetCrossEntropy()
#         criterion_soft = SoftTargetCrossEntropy()
#         mixup_fn = FixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
#                                     smoothing=config.AUG.LABEL_SMOOTH,
#                                     mixup_alpha=config.AUG.MIXUP,
#                                     fmix_alpha=config.AUG.CUTMIX,
#                                     switch_prob=config.AUG.MIXUP_SWITCH_PROB)
#     elif config.AUG.LABEL_SMOOTH > 0:
#         criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
#         criterion_soft = SoftTargetCrossEntropy()

#     else:
#         criterion = nn.CrossEntropyLoss()
#         criterion_soft = SoftTargetCrossEntropy()

#     optimizer = build_optimizer(config, model)
#     lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
#     model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
#                                                       find_unused_parameters=True)

#     start_epoch = 0
#     max_war_global, max_war_local, max_war_fuse = 0.0, 0.0, 0.0

#     # Track best epoch information (all from the same epoch with best WAR)
#     best_epoch = 0
#     best_war_global = 0.0
#     best_uar_global = 0.0
#     best_cls_acc_global = None
#     best_war_local = 0.0
#     best_uar_local = 0.0
#     best_cls_acc_local = None
#     best_war_fuse = 0.0
#     best_uar_fuse = 0.0
#     best_cls_acc_fuse = None
    
#     # 【新增】初始化保存最佳模型权重的变量
#     best_model_state = None 

#     # retrain
#     if config.TRAIN.AUTO_RESUME:
#         resume_file_path = auto_resume_helper(config.OUTPUT)
#         if resume_file_path:
#             config.defrost()
#             config.MODEL.RESUME = resume_file_path
#             config.freeze()
#             logger.info(f'auto resuming from {resume_file_path}')
#         else:
#             logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
#     if config.MODEL.RESUME:
#         start_epoch, _ = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

#     # textual prompt
#     text_labels = generate_text(train_data)

#     # model test
#     if config.TEST.ONLY_TEST:
#         results = validate(val_loader, text_labels, model, config)
#         acc1, acc1_local, acc1_fuse = results[0], results[1], results[2]
#         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
#         logger.info(f"Accuracy of the network on the {len(val_data)} test videos: WAR={acc1:.1f}% UAR={uar_global:.1f}%")
#         return
#     print("Start training")
#     #model train and valid
#     for epoch in range(start_epoch, config.TRAIN.EPOCHS):
#         train_loader.sampler.set_epoch(epoch)
#         train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft)

#         # Global, local and fuse classification accuracy
#         results = validate(val_loader, text_labels, model, config)
#         acc_global, acc_local, acc_fuse = results[0], results[1], results[2]
#         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
#         war_global, war_local, war_fuse = results[6], results[7], results[8]
#         cls_acc_global, cls_acc_local, cls_acc_fuse = results[9], results[10], results[11]

#         # Get emotion class names
#         if config.DATA.DATASET.lower() == 'dfew':
#             emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
#         else:
#             emotion_names = [f'Class_{i}' for i in range(len(cls_acc_global))]

#         # Log detailed results for current epoch
#         logger.info(f"=" * 80)
#         logger.info(f"Epoch [{epoch}/{config.TRAIN.EPOCHS - 1}] Validation Results:")
#         logger.info(f"-" * 80)

#         # Global results
#         logger.info(f"Global Branch:")
#         logger.info(f"  WAR: {war_global:.2f}%  |  UAR: {uar_global:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_global):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         # Local results
#         logger.info(f"-" * 80)
#         logger.info(f"Local Branch:")
#         logger.info(f"  WAR: {war_local:.2f}%  |  UAR: {uar_local:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_local):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         # Fuse results
#         logger.info(f"-" * 80)
#         logger.info(f"Fuse Branch:")
#         logger.info(f"  WAR: {war_fuse:.2f}%  |  UAR: {uar_fuse:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_fuse):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")
#         logger.info(f"=" * 80)

#         is_best = war_global > max_war_global  # Use WAR (acc_global) as the criterion for best model

#         # Update best epoch info - only update when finding a new best WAR
#         if is_best:
#             best_epoch = epoch

#             # 【新增】将当前表现最好的权重深拷贝并保存在内存中
#             best_model_state = copy.deepcopy(model.module.state_dict())

#             # Update all metrics from the SAME epoch (the one with best WAR)
#             best_war_global = war_global
#             best_uar_global = uar_global
#             best_cls_acc_global = cls_acc_global.copy()
#             best_war_local = war_local
#             best_uar_local = uar_local
#             best_cls_acc_local = cls_acc_local.copy()
#             best_war_fuse = war_fuse
#             best_uar_fuse = uar_fuse
#             best_cls_acc_fuse = cls_acc_fuse.copy()

#             # Update max values - these are from the best epoch
#             max_war_global = war_global
#             max_war_local = war_local
#             max_war_fuse = war_fuse

#             logger.info(f">>> New best model found at epoch {epoch}! WAR: {war_global:.2f}% UAR: {uar_global:.2f}% <<<")

#         logger.info(f'Current Best Epoch: {best_epoch}')
#         logger.info(f'Best Global - WAR: {max_war_global:.2f}% | UAR: {best_uar_global:.2f}%')
#         logger.info(f'Best Local  - WAR: {max_war_local:.2f}% | UAR: {best_uar_local:.2f}%')
#         logger.info(f'Best Fuse   - WAR: {max_war_fuse:.2f}% | UAR: {best_uar_fuse:.2f}%')
#         # save model
#         if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
#             epoch_saving(config, epoch, model.module, max_war_global, optimizer, lr_scheduler, logger, config.OUTPUT,
#                          is_best)

#         # 【新增】权重回溯逻辑
#         # 假设调度器设置为在第 20 轮重启，那么在第 19 轮结束时执行权重回溯
#         restart_epoch_idx = 19
#         if epoch == restart_epoch_idx:
#             logger.info(f"====== 触发热重启！准备回溯到第 {best_epoch} 轮的最佳权重 (WAR: {max_war_global:.2f}%) ======")
#             if best_model_state is not None:
#                 # 将保存在内存中的最佳权重重新加载到模型中
#                 model.module.load_state_dict(best_model_state)
#                 logger.info(f"====== 成功加载最佳权重！即将以高学习率开启下一轮冲刺！ ======")

#     # Print best epoch summary
#     logger.info(f"\n" + "=" * 80)
#     logger.info(f"TRAINING COMPLETED - BEST MODEL SUMMARY")
#     logger.info(f"=" * 80)
#     logger.info(f"Best Epoch: {best_epoch}")
#     logger.info(f"-" * 80)

#     # Get emotion class names
#     if config.DATA.DATASET.lower() == 'dfew':
#         emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
#     else:
#         emotion_names = [f'Class_{i}' for i in range(len(best_cls_acc_global)) if best_cls_acc_global is not None]

#     if best_cls_acc_global is not None:
#         logger.info(f"Global Branch (Best):")
#         logger.info(f"  WAR: {best_war_global:.2f}%  |  UAR: {best_uar_global:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_global):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         logger.info(f"-" * 80)
#         logger.info(f"Local Branch (Best):")
#         logger.info(f"  WAR: {best_war_local:.2f}%  |  UAR: {best_uar_local:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_local):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         logger.info(f"-" * 80)
#         logger.info(f"Fuse Branch (Best):")
#         logger.info(f"  WAR: {best_war_fuse:.2f}%  |  UAR: {best_uar_fuse:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_fuse):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")
#     logger.info(f"=" * 80 + "\n")

#     # validation after training
#     config.defrost()
#     config.TEST.NUM_CLIP = 4
#     config.TEST.NUM_CROP = 3
#     config.freeze()
#     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
#     results = validate(val_loader, text_labels, model, config)
#     acc = results[0]
#     uar_global = results[3]
#     logger.info(f"Final Accuracy: WAR={acc:.2f}% UAR={uar_global:.2f}%")


# def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft):
#     au_criterion = torch.nn.BCEWithLogitsLoss()
#     model.train()
#     optimizer.zero_grad()

#     num_steps = len(train_loader)
#     batch_time = AverageMeter()
#     tot_loss_meter = AverageMeter()

#     start = time.time()
#     end = time.time()
#     scaler = GradScaler()
#     texts = text_labels.cuda(non_blocking=True)

#     for idx, batch_data in enumerate(train_loader):
#         images = batch_data["imgs"].cuda(non_blocking=True)
#         label_id = batch_data["label"].cuda(non_blocking=True)
#         au_labels = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
#         label_id = label_id.reshape(-1)
#         images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

#         # Modified code - Apply mixup to both images/labels and AU labels
#         au_targets = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
#         original_label_id = label_id.clone() if config.AU.ENABLED and mixup_fn is not None else None

#         if mixup_fn is not None:
#             images, label_id = mixup_fn(images, label_id)
#             # Apply the same mixup to AU labels using stored parameters
#             if config.AU.ENABLED and hasattr(mixup_fn, 'last_lam') and hasattr(mixup_fn, 'last_rand_index'):
#                 lam = mixup_fn.last_lam
#                 rand_index = mixup_fn.last_rand_index
#                 # Mix AU labels with the same lambda and permutation
#                 au_targets = lam * au_targets + (1 - lam) * au_targets[rand_index, :]

#         if texts.shape[0] == 1:
#             texts = texts.view(1, -1)
#         with autocast():
#             output_global, output_local, feat, au_logits = model(images, texts)
#             if epoch>13:
#                 total_loss = criterion(output_global, label_id)
#                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#             else:
#                 pre_output_local = torch.sum(
#                     output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
#                 # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
#                 total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
#                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

#             # Compute AU loss regardless of mixup status
#             # When mixup is enabled, AU labels are also mixed with the same parameters
#             if config.AU.ENABLED:
#                 # Use original emotion labels before mixup for AU loss computation
#                 # because compute_au_loss_stage_b needs hard emotion class labels for knowledge weighting
#                 labels_for_au = original_label_id if original_label_id is not None else label_id
#                 au_loss = compute_au_loss_stage_b(
#                     au_logits, au_targets, labels_for_au,
#                     posw_option=config.AU.POSW_OPTION
#                 )
#                 total_loss = total_loss + config.AU.LAMBDA * au_loss
#                 #total_loss = total_loss * (1 - config.AU.LAMBDA) + config.AU.LAMBDA * au_loss

#         if config.TRAIN.ACCUMULATION_STEPS == 1:
#             optimizer.zero_grad()
#         if config.TRAIN.OPT_LEVEL != 'O0':
#             scaler.scale(total_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             total_loss.backward()
#             optimizer.step()

#         if config.TRAIN.ACCUMULATION_STEPS > 1:
#             if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()
#                 lr_scheduler.step_update(epoch * num_steps + idx)
#         else:
#             scaler.step(optimizer)
#             scaler.update()
#             lr_scheduler.step_update(epoch * num_steps + idx)

#         torch.cuda.synchronize()

#         tot_loss_meter.update(total_loss.item(), len(label_id))
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if idx % config.PRINT_FREQ == 0:
#             lr = optimizer.param_groups[0]['lr']
#             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
#             etas = batch_time.avg * (num_steps - idx)
#             logger.info(
#                 f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
#                 f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
#                 f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
#                 f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
#                 f'mem {memory_used:.0f}MB')
#     epoch_time = time.time() - start
#     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# @torch.no_grad()
# def validate(val_loader, text_labels, model, config):
#     model.eval()

#     acc_global_meter, acc_local_meter, acc_fuse_meter = AverageMeter(), AverageMeter(), AverageMeter()

#     probility = []
#     video_pre_global = []
#     video_pre_local = []
#     video_pre_fuse = []
#     video_label = []
#     with torch.no_grad():
#         text_inputs = text_labels.cuda()
#         logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
#         for idx, batch_data in enumerate(val_loader):
#             _image = batch_data["imgs"]
#             label_id = batch_data["label"]
#             label_id = label_id.reshape(-1)

#             b, tn, c, h, w = _image.size()

#             t = config.DATA.NUM_FRAMES
#             n = tn // t
#             _image = _image.view(b, n, t, c, h, w)
#             tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             tot_similarity_local = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             for i in range(n):
#                 image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
#                 label_id = label_id.cuda(non_blocking=True)
#                 image_input = image.cuda(non_blocking=True)

#                 if config.TRAIN.OPT_LEVEL == 'O2':
#                     image_input = image_input.half()
#                 with autocast():
#                     output, output_local, feat, _ = model(image_input, text_inputs)
#                 if idx < 1:
#                     feature = feat
#                 else:
#                     feature = torch.cat((feature, feat), dim=0)

#                 pre_output_global = output.view(b, -1)
#                 pre_output_local = torch.sum(output_local.view(b, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES),dim=1).squeeze(dim=-1)

#                 similarity = pre_output_global.view(b, -1).softmax(dim=-1)
#                 tot_similarity += similarity

#                 similarity_local = pre_output_local.view(b, -1).softmax(dim=-1)
#                 tot_similarity_local += similarity_local.view(b, -1)

#             probility.extend(tot_similarity.data.cpu().numpy().copy())
#             values_global, indices_global = tot_similarity.topk(1, dim=-1)

#             values_local, indices_local = tot_similarity_local.topk(1, dim=-1)

#             fuse_similarity = tot_similarity + tot_similarity_local
#             values_fuse, indices_fuse = fuse_similarity.topk(1, dim=-1)

#             acc_global = 0
#             acc_local = 0
#             acc_fuse = 0
#             for i in range(b):
#                 video_pre_global.append(indices_global[i].data.cpu().numpy().copy())
#                 video_pre_local.append(indices_local[i].data.cpu().numpy().copy())
#                 video_pre_fuse.append(indices_fuse[i].data.cpu().numpy().copy())
#                 video_label.append(label_id[i].data.cpu().numpy().copy())
#                 if indices_global[i] == label_id[i]:
#                     acc_global += 1
#                 if indices_local[i] == label_id[i]:
#                     acc_local += 1
#                 if indices_fuse[i] == label_id[i]:
#                     acc_fuse += 1

#             acc_global_meter.update(float(acc_global) / b * 100, b)
#             acc_local_meter.update(float(acc_local) / b * 100, b)
#             acc_fuse_meter.update(float(acc_fuse) / b * 100, b)
#             if idx % config.PRINT_FREQ == 0:
#                 logger.info(
#                     f'Test: [{idx}/{len(val_loader)}]\t'
#                     f'Acc@1: {acc_global_meter.avg:.3f}\t'
#                     f'Acc@1: {acc_local_meter.avg:.3f}\t'
#                     f'Acc@1: {acc_fuse_meter.avg:.3f}\t'
#                 )
#     # confusion matrix
#     cf = confusion_matrix(video_label, video_pre_global)
#     np.set_printoptions(precision=4)
#     normalized_cm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
#     normalized_cm = normalized_cm * 100

#     cls_cnt = normalized_cm.sum(axis=1)
#     cls_hit = np.diag(normalized_cm)
#     # print(cf)
#     cls_acc = cls_hit / cls_cnt
#     cls_acc = np.around(cls_acc, 4)
#     cm = np.array(normalized_cm)
#     # save_path = 'AU-CLIP/results'
#     # if not os.path.exists(save_path):
#     #     os.makedirs(save_path)
#     # labels_name = ['hap', 'sad', 'neu', 'ang', 'sur', 'dis', 'fea']
#     # plot_confusion_matrix(cm, labels_name, 'AUCLIP', cls_acc)
#     #
#     # #t-SNE
#     # col = ['orange', 'purple', 'g', 'r', 'darkblue', 'chocolate', 'c']
#     # x_embed = TSNE(n_components=2, perplexity=100, n_iter=10000).fit_transform(feature.data.cpu())
#     # label = np.array(video_label)
#     # plt.figure(figsize=(6, 6))
#     # for i in range(7):
#     #     idxs = np.where(label == i)[0]
#     #     plt.scatter(x_embed[idxs, 0], x_embed[idxs, 1], color=col[i], s=6, label=labels_name[i])
#     # plt.legend(loc='upper left')
#     # plt.xticks(fontsize=13)
#     # plt.yticks(fontsize=13)
#     # plt.savefig(os.path.join(save_path, 'AUCLIP_TSNE.jpg'), format='jpg')
#     # plt.show()

#     logger.info(f'Global - Class-wise Accuracy: {cls_acc}')
#     upper = np.mean(np.max(cf, axis=1) / cls_cnt)
#     logger.info(f'Global - Upper bound: {upper}')
#     logger.info('Global - Evaluation is finished')
#     logger.info(f'Global - Class Accuracy (UAR): {np.mean(cls_acc) * 100:.2f}%')

#     cf_local = confusion_matrix(video_label, video_pre_local).astype(float)
#     cls_cnt_local = cf_local.sum(axis=1)
#     cls_hit_local = np.diag(cf_local)
#     # print(cf)
#     cls_acc_local = cls_hit_local / cls_cnt_local
#     cls_acc_local = np.around(cls_acc_local, 4)
#     logger.info(f'Local - Class-wise Accuracy: {cls_acc_local}')
#     upper = np.mean(np.max(cf_local, axis=1) / cls_cnt_local)
#     logger.info(f'Local - Upper bound: {upper}')
#     logger.info('Local - Evaluation is finished')
#     logger.info(f'Local - Class Accuracy (UAR): {np.mean(cls_acc_local) * 100:.2f}%')

#     cf_fuse = confusion_matrix(video_label, video_pre_fuse).astype(float)
#     cls_cnt_fuse = cf_fuse.sum(axis=1)
#     cls_hit_fuse = np.diag(cf_fuse)
#     # print(cf)
#     cls_acc_fuse = cls_hit_fuse / cls_cnt_fuse
#     cls_acc_fuse = np.around(cls_acc_fuse, 4)
#     logger.info(f'Fuse - Class-wise Accuracy: {cls_acc_fuse}')
#     upper = np.mean(np.max(cf_fuse, axis=1) / cls_cnt_fuse)
#     logger.info(f'Fuse - Upper bound: {upper}')
#     logger.info('Fuse - Evaluation is finished')
#     logger.info(f'Fuse - Class Accuracy (UAR): {np.mean(cls_acc_fuse) * 100:.2f}%')

#     acc_global_meter.sync()
#     acc_local_meter.sync()
#     acc_fuse_meter.sync()
#     logger.info(f' * Acc@1 {acc_global_meter.avg:.3f} Acc_loca@1 {acc_local_meter.avg:.3f}  Acc_fuse@1 {acc_fuse_meter.avg:.3f}')

#     # Calculate UAR (Unweighted Average Recall) and WAR (Weighted Average Recall)
#     uar_global = np.mean(cls_acc) * 100  # UAR for global
#     uar_local = np.mean(cls_acc_local) * 100  # UAR for local
#     uar_fuse = np.mean(cls_acc_fuse) * 100  # UAR for fuse
#     war_global = acc_global_meter.avg  # WAR is the same as overall accuracy
#     war_local = acc_local_meter.avg
#     war_fuse = acc_fuse_meter.avg

#     return (acc_global_meter.avg, acc_local_meter.avg, acc_fuse_meter.avg,
#             uar_global, uar_local, uar_fuse, war_global, war_local, war_fuse,
#             cls_acc, cls_acc_local, cls_acc_fuse,
#             video_label, video_pre_global, video_pre_local, video_pre_fuse)


# if __name__ == '__main__':
#     args, config = parse_option()

#     # 初始化分布式环境
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ['WORLD_SIZE'])
#         local_rank = int(os.environ['LOCAL_RANK'])  # 使用环境变量中的 LOCAL_RANK
#         print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
#     else:
#         rank = -1
#         world_size = -1
#         local_rank = args.local_rank  # 如果未设置环境变量，则使用命令行参数

#     # 设置当前 GPU 设备
#     torch.cuda.set_device(local_rank)

#     # 初始化分布式进程组
#     dist.init_process_group(
#         backend='nccl',
#         init_method='env://',
#         world_size=world_size,
#         rank=rank
#     )

#     # 确保所有进程同步
#     dist.barrier()

#     # 设置随机种子
#     seed = config.SEED + dist.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     cudnn.benchmark = True

#     # 创建输出目录
#     output_dir = Path(config.OUTPUT)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 创建日志记录器
#     logger = create_logger(output_dir=output_dir, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
#     logger.info(f"Working directory: {output_dir}")

#     # 保存配置文件（仅主进程执行）
#     if dist.get_rank() == 0:
#         logger.info(config)
#         shutil.copy(args.config, output_dir)

#     # 启动主训练逻辑
#     main(config)































    


# import os
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import argparse
# import datetime
# import shutil
# import time
# import numpy as np
# import random
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from pathlib import Path
# from sklearn.metrics import confusion_matrix
# from sklearn.manifold import TSNE
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
# from utils.optimizer import build_optimizer, build_scheduler
# from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, plot_confusion_matrix
# from utils.logger import create_logger
# from datasets.build import build_dataloader
# from datasets.blending import FixMixupBlending
# from utils.config import get_config
# from models import AU_clip
# import torch.nn.functional as F
# K_TABLE_DFEW = {
#     0: [0.37458275378581574, 0.37673577978763106, 0.41551698634677164, 0.20406588498862172,
#         0.7927884197377446, 0.8938573401735845, 0.26152800062149906, 0.8999413926399066,
#         0.9306407335956202, 0.8244182955003968, 0.2344366715203141, 0.35584905526535116,
#         0.2637127706449251, 0.1980659294341865, 0.8749731256118369, 0.5415473658691405,
#         0.4089852072091214, 0.29454183039387216],
#     1: [0.3563166809248465, 0.22025123080620432, 0.7883122756737404, 0.18552393229679323, 0.4330479071201498, 0.6512463687571445, 0.19562608246203084, 0.57228053429373, 0.3919835645137292, 0.5539327475091499, 0.26436050797140115, 0.3859719011102806, 0.2624203556065227, 0.19099832378533962, 0.5784889747902296, 0.5465085543388307, 0.4089852072091214, 0.31238729511297314],  # sad
#     2: [0.29634994040489937, 0.26249346997682327, 0.5349197805247116, 0.211357886947774, 0.24956316484226185, 0.48303878792569305, 0.1750459267403643, 0.4222786033590969, 0.2595297603319664, 0.4038707629862569, 0.2255822802606922, 0.3300116586908106, 0.21700897117154114, 0.18584396273490125, 0.4406741647399928, 0.4892561535112399, 0.4089852072091214, 0.24616267045793816],  # neutral
#     3: [0.2565426020973117, 0.22590006587965014, 0.7487900008174104, 0.24260938453624706, 0.3297348731445176, 0.6253054547733351, 0.2331741603449138, 0.6002659975024218, 0.25865288585343615, 0.4042681727124247, 0.25714836786909934, 0.4073109215604269, 0.22487756342746285, 0.1940789790636582, 0.7184070119769908, 0.6300831522957303, 0.4089852072091214, 0.22674904330132986],  # angry
#     4: [0.45840639969725916, 0.4348135864186588, 0.5807746998255159, 0.32444326825812997, 0.27370636575780566, 0.49552874169247974, 0.22029419679136286, 0.45675173141757053, 0.30048086399990936, 0.35356016873960344, 0.2340917014067623, 0.35356016873960344, 0.22418778976044848, 0.18774040869030503, 0.6204770513319532, 0.6347324688342945, 0.4089852072091214, 0.2326942881893769],  # surprise
#     5: [0.3669101379226728, 0.23849898584920362, 0.8556853679596857, 0.2093396686826401, 0.47998741884320045, 0.9116481111605065, 0.47437419451562296, 0.8176370658093514, 0.3690013619929339, 0.6122010138157353, 0.37313028939087656, 0.43845703949339593, 0.2712229960450541, 0.22790281312128532, 0.7303843653361992, 0.5736046807175323, 0.4089852072091214, 0.3157383637072392],  # disgust
#     6: [0.4668184913800416, 0.3236830048705496, 0.7092147921039609, 0.3237692334141919, 0.284442322105034, 0.4833791428919572, 0.19268179747966238, 0.49211626418097015, 0.29857493916455136, 0.40402139627383793, 0.3114334189399665, 0.42236998344528387, 0.2762397282204001, 0.20086024967771943, 0.6377967064381651, 0.6643128457350147, 0.4089852072091214, 0.22004117429592432],  # fear
# }
# # K_TABLE_DFEW = {
# #     0: [0.26197021302251783, 0.2637489312360733, 0.2964310301204501, 0.13190483696240127,
# #         0.6939537051210115, 0.8330800031639156, 0.17347610820526463, 0.8420321208563623,
# #         0.8882934142237333, 0.7356394920285989, 0.15360866990050534, 0.24664740708336055,
# #         0.17509971698083543, 0.1276861973423668, 0.8057327549388998, 0.41178926421868306,
# #         0.2908397350173472, 0.19836039970638378],  # happy
# #     1: [0.2470265618083506, 0.1433979246927842, 0.6881822893577378, 0.11893982182697348,
# #         0.31161634476423544, 0.5253221085983302, 0.12597712051107948, 0.4422623779963743,
# #         0.27645211415344356, 0.42395131504088984, 0.17558170449960314, 0.27142133890425624,
# #         0.1741388696479932, 0.1227455638528141, 0.4485394778155102, 0.41664204112281555,
# #         0.2908397350173472, 0.2121307850723057],  # sad
# #     2: [0.19974525492944142, 0.17419319637385966, 0.4053456334801515, 0.13706230525786608,
# #         0.16464150662864904, 0.3564013123982043, 0.1117065857762667, 0.30225692349830036,
# #         0.17199394819786643, 0.28648655803768563, 0.14721999753281687, 0.22595757296022984,
# #         0.14108230584951664, 0.11916179838996314, 0.31830472814177574, 0.36213049142283893,
# #         0.2908397350173472, 0.16214811486455724],  # neutral
# #     3: [0.16978330014266482, 0.1474484099458089, 0.6385375278419455, 0.15955087911488458,
# #         0.2257386547042889, 0.4972455283651611, 0.15269462267081732, 0.470889570126452,
# #         0.1713443932394683, 0.28682403662377504, 0.17023111167232707, 0.28941226670462944,
# #         0.14671370568908396, 0.12489530852760639, 0.6019091910383833, 0.5023565940498623,
# #         0.2908397350173472, 0.1480589426871144],  # angry
# #     4: [0.3340537870187493, 0.31316038897162674, 0.4508609549567622, 0.2215642997098194,
# #         0.18256790904316805, 0.36794740171201545, 0.14342865607335295, 0.3325723523253354,
# #         0.20291790550542987, 0.24479399356303455, 0.15335881131831128, 0.24479399356303455,
# #         0.14621845929676547, 0.12047847974103684, 0.4921072671708563, 0.5073560806954985,
# #         0.2908397350173472, 0.15234747176484023],  # surprise
# #     5: [0.2556613250379876, 0.15655681634966592, 0.7784678175136164, 0.13563151267233145,
# #         0.353602732625092, 0.8594560438530944, 0.3484770561236273, 0.7265667054811852,
# #         0.2573762546624107, 0.4833627557878272, 0.26077234063862126, 0.3163550480772833,
# #         0.1807057096038953, 0.1488894096862477, 0.6161944289089377, 0.44359769238463886,
# #         0.2908397350173472, 0.21474224698158992],  # disgust
# #     6: [0.3416233199942143, 0.22096625856984287, 0.591078697660434, 0.2210340665261555,
# #         0.19066749128518232, 0.35671400783212986, 0.1239196009045801, 0.3647782154323708,
# #         0.2014525981038694, 0.28661445976904537, 0.21138892966648018, 0.30233592318202696,
# #         0.18447197078258057, 0.12964812274717674, 0.5106650931015151, 0.5397724065827655,
# #         0.2908397350173472, 0.1432476989807878],  # fear
# # }
# K_SCALE = 5.0

# POSW_GLOBAL_DFEW = [8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674,
#                    4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794,
#                    7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384,
#                    12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372,
#                    101.28938906752411, 8.483380533611566]

# POSW_DISTINCT_DFEW = {
#     0: [4.648862512363996, 4.162923411588553, 1.83239825175909, 2.5131103421760863, 1.0496164371270917, 1.191954326288771, 15.430099793221252, 0.636697444899202, 1.0033104960263086, 0.7217365088935785, 3.69328950409615, 2.862698681095705, 6.9568094740508535, 4.616744014506562, 2.920370688175734, 5.462006293978289, 57.59313882654697, 4.170958066889254],  # happy
#     1: [5.6823671940967, 4.992658194508461, 0.5149984185907146, 2.6989371862870613, 3.1921004145555045, 1.6714365386873313, 19.07732186190161, 2.1056834274599465, 7.311081685767773, 1.9735191296108734, 5.039584577609662, 3.51384996900186, 8.076543333000897, 6.52494935714581, 3.628397792864953, 5.6926866933852995, 70.26898981989036, 5.224216933388045],  # sad
#     2: [6.4774986002239645, 5.6010812480692, 0.8710279064472912, 1.9167337801498792, 7.8117860530331145, 2.7351547887496284, 21.302160526041124, 3.19245786489297, 20.999073406774425, 2.574808023689626, 5.714757086292502, 3.906024704963953, 8.191199242945629, 5.206488904380156, 4.249791165053314, 7.363091976516634, 81.4053220208253, 4.963300960035722],  # neutral
#     3: [4.928812812224508, 4.114906832298137, 0.9234234234234234, 1.4543961558346765, 4.781224255883091, 2.435997871208089, 12.75189571440743, 1.44547134935305, 13.252185430463577, 3.1313061506565307, 3.579413266753674, 2.578767654819184, 5.92967542503864, 5.292631578947368, 2.6113572291582763, 4.433813627794237, 55.85311729482212, 4.549840112780662],  # angry
#     4: [7.116157728166966, 6.34866790582404, 0.9600900658968373, 0.761682850299846, 17.648977987421382, 5.163029358274876, 21.278938718008924, 6.183435536376713, 35.41059094397544, 6.815336463223788, 6.358355951919348, 4.08091030789826, 9.571078431372548, 7.167857450288371, 5.090243902439024, 8.104394549990404, 94.26706827309236, 6.122504128509233],  # surprise
#     5: [4.229214780600462, 4.09392575928009, 0.7474435655026047, 2.1834797891036906, 5.054144385026738, 1.6915304606240713, 10.307116104868914, 2.2381122631390777, 10.731865284974093, 3.210599721059972, 4.137266023823029, 3.012848914488259, 5.983037779491133, 6.148382004735596, 3.6374807987711213, 5.387165021156559, 27.391849529780565, 3.2964895635673623],  # disgust
#     6: [5.96072648535643, 5.154494891980657, 0.7569813845450734, 1.5509251975236857, 7.060588375159415, 2.779909786630176, 17.84796563052818, 2.8554517069539505, 15.333362533397574, 3.2621352565348083, 4.923954312221004, 3.5370230679384855, 7.644713355124371, 5.762544656620061, 3.9785987023043443, 6.670773851153989, 57.87385538364383, 5.229860670252932],  # fear
# }

# # minor strategy: define which classes are "major"
# MAJOR_CLASSES_DFEW = {0, 1, 2, 3}

# def compute_au_loss_stage_b(au_logits, au_targets, labels, posw_option='global'):
#     """
#     au_logits:  [B,18] float
#     au_targets: [B,18] float {0,1}
#     labels:     [B] long, emotion class ids (0..C-1)
#     """
#     device = au_logits.device
#     B = labels.shape[0]
#     # pos_weight: [B,18]
#     if config.DATA.DATASET=='DFEW':
#         # knowledge weight: [B,18]
#         k = torch.stack(
#             [torch.tensor(K_TABLE_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#             dim=0
#         ) * K_SCALE
#         if posw_option == 'global':
#             pw = torch.tensor(POSW_GLOBAL_DFEW, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
#         elif posw_option == 'distinct':
#             pw = torch.stack(
#                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#                 dim=0
#             )
#         elif posw_option == 'minor':
#             pw = torch.stack(
#                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#                 dim=0
#             )
#             # overwrite major classes with 1s
#             mask_major = torch.tensor([int(c) in MAJOR_CLASSES_DFEW for c in labels], device=device, dtype=torch.bool)
#             if mask_major.any():
#                 pw = pw.clone()
#                 pw[mask_major] = 1.0
#         else:
#             raise ValueError(f'Unknown posw_option: {posw_option}')

#     # weighted BCE with logits
#     return F.binary_cross_entropy_with_logits(
#         au_logits, au_targets,
#         weight=k,
#         pos_weight=pw,
#         reduction='mean'
#     )

# def parse_option():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/dfew7/16_16.yaml')
#     parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )
#     parser.add_argument('--gpu', default=[0, 1], type=int,help='GPU id to use.')
#     parser.add_argument('--output', type=str, default="DFEWAS2")
#     parser.add_argument('--resume', type=str)
#     parser.add_argument('--pretrained', type=str)
#     parser.add_argument('--only_test', action='store_true')
#     parser.add_argument('--batch-size', type=int)
#     parser.add_argument('--accumulation-steps', type=int)
#     parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
#     args = parser.parse_args()
#     config = get_config(args)
#     return args, config


# def main(config):
#     # load train and valid dataset
#     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

#     # load pretrained model
#     model, _ = AU_clip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
#                           device="cpu", jit=False,
#                           T=config.DATA.NUM_FRAMES,
#                           droppath=config.MODEL.DROP_PATH_RATE,
#                           use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#                           use_cache=config.MODEL.FIX_TEXT,
#                           logger=logger,
#                           N=config.DATA.NUM_DIVIDE,
#                           cfg=config
#                           )
#     model = model.cuda()

#     # training data augmentation
#     mixup_fn = None
#     if config.AUG.MIXUP > 0:
#         criterion = SoftTargetCrossEntropy()
#         criterion_soft = SoftTargetCrossEntropy()
#         mixup_fn = FixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
#                                     smoothing=config.AUG.LABEL_SMOOTH,
#                                     mixup_alpha=config.AUG.MIXUP,
#                                     fmix_alpha=config.AUG.CUTMIX,
#                                     switch_prob=config.AUG.MIXUP_SWITCH_PROB)
#     elif config.AUG.LABEL_SMOOTH > 0:
#         criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
#         criterion_soft = SoftTargetCrossEntropy()

#     else:
#         criterion = nn.CrossEntropyLoss()
#         criterion_soft = SoftTargetCrossEntropy()

#     optimizer = build_optimizer(config, model)
#     lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
#     model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
#                                                       find_unused_parameters=True)

#     start_epoch = 0
#     max_war_global, max_war_local, max_war_fuse = 0.0, 0.0, 0.0

#     # Track best epoch information (all from the same epoch with best WAR)
#     best_epoch = 0
#     best_war_global = 0.0
#     best_uar_global = 0.0
#     best_cls_acc_global = None
#     best_war_local = 0.0
#     best_uar_local = 0.0
#     best_cls_acc_local = None
#     best_war_fuse = 0.0
#     best_uar_fuse = 0.0
#     best_cls_acc_fuse = None
#     # retrain
#     if config.TRAIN.AUTO_RESUME:
#         resume_file_path = auto_resume_helper(config.OUTPUT)
#         if resume_file_path:
#             config.defrost()
#             config.MODEL.RESUME = resume_file_path
#             config.freeze()
#             logger.info(f'auto resuming from {resume_file_path}')
#         else:
#             logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
#     if config.MODEL.RESUME:
#         start_epoch, _ = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

#     # textual prompt
#     text_labels = generate_text(train_data)

#     # model test
#     if config.TEST.ONLY_TEST:
#         results = validate(val_loader, text_labels, model, config)
#         acc1, acc1_local, acc1_fuse = results[0], results[1], results[2]
#         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
#         logger.info(f"Accuracy of the network on the {len(val_data)} test videos: WAR={acc1:.1f}% UAR={uar_global:.1f}%")
#         return
#     print("Start training")
#     #model train and valid
#     for epoch in range(start_epoch, config.TRAIN.EPOCHS):
#         train_loader.sampler.set_epoch(epoch)
#         train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft)

#         # Global, local and fuse classification accuracy
#         results = validate(val_loader, text_labels, model, config)
#         acc_global, acc_local, acc_fuse = results[0], results[1], results[2]
#         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
#         war_global, war_local, war_fuse = results[6], results[7], results[8]
#         cls_acc_global, cls_acc_local, cls_acc_fuse = results[9], results[10], results[11]

#         # Get emotion class names
#         if config.DATA.DATASET.lower() == 'dfew':
#             emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
#         else:
#             emotion_names = [f'Class_{i}' for i in range(len(cls_acc_global))]

#         # Log detailed results for current epoch
#         logger.info(f"=" * 80)
#         logger.info(f"Epoch [{epoch}/{config.TRAIN.EPOCHS - 1}] Validation Results:")
#         logger.info(f"-" * 80)

#         # Global results
#         logger.info(f"Global Branch:")
#         logger.info(f"  WAR: {war_global:.2f}%  |  UAR: {uar_global:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_global):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         # Local results
#         logger.info(f"-" * 80)
#         logger.info(f"Local Branch:")
#         logger.info(f"  WAR: {war_local:.2f}%  |  UAR: {uar_local:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_local):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         # Fuse results
#         logger.info(f"-" * 80)
#         logger.info(f"Fuse Branch:")
#         logger.info(f"  WAR: {war_fuse:.2f}%  |  UAR: {uar_fuse:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_fuse):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")
#         logger.info(f"=" * 80)

#         is_best = war_global > max_war_global  # Use WAR (acc_global) as the criterion for best model

#         # Update best epoch info - only update when finding a new best WAR
#         if is_best:
#             best_epoch = epoch
#             # Update all metrics from the SAME epoch (the one with best WAR)
#             best_war_global = war_global
#             best_uar_global = uar_global
#             best_cls_acc_global = cls_acc_global.copy()
#             best_war_local = war_local
#             best_uar_local = uar_local
#             best_cls_acc_local = cls_acc_local.copy()
#             best_war_fuse = war_fuse
#             best_uar_fuse = uar_fuse
#             best_cls_acc_fuse = cls_acc_fuse.copy()

#             # Update max values - these are from the best epoch
#             max_war_global = war_global
#             max_war_local = war_local
#             max_war_fuse = war_fuse

#             logger.info(f">>> New best model found at epoch {epoch}! WAR: {war_global:.2f}% UAR: {uar_global:.2f}% <<<")

#         logger.info(f'Current Best Epoch: {best_epoch}')
#         logger.info(f'Best Global - WAR: {max_war_global:.2f}% | UAR: {best_uar_global:.2f}%')
#         logger.info(f'Best Local  - WAR: {max_war_local:.2f}% | UAR: {best_uar_local:.2f}%')
#         logger.info(f'Best Fuse   - WAR: {max_war_fuse:.2f}% | UAR: {best_uar_fuse:.2f}%')
#         # save model
#         if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
#             epoch_saving(config, epoch, model.module, max_war_global, optimizer, lr_scheduler, logger, config.OUTPUT,
#                          is_best)

#     # Print best epoch summary
#     logger.info(f"\n" + "=" * 80)
#     logger.info(f"TRAINING COMPLETED - BEST MODEL SUMMARY")
#     logger.info(f"=" * 80)
#     logger.info(f"Best Epoch: {best_epoch}")
#     logger.info(f"-" * 80)

#     # Get emotion class names
#     if config.DATA.DATASET.lower() == 'dfew':
#         emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
#     else:
#         emotion_names = [f'Class_{i}' for i in range(len(best_cls_acc_global)) if best_cls_acc_global is not None]

#     if best_cls_acc_global is not None:
#         logger.info(f"Global Branch (Best):")
#         logger.info(f"  WAR: {best_war_global:.2f}%  |  UAR: {best_uar_global:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_global):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         logger.info(f"-" * 80)
#         logger.info(f"Local Branch (Best):")
#         logger.info(f"  WAR: {best_war_local:.2f}%  |  UAR: {best_uar_local:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_local):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         logger.info(f"-" * 80)
#         logger.info(f"Fuse Branch (Best):")
#         logger.info(f"  WAR: {best_war_fuse:.2f}%  |  UAR: {best_uar_fuse:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_fuse):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")
#     logger.info(f"=" * 80 + "\n")

#     # validation after training
#     logger.info("Loading best model for final evaluation...")
#     best_model_path = os.path.join(config.OUTPUT, 'best.pth')
#     checkpoint = torch.load(best_model_path, map_location='cpu')
#     model.module.load_state_dict(checkpoint['model'])
#     logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

#     config.defrost()
#     config.TEST.NUM_CLIP = 1
#     config.TEST.NUM_CROP = 1
#     config.freeze()
#     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
#     results = validate(val_loader, text_labels, model, config)
#     acc = results[0]
#     uar_global = results[3]
#     logger.info(f"Final Accuracy: WAR={acc:.2f}% UAR={uar_global:.2f}%")


# def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft):
#     au_criterion = torch.nn.BCEWithLogitsLoss()
#     model.train()
#     optimizer.zero_grad()

#     num_steps = len(train_loader)
#     batch_time = AverageMeter()
#     tot_loss_meter = AverageMeter()

#     start = time.time()
#     end = time.time()
#     scaler = GradScaler()
#     texts = text_labels.cuda(non_blocking=True)

#     for idx, batch_data in enumerate(train_loader):
#         images = batch_data["imgs"].cuda(non_blocking=True)
#         label_id = batch_data["label"].cuda(non_blocking=True)
#         au_labels = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
#         label_id = label_id.reshape(-1)
#         images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

#         # # Original code - commented out
#         # if mixup_fn is not None:
#         #     images, label_id = mixup_fn(images, label_id)

#         # if texts.shape[0] == 1:
#         #     texts = texts.view(1, -1)
#         # with autocast():
#         #     output_global, output_local, feat, au_logits = model(images, texts)
#         #     if epoch>13:
#         #         total_loss = criterion(output_global, label_id)
#         #         total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#         #     else:
#         #         pre_output_local = torch.sum(
#         #             output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
#         #         # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
#         #         total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
#         #         total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#         #     if config.AU.ENABLED and mixup_fn is None:
#         #         au_targets = batch_data["au"].cuda(non_blocking=True).float()
#         #         au_loss = compute_au_loss_stage_b(
#         #             au_logits, au_targets, label_id,
#         #             posw_option=config.AU.POSW_OPTION
#         #         )
#         #         total_loss = total_loss + config.AU.LAMBDA * au_loss
#         #         #total_loss = (1-config.AU.LAMBDA) * total_loss + config.AU.LAMBDA * au_loss

#         # Modified code - Apply mixup to both images/labels and AU labels
#         au_targets = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
#         original_label_id = label_id.clone() if config.AU.ENABLED and mixup_fn is not None else None

#         if mixup_fn is not None:
#             images, label_id = mixup_fn(images, label_id)
#             # Apply the same mixup to AU labels using stored parameters
#             if config.AU.ENABLED and hasattr(mixup_fn, 'last_lam') and hasattr(mixup_fn, 'last_rand_index'):
#                 lam = mixup_fn.last_lam
#                 rand_index = mixup_fn.last_rand_index
#                 # Mix AU labels with the same lambda and permutation
#                 au_targets = lam * au_targets + (1 - lam) * au_targets[rand_index, :]

#         if texts.shape[0] == 1:
#             texts = texts.view(1, -1)
#         with autocast():
#             output_global, output_local, feat, au_logits = model(images, texts)
#             if epoch>13:
#                 total_loss = criterion(output_global, label_id)
#                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#             else:
#                 pre_output_local = torch.sum(
#                     output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
#                 # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
#                 total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
#                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

#             # Compute AU loss regardless of mixup status
#             # When mixup is enabled, AU labels are also mixed with the same parameters
#             if config.AU.ENABLED:
#                 # Use original emotion labels before mixup for AU loss computation
#                 # because compute_au_loss_stage_b needs hard emotion class labels for knowledge weighting
#                 labels_for_au = original_label_id if original_label_id is not None else label_id
#                 au_loss = compute_au_loss_stage_b(
#                     au_logits, au_targets, labels_for_au,
#                     posw_option=config.AU.POSW_OPTION
#                 )
#                 total_loss = total_loss + config.AU.LAMBDA * au_loss
#                 #total_loss = total_loss * (1 - config.AU.LAMBDA) + config.AU.LAMBDA * au_loss

#         if config.TRAIN.ACCUMULATION_STEPS == 1:
#             optimizer.zero_grad()
#         if config.TRAIN.OPT_LEVEL != 'O0':
#             scaler.scale(total_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             total_loss.backward()
#             optimizer.step()

#         if config.TRAIN.ACCUMULATION_STEPS > 1:
#             if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()
#                 lr_scheduler.step_update(epoch * num_steps + idx)
#         else:
#             scaler.step(optimizer)
#             scaler.update()
#             lr_scheduler.step_update(epoch * num_steps + idx)

#         torch.cuda.synchronize()

#         tot_loss_meter.update(total_loss.item(), len(label_id))
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if idx % config.PRINT_FREQ == 0:
#             lr = optimizer.param_groups[0]['lr']
#             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
#             etas = batch_time.avg * (num_steps - idx)
#             logger.info(
#                 f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
#                 f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
#                 f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
#                 f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
#                 f'mem {memory_used:.0f}MB')
#     epoch_time = time.time() - start
#     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# @torch.no_grad()
# def validate(val_loader, text_labels, model, config):
#     model.eval()

#     acc_global_meter, acc_local_meter, acc_fuse_meter = AverageMeter(), AverageMeter(), AverageMeter()

#     probility = []
#     video_pre_global = []
#     video_pre_local = []
#     video_pre_fuse = []
#     video_label = []
#     with torch.no_grad():
#         text_inputs = text_labels.cuda()
#         logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
#         for idx, batch_data in enumerate(val_loader):
#             _image = batch_data["imgs"]
#             label_id = batch_data["label"]
#             label_id = label_id.reshape(-1)

#             b, tn, c, h, w = _image.size()

#             t = config.DATA.NUM_FRAMES
#             n = tn // t
#             _image = _image.view(b, n, t, c, h, w)
#             tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             tot_similarity_local = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             for i in range(n):
#                 image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
#                 label_id = label_id.cuda(non_blocking=True)
#                 image_input = image.cuda(non_blocking=True)

#                 if config.TRAIN.OPT_LEVEL == 'O2':
#                     image_input = image_input.half()
#                 with autocast():
#                     output, output_local, feat, _ = model(image_input, text_inputs)
#                 if idx < 1:
#                     feature = feat
#                 else:
#                     feature = torch.cat((feature, feat), dim=0)

#                 pre_output_global = output.view(b, -1)
#                 pre_output_local = torch.sum(output_local.view(b, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES),dim=1).squeeze(dim=-1)

#                 similarity = pre_output_global.view(b, -1).softmax(dim=-1)
#                 tot_similarity += similarity

#                 similarity_local = pre_output_local.view(b, -1).softmax(dim=-1)
#                 tot_similarity_local += similarity_local.view(b, -1)

#             probility.extend(tot_similarity.data.cpu().numpy().copy())
#             values_global, indices_global = tot_similarity.topk(1, dim=-1)

#             values_local, indices_local = tot_similarity_local.topk(1, dim=-1)

#             fuse_similarity = tot_similarity + tot_similarity_local
#             values_fuse, indices_fuse = fuse_similarity.topk(1, dim=-1)

#             acc_global = 0
#             acc_local = 0
#             acc_fuse = 0
#             for i in range(b):
#                 video_pre_global.append(indices_global[i].data.cpu().numpy().copy())
#                 video_pre_local.append(indices_local[i].data.cpu().numpy().copy())
#                 video_pre_fuse.append(indices_fuse[i].data.cpu().numpy().copy())
#                 video_label.append(label_id[i].data.cpu().numpy().copy())
#                 if indices_global[i] == label_id[i]:
#                     acc_global += 1
#                 if indices_local[i] == label_id[i]:
#                     acc_local += 1
#                 if indices_fuse[i] == label_id[i]:
#                     acc_fuse += 1

#             acc_global_meter.update(float(acc_global) / b * 100, b)
#             acc_local_meter.update(float(acc_local) / b * 100, b)
#             acc_fuse_meter.update(float(acc_fuse) / b * 100, b)
#             if idx % config.PRINT_FREQ == 0:
#                 logger.info(
#                     f'Test: [{idx}/{len(val_loader)}]\t'
#                     f'Acc@1: {acc_global_meter.avg:.3f}\t'
#                     f'Acc@1: {acc_local_meter.avg:.3f}\t'
#                     f'Acc@1: {acc_fuse_meter.avg:.3f}\t'
#                 )
#     # confusion matrix
#     cf = confusion_matrix(video_label, video_pre_global)
#     np.set_printoptions(precision=4)
#     normalized_cm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
#     normalized_cm = normalized_cm * 100

#     cls_cnt = normalized_cm.sum(axis=1)
#     cls_hit = np.diag(normalized_cm)
#     # print(cf)
#     cls_acc = cls_hit / cls_cnt
#     cls_acc = np.around(cls_acc, 4)
#     cm = np.array(normalized_cm)
#     # save_path = 'AU-CLIP/results'
#     # if not os.path.exists(save_path):
#     #     os.makedirs(save_path)
#     # labels_name = ['hap', 'sad', 'neu', 'ang', 'sur', 'dis', 'fea']
#     # plot_confusion_matrix(cm, labels_name, 'AUCLIP', cls_acc)
#     #
#     # #t-SNE
#     # col = ['orange', 'purple', 'g', 'r', 'darkblue', 'chocolate', 'c']
#     # x_embed = TSNE(n_components=2, perplexity=100, n_iter=10000).fit_transform(feature.data.cpu())
#     # label = np.array(video_label)
#     # plt.figure(figsize=(6, 6))
#     # for i in range(7):
#     #     idxs = np.where(label == i)[0]
#     #     plt.scatter(x_embed[idxs, 0], x_embed[idxs, 1], color=col[i], s=6, label=labels_name[i])
#     # plt.legend(loc='upper left')
#     # plt.xticks(fontsize=13)
#     # plt.yticks(fontsize=13)
#     # plt.savefig(os.path.join(save_path, 'AUCLIP_TSNE.jpg'), format='jpg')
#     # plt.show()

#     logger.info(f'Global - Class-wise Accuracy: {cls_acc}')
#     upper = np.mean(np.max(cf, axis=1) / cls_cnt)
#     logger.info(f'Global - Upper bound: {upper}')
#     logger.info('Global - Evaluation is finished')
#     logger.info(f'Global - Class Accuracy (UAR): {np.mean(cls_acc) * 100:.2f}%')

#     cf_local = confusion_matrix(video_label, video_pre_local).astype(float)
#     cls_cnt_local = cf_local.sum(axis=1)
#     cls_hit_local = np.diag(cf_local)
#     # print(cf)
#     cls_acc_local = cls_hit_local / cls_cnt_local
#     cls_acc_local = np.around(cls_acc_local, 4)
#     logger.info(f'Local - Class-wise Accuracy: {cls_acc_local}')
#     upper = np.mean(np.max(cf_local, axis=1) / cls_cnt_local)
#     logger.info(f'Local - Upper bound: {upper}')
#     logger.info('Local - Evaluation is finished')
#     logger.info(f'Local - Class Accuracy (UAR): {np.mean(cls_acc_local) * 100:.2f}%')

#     cf_fuse = confusion_matrix(video_label, video_pre_fuse).astype(float)
#     cls_cnt_fuse = cf_fuse.sum(axis=1)
#     cls_hit_fuse = np.diag(cf_fuse)
#     # print(cf)
#     cls_acc_fuse = cls_hit_fuse / cls_cnt_fuse
#     cls_acc_fuse = np.around(cls_acc_fuse, 4)
#     logger.info(f'Fuse - Class-wise Accuracy: {cls_acc_fuse}')
#     upper = np.mean(np.max(cf_fuse, axis=1) / cls_cnt_fuse)
#     logger.info(f'Fuse - Upper bound: {upper}')
#     logger.info('Fuse - Evaluation is finished')
#     logger.info(f'Fuse - Class Accuracy (UAR): {np.mean(cls_acc_fuse) * 100:.2f}%')

#     acc_global_meter.sync()
#     acc_local_meter.sync()
#     acc_fuse_meter.sync()
#     logger.info(f' * Acc@1 {acc_global_meter.avg:.3f} Acc_loca@1 {acc_local_meter.avg:.3f}  Acc_fuse@1 {acc_fuse_meter.avg:.3f}')

#     # Calculate UAR (Unweighted Average Recall) and WAR (Weighted Average Recall)
#     uar_global = np.mean(cls_acc) * 100  # UAR for global
#     uar_local = np.mean(cls_acc_local) * 100  # UAR for local
#     uar_fuse = np.mean(cls_acc_fuse) * 100  # UAR for fuse
#     war_global = acc_global_meter.avg  # WAR is the same as overall accuracy
#     war_local = acc_local_meter.avg
#     war_fuse = acc_fuse_meter.avg

#     return (acc_global_meter.avg, acc_local_meter.avg, acc_fuse_meter.avg,
#             uar_global, uar_local, uar_fuse, war_global, war_local, war_fuse,
#             cls_acc, cls_acc_local, cls_acc_fuse,
#             video_label, video_pre_global, video_pre_local, video_pre_fuse)


# if __name__ == '__main__':
#     args, config = parse_option()

#     # 初始化分布式环境
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ['WORLD_SIZE'])
#         local_rank = int(os.environ['LOCAL_RANK'])  # 使用环境变量中的 LOCAL_RANK
#         print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
#     else:
#         rank = -1
#         world_size = -1
#         local_rank = args.local_rank  # 如果未设置环境变量，则使用命令行参数

#     # 设置当前 GPU 设备
#     torch.cuda.set_device(local_rank)

#     # 初始化分布式进程组
#     dist.init_process_group(
#         backend='nccl',
#         init_method='env://',
#         world_size=world_size,
#         rank=rank
#     )

#     # 确保所有进程同步
#     dist.barrier()

#     # 设置随机种子
#     seed = config.SEED + dist.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     cudnn.benchmark = True

#     # 创建输出目录
#     output_dir = Path(config.OUTPUT)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 创建日志记录器
#     logger = create_logger(output_dir=output_dir, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
#     logger.info(f"Working directory: {output_dir}")

#     # 保存配置文件（仅主进程执行）
#     if dist.get_rank() == 0:
#         logger.info(config)
#         shutil.copy(args.config, output_dir)

#     # 启动主训练逻辑
#     main(config)






















































# import os
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import argparse
# import datetime
# import shutil
# import time
# import numpy as np
# import random
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from pathlib import Path
# from sklearn.metrics import confusion_matrix
# from sklearn.manifold import TSNE
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
# from utils.optimizer import build_optimizer, build_scheduler
# from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, plot_confusion_matrix
# from utils.logger import create_logger
# from datasets.build import build_dataloader
# # from utils.config import get_config
# from models import AU_clip
# import torch.nn.functional as F
# K_TABLE_DFEW = {
#     0: [0.37458275378581574, 0.37673577978763106, 0.41551698634677164, 0.20406588498862172,
#         0.7927884197377446, 0.8938573401735845, 0.26152800062149906, 0.8999413926399066,
#         0.9306407335956202, 0.8244182955003968, 0.2344366715203141, 0.35584905526535116,
#         0.2637127706449251, 0.1980659294341865, 0.8749731256118369, 0.5415473658691405,
#         0.4089852072091214, 0.29454183039387216],
#     1: [0.3563166809248465, 0.22025123080620432, 0.7883122756737404, 0.18552393229679323, 0.4330479071201498, 0.6512463687571445, 0.19562608246203084, 0.57228053429373, 0.3919835645137292, 0.5539327475091499, 0.26436050797140115, 0.3859719011102806, 0.2624203556065227, 0.19099832378533962, 0.5784889747902296, 0.5465085543388307, 0.4089852072091214, 0.31238729511297314],  # sad
#     2: [0.29634994040489937, 0.26249346997682327, 0.5349197805247116, 0.211357886947774, 0.24956316484226185, 0.48303878792569305, 0.1750459267403643, 0.4222786033590969, 0.2595297603319664, 0.4038707629862569, 0.2255822802606922, 0.3300116586908106, 0.21700897117154114, 0.18584396273490125, 0.4406741647399928, 0.4892561535112399, 0.4089852072091214, 0.24616267045793816],  # neutral
#     3: [0.2565426020973117, 0.22590006587965014, 0.7487900008174104, 0.24260938453624706, 0.3297348731445176, 0.6253054547733351, 0.2331741603449138, 0.6002659975024218, 0.25865288585343615, 0.4042681727124247, 0.25714836786909934, 0.4073109215604269, 0.22487756342746285, 0.1940789790636582, 0.7184070119769908, 0.6300831522957303, 0.4089852072091214, 0.22674904330132986],  # angry
#     4: [0.45840639969725916, 0.4348135864186588, 0.5807746998255159, 0.32444326825812997, 0.27370636575780566, 0.49552874169247974, 0.22029419679136286, 0.45675173141757053, 0.30048086399990936, 0.35356016873960344, 0.2340917014067623, 0.35356016873960344, 0.22418778976044848, 0.18774040869030503, 0.6204770513319532, 0.6347324688342945, 0.4089852072091214, 0.2326942881893769],  # surprise
#     5: [0.3669101379226728, 0.23849898584920362, 0.8556853679596857, 0.2093396686826401, 0.47998741884320045, 0.9116481111605065, 0.47437419451562296, 0.8176370658093514, 0.3690013619929339, 0.6122010138157353, 0.37313028939087656, 0.43845703949339593, 0.2712229960450541, 0.22790281312128532, 0.7303843653361992, 0.5736046807175323, 0.4089852072091214, 0.3157383637072392],  # disgust
#     6: [0.4668184913800416, 0.3236830048705496, 0.7092147921039609, 0.3237692334141919, 0.284442322105034, 0.4833791428919572, 0.19268179747966238, 0.49211626418097015, 0.29857493916455136, 0.40402139627383793, 0.3114334189399665, 0.42236998344528387, 0.2762397282204001, 0.20086024967771943, 0.6377967064381651, 0.6643128457350147, 0.4089852072091214, 0.22004117429592432],  # fear
# }
# K_SCALE = 5.0

# POSW_GLOBAL_DFEW = [8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674,
#                    4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794,
#                    7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384,
#                    12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372,
#                    101.28938906752411, 8.483380533611566]

# POSW_DISTINCT_DFEW = {
#     0: [4.648862512363996, 4.162923411588553, 1.83239825175909, 2.5131103421760863, 1.0496164371270917, 1.191954326288771, 15.430099793221252, 0.636697444899202, 1.0033104960263086, 0.7217365088935785, 3.69328950409615, 2.862698681095705, 6.9568094740508535, 4.616744014506562, 2.920370688175734, 5.462006293978289, 57.59313882654697, 4.170958066889254],  # happy
#     1: [5.6823671940967, 4.992658194508461, 0.5149984185907146, 2.6989371862870613, 3.1921004145555045, 1.6714365386873313, 19.07732186190161, 2.1056834274599465, 7.311081685767773, 1.9735191296108734, 5.039584577609662, 3.51384996900186, 8.076543333000897, 6.52494935714581, 3.628397792864953, 5.6926866933852995, 70.26898981989036, 5.224216933388045],  # sad
#     2: [6.4774986002239645, 5.6010812480692, 0.8710279064472912, 1.9167337801498792, 7.8117860530331145, 2.7351547887496284, 21.302160526041124, 3.19245786489297, 20.999073406774425, 2.574808023689626, 5.714757086292502, 3.906024704963953, 8.191199242945629, 5.206488904380156, 4.249791165053314, 7.363091976516634, 81.4053220208253, 4.963300960035722],  # neutral
#     3: [4.928812812224508, 4.114906832298137, 0.9234234234234234, 1.4543961558346765, 4.781224255883091, 2.435997871208089, 12.75189571440743, 1.44547134935305, 13.252185430463577, 3.1313061506565307, 3.579413266753674, 2.578767654819184, 5.92967542503864, 5.292631578947368, 2.6113572291582763, 4.433813627794237, 55.85311729482212, 4.549840112780662],  # angry
#     4: [7.116157728166966, 6.34866790582404, 0.9600900658968373, 0.761682850299846, 17.648977987421382, 5.163029358274876, 21.278938718008924, 6.183435536376713, 35.41059094397544, 6.815336463223788, 6.358355951919348, 4.08091030789826, 9.571078431372548, 7.167857450288371, 5.090243902439024, 8.104394549990404, 94.26706827309236, 6.122504128509233],  # surprise
#     5: [4.229214780600462, 4.09392575928009, 0.7474435655026047, 2.1834797891036906, 5.054144385026738, 1.6915304606240713, 10.307116104868914, 2.2381122631390777, 10.731865284974093, 3.210599721059972, 4.137266023823029, 3.012848914488259, 5.983037779491133, 6.148382004735596, 3.6374807987711213, 5.387165021156559, 27.391849529780565, 3.2964895635673623],  # disgust
#     6: [5.96072648535643, 5.154494891980657, 0.7569813845450734, 1.5509251975236857, 7.060588375159415, 2.779909786630176, 17.84796563052818, 2.8554517069539505, 15.333362533397574, 3.2621352565348083, 4.923954312221004, 3.5370230679384855, 7.644713355124371, 5.762544656620061, 3.9785987023043443, 6.670773851153989, 57.87385538364383, 5.229860670252932],  # fear
# }

# # minor strategy: define which classes are "major"
# MAJOR_CLASSES_DFEW = {0, 1, 2, 3}

# def compute_au_loss_stage_b(au_logits, au_targets, labels, posw_option='global', mix_labels=None, mix_lambda=None):
#     """
#     au_logits:  [B,18] float
#     au_targets: [B,18] float {0,1} or mixed
#     labels:     [B] long, emotion class ids (0..C-1) - primary labels
#     mix_labels: [B] long, optional - secondary labels for mixup
#     mix_lambda: float, optional - mixup ratio
#     """
#     device = au_logits.device
#     B = labels.shape[0]

#     if config.DATA.DATASET=='DFEW':
#         # knowledge weight: [B,18]
#         k = torch.stack(
#             [torch.tensor(K_TABLE_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#             dim=0
#         ) * K_SCALE

#         # If mixup is enabled, blend knowledge weights
#         if mix_labels is not None and mix_lambda is not None:
#             k_secondary = torch.stack(
#                 [torch.tensor(K_TABLE_DFEW[int(c)], device=device, dtype=torch.float32) for c in mix_labels],
#                 dim=0
#             ) * K_SCALE
#             k = mix_lambda * k + (1 - mix_lambda) * k_secondary

#         if posw_option == 'global':
#             pw = torch.tensor(POSW_GLOBAL_DFEW, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
#         elif posw_option == 'distinct':
#             pw = torch.stack(
#                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#                 dim=0
#             )
#             # If mixup is enabled, blend pos_weights
#             if mix_labels is not None and mix_lambda is not None:
#                 pw_secondary = torch.stack(
#                     [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in mix_labels],
#                     dim=0
#                 )
#                 pw = mix_lambda * pw + (1 - mix_lambda) * pw_secondary
#         elif posw_option == 'minor':
#             pw = torch.stack(
#                 [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
#                 dim=0
#             )
#             # overwrite major classes with 1s
#             mask_major = torch.tensor([int(c) in MAJOR_CLASSES_DFEW for c in labels], device=device, dtype=torch.bool)
#             if mask_major.any():
#                 pw = pw.clone()
#                 pw[mask_major] = 1.0
#             # If mixup is enabled, blend pos_weights
#             if mix_labels is not None and mix_lambda is not None:
#                 pw_secondary = torch.stack(
#                     [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in mix_labels],
#                     dim=0
#                 )
#                 mask_major_secondary = torch.tensor([int(c) in MAJOR_CLASSES_DFEW for c in mix_labels], device=device, dtype=torch.bool)
#                 if mask_major_secondary.any():
#                     pw_secondary = pw_secondary.clone()
#                     pw_secondary[mask_major_secondary] = 1.0
#                 pw = mix_lambda * pw + (1 - mix_lambda) * pw_secondary
#         else:
#             raise ValueError(f'Unknown posw_option: {posw_option}')

#     # weighted BCE with logits
#     return F.binary_cross_entropy_with_logits(
#         au_logits, au_targets,
#         weight=k,
#         pos_weight=pw,
#         reduction='mean'
#     )

# def parse_option():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/dfew7/16_16.yaml')
#     parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )
#     parser.add_argument('--gpu', default=[0, 1], type=int,help='GPU id to use.')
#     parser.add_argument('--output', type=str, default="DFEWAS2")
#     parser.add_argument('--resume', type=str)
#     parser.add_argument('--pretrained', type=str)
#     parser.add_argument('--only_test', action='store_true')
#     parser.add_argument('--batch-size', type=int)
#     parser.add_argument('--accumulation-steps', type=int)
#     parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
#     args = parser.parse_args()
#     config = get_config(args)
#     return args, config


# def main(config):
#     # load train and valid dataset
#     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

#     # load pretrained model
#     model, _ = AU_clip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
#                           device="cpu", jit=False,
#                           T=config.DATA.NUM_FRAMES,
#                           droppath=config.MODEL.DROP_PATH_RATE,
#                           use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#                           use_cache=config.MODEL.FIX_TEXT,
#                           logger=logger,
#                           N=config.DATA.NUM_DIVIDE,
#                           cfg=config
#                           )
#     model = model.cuda()

#     # training data augmentation
#     mixup_fn = None
#     if config.AUG.MIXUP > 0:
#         criterion = SoftTargetCrossEntropy()
#         criterion_soft = SoftTargetCrossEntropy()
#         mixup_fn = FixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
#                                     smoothing=config.AUG.LABEL_SMOOTH,
#                                     mixup_alpha=config.AUG.MIXUP,
#                                     fmix_alpha=config.AUG.CUTMIX,
#                                     switch_prob=config.AUG.MIXUP_SWITCH_PROB)
#     elif config.AUG.LABEL_SMOOTH > 0:
#         criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
#         criterion_soft = SoftTargetCrossEntropy()

#     else:
#         criterion = nn.CrossEntropyLoss()
#         criterion_soft = SoftTargetCrossEntropy()

#     optimizer = build_optimizer(config, model)
#     lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
#     model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
#                                                       find_unused_parameters=True)

#     start_epoch = 0
#     max_war_global, max_war_local, max_war_fuse = 0.0, 0.0, 0.0

#     # Track best epoch information (all from the same epoch with best WAR)
#     best_epoch = 0
#     best_war_global = 0.0
#     best_uar_global = 0.0
#     best_cls_acc_global = None
#     best_war_local = 0.0
#     best_uar_local = 0.0
#     best_cls_acc_local = None
#     best_war_fuse = 0.0
#     best_uar_fuse = 0.0
#     best_cls_acc_fuse = None
#     # retrain
#     if config.TRAIN.AUTO_RESUME:
#         resume_file_path = auto_resume_helper(config.OUTPUT)
#         if resume_file_path:
#             config.defrost()
#             config.MODEL.RESUME = resume_file_path
#             config.freeze()
#             logger.info(f'auto resuming from {resume_file_path}')
#         else:
#             logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
#     if config.MODEL.RESUME:
#         start_epoch, _ = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

#     # textual prompt
#     text_labels = generate_text(train_data)

#     # model test
#     if config.TEST.ONLY_TEST:
#         results = validate(val_loader, text_labels, model, config)
#         acc1, acc1_local, acc1_fuse = results[0], results[1], results[2]
#         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
#         logger.info(f"Accuracy of the network on the {len(val_data)} test videos: WAR={acc1:.1f}% UAR={uar_global:.1f}%")
#         return
#     print("Start training")
#     #model train and valid
#     for epoch in range(start_epoch, config.TRAIN.EPOCHS):
#         train_loader.sampler.set_epoch(epoch)
#         train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft)

#         # Global, local and fuse classification accuracy
#         results = validate(val_loader, text_labels, model, config)
#         acc_global, acc_local, acc_fuse = results[0], results[1], results[2]
#         uar_global, uar_local, uar_fuse = results[3], results[4], results[5]
#         war_global, war_local, war_fuse = results[6], results[7], results[8]
#         cls_acc_global, cls_acc_local, cls_acc_fuse = results[9], results[10], results[11]

#         # Get emotion class names
#         if config.DATA.DATASET.lower() == 'dfew':
#             emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
#         else:
#             emotion_names = [f'Class_{i}' for i in range(len(cls_acc_global))]

#         # Log detailed results for current epoch
#         logger.info(f"=" * 80)
#         logger.info(f"Epoch [{epoch}/{config.TRAIN.EPOCHS - 1}] Validation Results:")
#         logger.info(f"-" * 80)

#         # Global results
#         logger.info(f"Global Branch:")
#         logger.info(f"  WAR: {war_global:.2f}%  |  UAR: {uar_global:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_global):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         # Local results
#         logger.info(f"-" * 80)
#         logger.info(f"Local Branch:")
#         logger.info(f"  WAR: {war_local:.2f}%  |  UAR: {uar_local:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_local):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         # Fuse results
#         logger.info(f"-" * 80)
#         logger.info(f"Fuse Branch:")
#         logger.info(f"  WAR: {war_fuse:.2f}%  |  UAR: {uar_fuse:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, cls_acc_fuse):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")
#         logger.info(f"=" * 80)

#         is_best = war_global > max_war_global  # Use WAR (acc_global) as the criterion for best model

#         # Update best epoch info - only update when finding a new best WAR
#         if is_best:
#             best_epoch = epoch
#             # Update all metrics from the SAME epoch (the one with best WAR)
#             best_war_global = war_global
#             best_uar_global = uar_global
#             best_cls_acc_global = cls_acc_global.copy()
#             best_war_local = war_local
#             best_uar_local = uar_local
#             best_cls_acc_local = cls_acc_local.copy()
#             best_war_fuse = war_fuse
#             best_uar_fuse = uar_fuse
#             best_cls_acc_fuse = cls_acc_fuse.copy()

#             # Update max values - these are from the best epoch
#             max_war_global = war_global
#             max_war_local = war_local
#             max_war_fuse = war_fuse

#             logger.info(f">>> New best model found at epoch {epoch}! WAR: {war_global:.2f}% UAR: {uar_global:.2f}% <<<")

#         logger.info(f'Current Best Epoch: {best_epoch}')
#         logger.info(f'Best Global - WAR: {max_war_global:.2f}% | UAR: {best_uar_global:.2f}%')
#         logger.info(f'Best Local  - WAR: {max_war_local:.2f}% | UAR: {best_uar_local:.2f}%')
#         logger.info(f'Best Fuse   - WAR: {max_war_fuse:.2f}% | UAR: {best_uar_fuse:.2f}%')
#         # save model
#         if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
#             epoch_saving(config, epoch, model.module, max_war_global, optimizer, lr_scheduler, logger, config.OUTPUT,
#                          is_best)

#     # Print best epoch summary
#     logger.info(f"\n" + "=" * 80)
#     logger.info(f"TRAINING COMPLETED - BEST MODEL SUMMARY")
#     logger.info(f"=" * 80)
#     logger.info(f"Best Epoch: {best_epoch}")
#     logger.info(f"-" * 80)

#     # Get emotion class names
#     if config.DATA.DATASET.lower() == 'dfew':
#         emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
#     else:
#         emotion_names = [f'Class_{i}' for i in range(len(best_cls_acc_global)) if best_cls_acc_global is not None]

#     if best_cls_acc_global is not None:
#         logger.info(f"Global Branch (Best):")
#         logger.info(f"  WAR: {best_war_global:.2f}%  |  UAR: {best_uar_global:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_global):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         logger.info(f"-" * 80)
#         logger.info(f"Local Branch (Best):")
#         logger.info(f"  WAR: {best_war_local:.2f}%  |  UAR: {best_uar_local:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_local):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")

#         logger.info(f"-" * 80)
#         logger.info(f"Fuse Branch (Best):")
#         logger.info(f"  WAR: {best_war_fuse:.2f}%  |  UAR: {best_uar_fuse:.2f}%")
#         logger.info(f"  Per-class Accuracy:")
#         for name, acc in zip(emotion_names, best_cls_acc_fuse):
#             logger.info(f"    {name:10s}: {acc*100:.2f}%")
#     logger.info(f"=" * 80 + "\n")

#     # validation after training
#     config.defrost()
#     config.TEST.NUM_CLIP = 4
#     config.TEST.NUM_CROP = 3
#     config.freeze()
#     train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
#     results = validate(val_loader, text_labels, model, config)
#     acc = results[0]
#     uar_global = results[3]
#     logger.info(f"Final Accuracy: WAR={acc:.2f}% UAR={uar_global:.2f}%")


# def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft):
#     au_criterion = torch.nn.BCEWithLogitsLoss()
#     model.train()
#     optimizer.zero_grad()

#     num_steps = len(train_loader)
#     batch_time = AverageMeter()
#     tot_loss_meter = AverageMeter()

#     start = time.time()
#     end = time.time()
#     scaler = GradScaler()
#     texts = text_labels.cuda(non_blocking=True)

#     for idx, batch_data in enumerate(train_loader):
#         images = batch_data["imgs"].cuda(non_blocking=True)
#         label_id = batch_data["label"].cuda(non_blocking=True)
#         au_labels = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
#         label_id = label_id.reshape(-1)
#         images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

#         # # Original code - commented out
#         # if mixup_fn is not None:
#         #     images, label_id = mixup_fn(images, label_id)

#         # if texts.shape[0] == 1:
#         #     texts = texts.view(1, -1)
#         # with autocast():
#         #     output_global, output_local, feat, au_logits = model(images, texts)
#         #     if epoch>13:
#         #         total_loss = criterion(output_global, label_id)
#         #         total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#         #     else:
#         #         pre_output_local = torch.sum(
#         #             output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
#         #         # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
#         #         total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
#         #         total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#         #     if config.AU.ENABLED and mixup_fn is None:
#         #         au_targets = batch_data["au"].cuda(non_blocking=True).float()
#         #         au_loss = compute_au_loss_stage_b(
#         #             au_logits, au_targets, label_id,
#         #             posw_option=config.AU.POSW_OPTION
#         #         )
#         #         total_loss = total_loss + config.AU.LAMBDA * au_loss
#         #         #total_loss = (1-config.AU.LAMBDA) * total_loss + config.AU.LAMBDA * au_loss

#         # Modified code - Apply mixup to both images/labels and AU labels
#         au_targets = batch_data["au"].cuda(non_blocking=True).float()  # [B,18]
#         original_label_id = label_id.clone() if config.AU.ENABLED and mixup_fn is not None else None
#         mix_labels = None
#         mix_lambda = None

#         if mixup_fn is not None:
#             images, label_id = mixup_fn(images, label_id)
#             # Apply the same mixup to AU labels using stored parameters
#             if config.AU.ENABLED and hasattr(mixup_fn, 'last_lam') and hasattr(mixup_fn, 'last_rand_index'):
#                 lam = mixup_fn.last_lam
#                 rand_index = mixup_fn.last_rand_index
#                 # Mix AU labels with the same lambda and permutation
#                 au_targets = lam * au_targets + (1 - lam) * au_targets[rand_index, :]
#                 # Store mixup info for AU loss computation
#                 mix_labels = original_label_id[rand_index]  # Secondary labels
#                 mix_lambda = lam

#         if texts.shape[0] == 1:
#             texts = texts.view(1, -1)
#         with autocast():
#             output_global, output_local, feat, au_logits = model(images, texts)
#             if epoch>13:
#                 total_loss = criterion(output_global, label_id)
#                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#             else:
#                 pre_output_local = torch.sum(
#                     output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
#                 # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
#                 total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
#                 total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

#             # Compute AU loss regardless of mixup status
#             # When mixup is enabled, weights are also blended
#             if config.AU.ENABLED:
#                 labels_for_au = original_label_id if original_label_id is not None else label_id
#                 au_loss = compute_au_loss_stage_b(
#                     au_logits, au_targets, labels_for_au,
#                     posw_option=config.AU.POSW_OPTION,
#                     mix_labels=mix_labels,
#                     mix_lambda=mix_lambda
#                 )
#                 total_loss = total_loss + config.AU.LAMBDA * au_loss

#         if config.TRAIN.ACCUMULATION_STEPS == 1:
#             optimizer.zero_grad()
#         if config.TRAIN.OPT_LEVEL != 'O0':
#             scaler.scale(total_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             total_loss.backward()
#             optimizer.step()

#         if config.TRAIN.ACCUMULATION_STEPS > 1:
#             if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()
#                 lr_scheduler.step_update(epoch * num_steps + idx)
#         else:
#             scaler.step(optimizer)
#             scaler.update()
#             lr_scheduler.step_update(epoch * num_steps + idx)

#         torch.cuda.synchronize()

#         tot_loss_meter.update(total_loss.item(), len(label_id))
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if idx % config.PRINT_FREQ == 0:
#             lr = optimizer.param_groups[0]['lr']
#             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
#             etas = batch_time.avg * (num_steps - idx)
#             logger.info(
#                 f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
#                 f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
#                 f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
#                 f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
#                 f'mem {memory_used:.0f}MB')
#     epoch_time = time.time() - start
#     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# @torch.no_grad()
# def validate(val_loader, text_labels, model, config):
#     model.eval()

#     acc_global_meter, acc_local_meter, acc_fuse_meter = AverageMeter(), AverageMeter(), AverageMeter()

#     probility = []
#     video_pre_global = []
#     video_pre_local = []
#     video_pre_fuse = []
#     video_label = []
#     with torch.no_grad():
#         text_inputs = text_labels.cuda()
#         logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
#         for idx, batch_data in enumerate(val_loader):
#             _image = batch_data["imgs"]
#             label_id = batch_data["label"]
#             label_id = label_id.reshape(-1)

#             b, tn, c, h, w = _image.size()

#             t = config.DATA.NUM_FRAMES
#             n = tn // t
#             _image = _image.view(b, n, t, c, h, w)
#             tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             tot_similarity_local = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             for i in range(n):
#                 image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
#                 label_id = label_id.cuda(non_blocking=True)
#                 image_input = image.cuda(non_blocking=True)

#                 if config.TRAIN.OPT_LEVEL == 'O2':
#                     image_input = image_input.half()
#                 with autocast():
#                     output, output_local, feat, _ = model(image_input, text_inputs)
#                 if idx < 1:
#                     feature = feat
#                 else:
#                     feature = torch.cat((feature, feat), dim=0)

#                 pre_output_global = output.view(b, -1)
#                 pre_output_local = torch.sum(output_local.view(b, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES),dim=1).squeeze(dim=-1)

#                 similarity = pre_output_global.view(b, -1).softmax(dim=-1)
#                 tot_similarity += similarity

#                 similarity_local = pre_output_local.view(b, -1).softmax(dim=-1)
#                 tot_similarity_local += similarity_local.view(b, -1)

#             probility.extend(tot_similarity.data.cpu().numpy().copy())
#             values_global, indices_global = tot_similarity.topk(1, dim=-1)

#             values_local, indices_local = tot_similarity_local.topk(1, dim=-1)

#             fuse_similarity = tot_similarity + tot_similarity_local
#             values_fuse, indices_fuse = fuse_similarity.topk(1, dim=-1)

#             acc_global = 0
#             acc_local = 0
#             acc_fuse = 0
#             for i in range(b):
#                 video_pre_global.append(indices_global[i].data.cpu().numpy().copy())
#                 video_pre_local.append(indices_local[i].data.cpu().numpy().copy())
#                 video_pre_fuse.append(indices_fuse[i].data.cpu().numpy().copy())
#                 video_label.append(label_id[i].data.cpu().numpy().copy())
#                 if indices_global[i] == label_id[i]:
#                     acc_global += 1
#                 if indices_local[i] == label_id[i]:
#                     acc_local += 1
#                 if indices_fuse[i] == label_id[i]:
#                     acc_fuse += 1

#             acc_global_meter.update(float(acc_global) / b * 100, b)
#             acc_local_meter.update(float(acc_local) / b * 100, b)
#             acc_fuse_meter.update(float(acc_fuse) / b * 100, b)
#             if idx % config.PRINT_FREQ == 0:
#                 logger.info(
#                     f'Test: [{idx}/{len(val_loader)}]\t'
#                     f'Acc@1: {acc_global_meter.avg:.3f}\t'
#                     f'Acc@1: {acc_local_meter.avg:.3f}\t'
#                     f'Acc@1: {acc_fuse_meter.avg:.3f}\t'
#                 )
#     # confusion matrix
#     cf = confusion_matrix(video_label, video_pre_global)
#     np.set_printoptions(precision=4)
#     normalized_cm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
#     normalized_cm = normalized_cm * 100

#     cls_cnt = normalized_cm.sum(axis=1)
#     cls_hit = np.diag(normalized_cm)
#     # print(cf)
#     cls_acc = cls_hit / cls_cnt
#     cls_acc = np.around(cls_acc, 4)
#     cm = np.array(normalized_cm)
#     # save_path = 'AU-CLIP/results'
#     # if not os.path.exists(save_path):
#     #     os.makedirs(save_path)
#     # labels_name = ['hap', 'sad', 'neu', 'ang', 'sur', 'dis', 'fea']
#     # plot_confusion_matrix(cm, labels_name, 'AUCLIP', cls_acc)
#     #
#     # #t-SNE
#     # col = ['orange', 'purple', 'g', 'r', 'darkblue', 'chocolate', 'c']
#     # x_embed = TSNE(n_components=2, perplexity=100, n_iter=10000).fit_transform(feature.data.cpu())
#     # label = np.array(video_label)
#     # plt.figure(figsize=(6, 6))
#     # for i in range(7):
#     #     idxs = np.where(label == i)[0]
#     #     plt.scatter(x_embed[idxs, 0], x_embed[idxs, 1], color=col[i], s=6, label=labels_name[i])
#     # plt.legend(loc='upper left')
#     # plt.xticks(fontsize=13)
#     # plt.yticks(fontsize=13)
#     # plt.savefig(os.path.join(save_path, 'AUCLIP_TSNE.jpg'), format='jpg')
#     # plt.show()

#     logger.info(f'Global - Class-wise Accuracy: {cls_acc}')
#     upper = np.mean(np.max(cf, axis=1) / cls_cnt)
#     logger.info(f'Global - Upper bound: {upper}')
#     logger.info('Global - Evaluation is finished')
#     logger.info(f'Global - Class Accuracy (UAR): {np.mean(cls_acc) * 100:.2f}%')

#     cf_local = confusion_matrix(video_label, video_pre_local).astype(float)
#     cls_cnt_local = cf_local.sum(axis=1)
#     cls_hit_local = np.diag(cf_local)
#     # print(cf)
#     cls_acc_local = cls_hit_local / cls_cnt_local
#     cls_acc_local = np.around(cls_acc_local, 4)
#     logger.info(f'Local - Class-wise Accuracy: {cls_acc_local}')
#     upper = np.mean(np.max(cf_local, axis=1) / cls_cnt_local)
#     logger.info(f'Local - Upper bound: {upper}')
#     logger.info('Local - Evaluation is finished')
#     logger.info(f'Local - Class Accuracy (UAR): {np.mean(cls_acc_local) * 100:.2f}%')

#     cf_fuse = confusion_matrix(video_label, video_pre_fuse).astype(float)
#     cls_cnt_fuse = cf_fuse.sum(axis=1)
#     cls_hit_fuse = np.diag(cf_fuse)
#     # print(cf)
#     cls_acc_fuse = cls_hit_fuse / cls_cnt_fuse
#     cls_acc_fuse = np.around(cls_acc_fuse, 4)
#     logger.info(f'Fuse - Class-wise Accuracy: {cls_acc_fuse}')
#     upper = np.mean(np.max(cf_fuse, axis=1) / cls_cnt_fuse)
#     logger.info(f'Fuse - Upper bound: {upper}')
#     logger.info('Fuse - Evaluation is finished')
#     logger.info(f'Fuse - Class Accuracy (UAR): {np.mean(cls_acc_fuse) * 100:.2f}%')

#     acc_global_meter.sync()
#     acc_local_meter.sync()
#     acc_fuse_meter.sync()
#     logger.info(f' * Acc@1 {acc_global_meter.avg:.3f} Acc_loca@1 {acc_local_meter.avg:.3f}  Acc_fuse@1 {acc_fuse_meter.avg:.3f}')

#     # Calculate UAR (Unweighted Average Recall) and WAR (Weighted Average Recall)
#     uar_global = np.mean(cls_acc) * 100  # UAR for global
#     uar_local = np.mean(cls_acc_local) * 100  # UAR for local
#     uar_fuse = np.mean(cls_acc_fuse) * 100  # UAR for fuse
#     war_global = acc_global_meter.avg  # WAR is the same as overall accuracy
#     war_local = acc_local_meter.avg
#     war_fuse = acc_fuse_meter.avg

#     return (acc_global_meter.avg, acc_local_meter.avg, acc_fuse_meter.avg,
#             uar_global, uar_local, uar_fuse, war_global, war_local, war_fuse,
#             cls_acc, cls_acc_local, cls_acc_fuse,
#             video_label, video_pre_global, video_pre_local, video_pre_fuse)


# if __name__ == '__main__':
#     args, config = parse_option()

#     # 初始化分布式环境
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ['WORLD_SIZE'])
#         local_rank = int(os.environ['LOCAL_RANK'])  # 使用环境变量中的 LOCAL_RANK
#         print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
#     else:
#         rank = -1
#         world_size = -1
#         local_rank = args.local_rank  # 如果未设置环境变量，则使用命令行参数

#     # 设置当前 GPU 设备
#     torch.cuda.set_device(local_rank)

#     # 初始化分布式进程组
#     dist.init_process_group(
#         backend='nccl',
#         init_method='env://',
#         world_size=world_size,
#         rank=rank
#     )

#     # 确保所有进程同步
#     dist.barrier()

#     # 设置随机种子
#     seed = config.SEED + dist.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     cudnn.benchmark = True

#     # 创建输出目录
#     output_dir = Path(config.OUTPUT)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 创建日志记录器
#     logger = create_logger(output_dir=output_dir, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
#     logger.info(f"Working directory: {output_dir}")

#     # 保存配置文件（仅主进程执行）
#     if dist.get_rank() == 0:
#         logger.info(config)
#         shutil.copy(args.config, output_dir)

#     # 启动主训练逻辑
#     main(config)


































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
K_TABLE_DFEW = {
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
# K_TABLE_DFEW = {
#     0: [0.26197021302251783, 0.2637489312360733, 0.2964310301204501, 0.13190483696240127,
#         0.6939537051210115, 0.8330800031639156, 0.17347610820526463, 0.8420321208563623,
#         0.8882934142237333, 0.7356394920285989, 0.15360866990050534, 0.24664740708336055,
#         0.17509971698083543, 0.1276861973423668, 0.8057327549388998, 0.41178926421868306,
#         0.2908397350173472, 0.19836039970638378],  # happy
#     1: [0.2470265618083506, 0.1433979246927842, 0.6881822893577378, 0.11893982182697348,
#         0.31161634476423544, 0.5253221085983302, 0.12597712051107948, 0.4422623779963743,
#         0.27645211415344356, 0.42395131504088984, 0.17558170449960314, 0.27142133890425624,
#         0.1741388696479932, 0.1227455638528141, 0.4485394778155102, 0.41664204112281555,
#         0.2908397350173472, 0.2121307850723057],  # sad
#     2: [0.19974525492944142, 0.17419319637385966, 0.4053456334801515, 0.13706230525786608,
#         0.16464150662864904, 0.3564013123982043, 0.1117065857762667, 0.30225692349830036,
#         0.17199394819786643, 0.28648655803768563, 0.14721999753281687, 0.22595757296022984,
#         0.14108230584951664, 0.11916179838996314, 0.31830472814177574, 0.36213049142283893,
#         0.2908397350173472, 0.16214811486455724],  # neutral
#     3: [0.16978330014266482, 0.1474484099458089, 0.6385375278419455, 0.15955087911488458,
#         0.2257386547042889, 0.4972455283651611, 0.15269462267081732, 0.470889570126452,
#         0.1713443932394683, 0.28682403662377504, 0.17023111167232707, 0.28941226670462944,
#         0.14671370568908396, 0.12489530852760639, 0.6019091910383833, 0.5023565940498623,
#         0.2908397350173472, 0.1480589426871144],  # angry
#     4: [0.3340537870187493, 0.31316038897162674, 0.4508609549567622, 0.2215642997098194,
#         0.18256790904316805, 0.36794740171201545, 0.14342865607335295, 0.3325723523253354,
#         0.20291790550542987, 0.24479399356303455, 0.15335881131831128, 0.24479399356303455,
#         0.14621845929676547, 0.12047847974103684, 0.4921072671708563, 0.5073560806954985,
#         0.2908397350173472, 0.15234747176484023],  # surprise
#     5: [0.2556613250379876, 0.15655681634966592, 0.7784678175136164, 0.13563151267233145,
#         0.353602732625092, 0.8594560438530944, 0.3484770561236273, 0.7265667054811852,
#         0.2573762546624107, 0.4833627557878272, 0.26077234063862126, 0.3163550480772833,
#         0.1807057096038953, 0.1488894096862477, 0.6161944289089377, 0.44359769238463886,
#         0.2908397350173472, 0.21474224698158992],  # disgust
#     6: [0.3416233199942143, 0.22096625856984287, 0.591078697660434, 0.2210340665261555,
#         0.19066749128518232, 0.35671400783212986, 0.1239196009045801, 0.3647782154323708,
#         0.2014525981038694, 0.28661445976904537, 0.21138892966648018, 0.30233592318202696,
#         0.18447197078258057, 0.12964812274717674, 0.5106650931015151, 0.5397724065827655,
#         0.2908397350173472, 0.1432476989807878],  # fear
# }
K_SCALE = 5.0

POSW_GLOBAL_DFEW = [8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674,
                   4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794,
                   7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384,
                   12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372,
                   101.28938906752411, 8.483380533611566]

POSW_DISTINCT_DFEW = {
    0: [4.648862512363996, 4.162923411588553, 1.83239825175909, 2.5131103421760863, 1.0496164371270917, 1.191954326288771, 15.430099793221252, 0.636697444899202, 1.0033104960263086, 0.7217365088935785, 3.69328950409615, 2.862698681095705, 6.9568094740508535, 4.616744014506562, 2.920370688175734, 5.462006293978289, 57.59313882654697, 4.170958066889254],  # happy
    1: [5.6823671940967, 4.992658194508461, 0.5149984185907146, 2.6989371862870613, 3.1921004145555045, 1.6714365386873313, 19.07732186190161, 2.1056834274599465, 7.311081685767773, 1.9735191296108734, 5.039584577609662, 3.51384996900186, 8.076543333000897, 6.52494935714581, 3.628397792864953, 5.6926866933852995, 70.26898981989036, 5.224216933388045],  # sad
    2: [6.4774986002239645, 5.6010812480692, 0.8710279064472912, 1.9167337801498792, 7.8117860530331145, 2.7351547887496284, 21.302160526041124, 3.19245786489297, 20.999073406774425, 2.574808023689626, 5.714757086292502, 3.906024704963953, 8.191199242945629, 5.206488904380156, 4.249791165053314, 7.363091976516634, 81.4053220208253, 4.963300960035722],  # neutral
    3: [4.928812812224508, 4.114906832298137, 0.9234234234234234, 1.4543961558346765, 4.781224255883091, 2.435997871208089, 12.75189571440743, 1.44547134935305, 13.252185430463577, 3.1313061506565307, 3.579413266753674, 2.578767654819184, 5.92967542503864, 5.292631578947368, 2.6113572291582763, 4.433813627794237, 55.85311729482212, 4.549840112780662],  # angry
    4: [7.116157728166966, 6.34866790582404, 0.9600900658968373, 0.761682850299846, 17.648977987421382, 5.163029358274876, 21.278938718008924, 6.183435536376713, 35.41059094397544, 6.815336463223788, 6.358355951919348, 4.08091030789826, 9.571078431372548, 7.167857450288371, 5.090243902439024, 8.104394549990404, 94.26706827309236, 6.122504128509233],  # surprise
    5: [4.229214780600462, 4.09392575928009, 0.7474435655026047, 2.1834797891036906, 5.054144385026738, 1.6915304606240713, 10.307116104868914, 2.2381122631390777, 10.731865284974093, 3.210599721059972, 4.137266023823029, 3.012848914488259, 5.983037779491133, 6.148382004735596, 3.6374807987711213, 5.387165021156559, 27.391849529780565, 3.2964895635673623],  # disgust
    6: [5.96072648535643, 5.154494891980657, 0.7569813845450734, 1.5509251975236857, 7.060588375159415, 2.779909786630176, 17.84796563052818, 2.8554517069539505, 15.333362533397574, 3.2621352565348083, 4.923954312221004, 3.5370230679384855, 7.644713355124371, 5.762544656620061, 3.9785987023043443, 6.670773851153989, 57.87385538364383, 5.229860670252932],  # fear
}

# minor strategy: define which classes are "major"
MAJOR_CLASSES_DFEW = {0, 1, 2, 3}

def compute_au_loss_stage_b(au_logits, au_targets, labels, posw_option='global'):
    """
    au_logits:  [B,18] float
    au_targets: [B,18] float {0,1}
    labels:     [B] long, emotion class ids (0..C-1)
    """
    device = au_logits.device
    B = labels.shape[0]
    # pos_weight: [B,18]
    if config.DATA.DATASET=='DFEW':
        # knowledge weight: [B,18]
        k = torch.stack(
            [torch.tensor(K_TABLE_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
            dim=0
        ) * K_SCALE
        if posw_option == 'global':
            pw = torch.tensor(POSW_GLOBAL_DFEW, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
        elif posw_option == 'distinct':
            pw = torch.stack(
                [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
                dim=0
            )
        elif posw_option == 'minor':
            pw = torch.stack(
                [torch.tensor(POSW_DISTINCT_DFEW[int(c)], device=device, dtype=torch.float32) for c in labels],
                dim=0
            )
            # overwrite major classes with 1s
            mask_major = torch.tensor([int(c) in MAJOR_CLASSES_DFEW for c in labels], device=device, dtype=torch.bool)
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
        start_epoch, _ = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

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
        if config.DATA.DATASET.lower() == 'dfew':
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
                         is_best)

    # Print best epoch summary
    logger.info(f"\n" + "=" * 80)
    logger.info(f"TRAINING COMPLETED - BEST MODEL SUMMARY")
    logger.info(f"=" * 80)
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"-" * 80)

    # Get emotion class names
    if config.DATA.DATASET.lower() == 'dfew':
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
    logger.info("Loading best model for final evaluation...")
    best_model_path = os.path.join(config.OUTPUT, 'best.pth')
    checkpoint = torch.load(best_model_path, map_location='cpu')
    model.module.load_state_dict(checkpoint['model'])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    config.defrost()
    config.TEST.NUM_CLIP = 1
    config.TEST.NUM_CROP = 1
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

        # # Original code - commented out
        # if mixup_fn is not None:
        #     images, label_id = mixup_fn(images, label_id)

        # if texts.shape[0] == 1:
        #     texts = texts.view(1, -1)
        # with autocast():
        #     output_global, output_local, feat, au_logits = model(images, texts)
        #     if epoch>13:
        #         total_loss = criterion(output_global, label_id)
        #         total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
        #     else:
        #         pre_output_local = torch.sum(
        #             output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
        #         # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
        #         total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
        #         total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
        #     if config.AU.ENABLED and mixup_fn is None:
        #         au_targets = batch_data["au"].cuda(non_blocking=True).float()
        #         au_loss = compute_au_loss_stage_b(
        #             au_logits, au_targets, label_id,
        #             posw_option=config.AU.POSW_OPTION
        #         )
        #         total_loss = total_loss + config.AU.LAMBDA * au_loss
        #         #total_loss = (1-config.AU.LAMBDA) * total_loss + config.AU.LAMBDA * au_loss

        # Modified code - Apply mixup to both images/labels and AU labels
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
                pre_output_local = torch.sum(
                    output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
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
                total_loss = total_loss + config.AU.LAMBDA * au_loss
                #total_loss = total_loss * (1 - config.AU.LAMBDA) + config.AU.LAMBDA * au_loss

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
    # save_path = 'AU-CLIP/results'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # labels_name = ['hap', 'sad', 'neu', 'ang', 'sur', 'dis', 'fea']
    # plot_confusion_matrix(cm, labels_name, 'AUCLIP', cls_acc)
    #
    # #t-SNE
    # col = ['orange', 'purple', 'g', 'r', 'darkblue', 'chocolate', 'c']
    # x_embed = TSNE(n_components=2, perplexity=100, n_iter=10000).fit_transform(feature.data.cpu())
    # label = np.array(video_label)
    # plt.figure(figsize=(6, 6))
    # for i in range(7):
    #     idxs = np.where(label == i)[0]
    #     plt.scatter(x_embed[idxs, 0], x_embed[idxs, 1], color=col[i], s=6, label=labels_name[i])
    # plt.legend(loc='upper left')
    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)
    # plt.savefig(os.path.join(save_path, 'AUCLIP_TSNE.jpg'), format='jpg')
    # plt.show()

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

    # 初始化分布式环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])  # 使用环境变量中的 LOCAL_RANK
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
        local_rank = args.local_rank  # 如果未设置环境变量，则使用命令行参数

    # 设置当前 GPU 设备
    torch.cuda.set_device(local_rank)

    # 初始化分布式进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # 确保所有进程同步
    dist.barrier()

    # 设置随机种子
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # 创建输出目录
    output_dir = Path(config.OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志记录器
    logger = create_logger(output_dir=output_dir, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"Working directory: {output_dir}")

    # 保存配置文件（仅主进程执行）
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, output_dir)

    # 启动主训练逻辑
    main(config)