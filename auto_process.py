import argparse
import os
import pprint
import sys

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
from netspresso.compressor import ModelCompressor, Task, Framework

import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import train, validate
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    """
        Common arguments
    """
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str
    )
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
    )

    """
        Compression arguments
    """
    parser.add_argument(
        "--compression_method",
        type=str,
        choices=["PR_L2", "PR_GM", "PR_NN", "PR_ID", "FD_TK", "FD_CP", "FD_SVD"],
        default="PR_L2"
    )
    parser.add_argument(
        "--recommendation_method",
        type=str,
        choices=["slamp", "vbmf"],
        default="slamp"
    )
    parser.add_argument(
        "--compression_ratio",
        type=int,
        default=0.5
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        type=str
    )
    parser.add_argument(
        "-m",
        "--np_email",
        help="NetsPresso login e-mail",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--np_password",
        help="NetsPresso login password",
        type=str,
    )


    args = parser.parse_args()
    update_config(config, args)

    return args

def trainer(args, config, logger, final_output_dir, tb_log_dir, model, head):
    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)

    model_model = torch.load(model)
    model_head = torch.load(head)

    model = nn.Sequential(
        model_model,
        model_head
    )
 
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)

    bd_criterion = BondaryLoss()
    
    model = FullModel(model, sem_criterion, bd_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    best_mIoU = 0
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME
    if config.TRAIN.RESUME and not args.netspresso:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict)

        if flag_rm == 1 or (epoch % 5 == 0 and epoch < real_end - 100) or (epoch >= real_end - 100):
            valid_loss, mean_IoU, IoU_array = validate(config, 
                        testloader, model, writer_dict)
        if flag_rm == 1:
            flag_rm = 0
            
        #Netspresso_ 분리해서 저장하는 코드 작성
        logger.info('=> saving model_model to {}'.format(final_output_dir + 'retrain_model_model.pt'))
        torch.save(model_model, os.path.join(final_output_dir,'retrain_model_model.pt'))
        logger.info('=> saving model_head to {}'.format(final_output_dir + 'retrain_model_head.pt'))
        torch.save(model_head, os.path.join(final_output_dir,'retrain_model_head.pt'))
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model_model, os.path.join(final_output_dir, 'best_model_model.pt'))
            torch.save(model_head, os.path.join(final_output_dir, 'best_model_head.pt'))
        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                    valid_loss, mean_IoU, best_mIoU)

        logging.info(msg)
        logging.info(IoU_array)
        
    torch.save(model_model, os.path.join(final_output_dir, 'final_model_model.pt'))
    torch.save(model_head, os.path.join(final_output_dir, 'final_model_head.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % np.int((end-start)/3600))
    logger.info('Done')


if __name__ == "__main__":
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test')
    
    """ 
        Convert PIDNet model to fx 
    """
    logger.info("PIDNet to fx graph start.")
    # build model
    model_model, model_head = models.pidnet.get_netspresso_model(config, imgnet_pretrained=True)

    model_state_file = args.weight_path
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
        
    #model_model    
    model_model_dict = model_model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_model_dict.update(pretrained_dict)
    
    model_model.load_state_dict(model_model_dict)
    
    #model_head
    model_head_dict = model_head.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_head_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_head_dict.update(pretrained_dict)
    
    model_head.load_state_dict(model_head_dict)
    
    import torch.fx as fx
    
    #save model_model
    model_model.train()
    _graph = fx.Tracer().trace(model_model)
    traced_model = fx.GraphModule(model_model, _graph)
    torch.save(traced_model, config.MODEL.NAME + '_fx.pt')
    logger.info('=> saving model_model torchfx')
    
    #save model_head
    model_head.train()
    torch.save(model_head, config.MODEL.NAME + '_head_fx.pt')
    logger.info('=> saving model_head torchfx')

    logger.info("PIDNet to fx graph end.")

    """ 
        Model compression - recommendation compression 
    """
    logger.info("Compression step start.")

    compressor = ModelCompressor(email=args.np_email, password=args.np_password)

    UPLOAD_MODEL_NAME = config.MODEL.NAME
    TASK = Task.OBJECT_DETECTION
    FRAMEWORK = Framework.PYTORCH
    UPLOAD_MODEL_PATH = config.MODEL.NAME + '_fx.pt'
    INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": config.TRAIN.IMAGE_SIZE}]
    model = compressor.upload_model(
        model_name=UPLOAD_MODEL_NAME,
        task=TASK,
        framework=FRAMEWORK,
        file_path=UPLOAD_MODEL_PATH,
        input_shapes=INPUT_SHAPES,
    )
    
    COMPRESSION_METHOD = args.compression_method
    RECOMMENDATION_METHOD = args.recommendation_method
    RECOMMENDATION_RATIO = args.compression_ratio
    COMPRESSED_MODEL_NAME = f'{UPLOAD_MODEL_NAME}_{COMPRESSION_METHOD}_{RECOMMENDATION_RATIO}'
    OUTPUT_PATH = COMPRESSED_MODEL_NAME + '.pt'
    compressed_model = compressor.recommendation_compression(
        model_id=model.model_id,
        model_name=COMPRESSED_MODEL_NAME,
        compression_method=COMPRESSION_METHOD,
        recommendation_method=RECOMMENDATION_METHOD,
        recommendation_ratio=RECOMMENDATION_RATIO,
        output_path=OUTPUT_PATH,
    )

    logger.info("Compression step end.")

    """ 
        Retrain PIDNet model 
    """
    logger.info("Fine-tuning step start.")

    trainer(args, config, logger, final_output_dir, tb_log_dir, 
            model=COMPRESSED_MODEL_NAME + '.pt', 
            head=config.MODEL.NAME + '_head_fx.pt')

    logger.info("Fine-tuning step end.")
