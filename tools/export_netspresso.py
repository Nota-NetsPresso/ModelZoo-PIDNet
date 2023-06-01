# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os

import torch

import _init_paths
import models
from configs import config
from configs import update_config
from utils.utils import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    # build model
    model_model, model_head = models.pidnet.get_netspresso_model(config, imgnet_pretrained=True)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pt')
   
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
    torch.save(traced_model, "/PIDNet_nota/model_modelfx.pt")
    logger.info('=> saving model_model torchfx to /PIDNet/model_modelfx.pt')
    
    #save model_head
    model_head.train()
    torch.save(model_head, "/PIDNet_nota/model_head.pt")
    logger.info('=> saving model_head torchfx to /PIDNet/model_head.pt')
    
if __name__ == '__main__':
    main()