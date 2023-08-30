import argparse
import os

import torch
from netspresso.compressor import ModelCompressor, Task, Framework

import models
from configs import config
from configs import update_config
from utils.utils import create_logger


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

if __name__ == "__main__":
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
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
