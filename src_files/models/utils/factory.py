import os

import torch
import timm
from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.models.model_store import get_model_file

from ..ofa.model_zoo import ofa_flops_595m_s
from ..tresnet import TResnetM, TResnetL
from src_files.helper_functions.distributed import print_at_master


def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if 'state_dict' in state:
            state = state['state_dict']
        if key in state:
            ip = state[key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print_at_master(
                    'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print_at_master('could not load layer: {}, not in checkpoint'.format(key))
    return model


def create_model(args):
    print_at_master('creating model {}...'.format(args.model_name))

    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name == 'ofa_flops_595m_s':
        model = ofa_flops_595m_s(model_params)
    elif args.model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=False, num_classes=args.num_classes)
    elif args.model_name == 'vit_base_patch16_224': # notice - qkv_bias==False currently
        model_kwargs = dict(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=None, qkv_bias=False)
        model = timm.models.vision_transformer._create_vision_transformer('vit_base_patch16_224_in21k',
                                                                          pretrained=False,
                                                                          num_classes=args.num_classes, **model_kwargs)
    elif args.model_name == 'mobilenetv3_large_100':
        model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=args.num_classes)
    elif args.model_name == 'mobilenetv2_w1':
        model = ptcv_get_model(args.model_name, pretrained=False, num_classes=args.num_classes)
        file_path = get_model_file(model_name=args.model_name,
                                   local_model_store_dir_path=os.path.join("~", ".torch", "models"))
        model = load_model_weights(model, file_path)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    if args.model_path and args.model_path!='':  # make sure to load pretrained ImageNet-1K model
        model = load_model_weights(model, args.model_path)
    print('done\n')

    return model
