from datetime import datetime
import os
import os.path as osp

# PyTorch includes
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer

# Custom includes
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *
from networks.GAN import BoundaryDiscriminator, UncertaintyDiscriminator


here = osp.dirname(osp.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')

    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--datasetS', type=str, default='refuge', help='test folder id contain images ROIs to test'
    )
    parser.add_argument(
        '--datasetT', type=str, default='Drishti-GS', help='refuge / Drishti-GS/ RIM-ONE_r3'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8, help='batch size for training the model'
    )
    parser.add_argument(
        '--group-num', type=int, default=1, help='group number for group normalization'
    )
    parser.add_argument(
        '--max-epoch', type=int, default=200, help='max epoch'
    )
    parser.add_argument(
        '--stop-epoch', type=int, default=200, help='stop epoch'
    )
    parser.add_argument(
        '--warmup-epoch', type=int, default=-1, help='warmup epoch begin train GAN'
    )

    parser.add_argument(
        '--interval-validate', type=int, default=10, help='interval epoch number to valide the model'
    )
    parser.add_argument(
        '--lr-gen', type=float, default=1e-3, help='learning rate',
    )
    parser.add_argument(
        '--lr-dis', type=float, default=2.5e-5, help='learning rate',
    )
    parser.add_argument(
        '--lr-decrease-rate', type=float, default=0.1, help='ratio multiplied to initial lr',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--data-dir',
        default='/home/sjwang/ssd1T/fundus/domain_adaptation/',
        help='data root path'
    )
    parser.add_argument(
        '--pretrained-model',
        default='../../../models/pytorch/fcn16s_from_caffe.pth',
        help='pretrained model of FCN16s',
    )
    parser.add_argument(
        '--out-stride',
        type=int,
        default=16,
        help='out-stride of deeplabv3+',
    )
    parser.add_argument(
        '--sync-bn',
        type=bool,
        default=True,
        help='sync-bn in deeplabv3+',
    )
    parser.add_argument(
        '--freeze-bn',
        type=bool,
        default=False,
        help='freeze batch normalization of deeplabv3+',
    )

    args = parser.parse_args()

    args.model = 'FCN8s'

    now = datetime.now()
    args.out = osp.join(here, 'logs', args.datasetT, now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)


    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(512),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.RandomCrop(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetS, split='train',
                                                         transform=composed_transforms_tr)
    domain_loaderS = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    domain_T = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='train',
                                                             transform=composed_transforms_tr)
    domain_loaderT = DataLoader(domain_T, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    domain_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='train',
                                       transform=composed_transforms_ts)
    domain_loader_val = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. model
    model_gen = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                        sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    model_dis = BoundaryDiscriminator().cuda()
    model_dis2 = UncertaintyDiscriminator().cuda()
    
    start_epoch = 0
    start_iteration = 0

    # 3. optimizer

    optim_gen = torch.optim.Adam(
        model_gen.parameters(),
        lr=args.lr_gen,
        betas=(0.9, 0.99)
    )
    optim_dis = torch.optim.SGD(
        model_dis.parameters(),
        lr=args.lr_dis,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optim_dis2 = torch.optim.SGD(
        model_dis2.parameters(),
        lr=args.lr_dis,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model_gen.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model_gen.load_state_dict(model_dict)

        pretrained_dict = checkpoint['model_dis_state_dict']
        model_dict = model_dis.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_dis.load_state_dict(model_dict)

        pretrained_dict = checkpoint['model_dis2_state_dict']
        model_dict = model_dis2.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_dis2.load_state_dict(model_dict)


        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iteration'] + 1
        optim_gen.load_state_dict(checkpoint['optim_state_dict'])
        optim_dis.load_state_dict(checkpoint['optim_dis_state_dict'])
        optim_dis2.load_state_dict(checkpoint['optim_dis2_state_dict'])

    trainer = Trainer.Trainer(
        cuda=cuda,
        model_gen=model_gen,
        model_dis=model_dis,
        model_uncertainty_dis=model_dis2,
        optimizer_gen=optim_gen,
        optimizer_dis=optim_dis,
        optimizer_uncertainty_dis=optim_dis2,
        lr_gen=args.lr_gen,
        lr_dis=args.lr_dis,
        lr_decrease_rate=args.lr_decrease_rate,
        val_loader=domain_loader_val,
        domain_loaderS=domain_loaderS,
        domain_loaderT=domain_loaderT,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
        warmup_epoch=args.warmup_epoch,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    main()
