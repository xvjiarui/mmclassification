import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn.bricks import ContextBlock
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.show or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--show" or '
         '"--show-dir"')

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    assert not distributed
    model = MMDataParallel(model, device_ids=[0])
    single_gpu_vis(model, data_loader, args.show, args.show_dir)


hidden_outputs = {}


def activation_hook(name):

    def hook(module, input, output):
        # [N, C, 1, 1]
        x = input[0]
        context = module.spatial_pool(x)

        assert module.channel_add_conv is not None
        # [N, C, 1, 1]
        channel_add_term = module.channel_add_conv(context)
        hidden_outputs[name] = channel_add_term.squeeze(-1).squeeze(-1)

    return hook


def register_activation_hook(model):
    for module_name, module in model.module.named_modules():
        if isinstance(module, ContextBlock):
            module.register_forward_hook(activation_hook(module_name))
            print(f'{module_name} is registered')


activations = dict()


def single_gpu_vis(model, data_loader, show=False, out_dir=None):
    model.eval()
    register_activation_hook(model)

    dataset = data_loader.dataset
    dataset_length = len(dataset)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        batch_size = data['img'].size(0)
        with torch.no_grad():
            model(return_loss=True, **data)

        gt_label = data['gt_label'].cuda()

        for name in hidden_outputs:
            hidden_output = hidden_outputs[name].view(batch_size, -1)
            if name not in activations:
                activations[name] = hidden_output.new_zeros(
                    1000, hidden_output.shape[-1]) / dataset_length
            activations[name].scatter_add_(
                0,
                gt_label.unsqueeze(1).expand_as(hidden_output), hidden_output)

        hidden_outputs.clear()

        for _ in range(batch_size):
            prog_bar.update()
    for name in activations:
        activations[name] = activations[name].detach().cpu().numpy()
    mmcv.dump(activations, osp.join(out_dir, 'activations.pkl'))
    mapping = mmcv.load('imagenet_class_index.json')
    vis_indices = [1, 254, 726, 972]
    for name in activations:
        labels = []
        for vis_index in vis_indices:
            class_name = mapping[str(vis_index)]

            x = np.arange(activations[name].shape[1])
            y = activations[name][vis_index]
            plt.plot(x, y)
            labels.append(class_name)
        class_name = 'all'
        x = np.arange(activations[name].shape[1])
        y = activations[name].sum(0)
        plt.plot(x, y)
        labels.append(class_name)

    plt.legend(
        labels,
        ncol=1,
        loc='upper right',
        columnspacing=2.0,
        labelspacing=1,
        handletextpad=0.5,
        handlelength=1.5,
        fancybox=True,
        shadow=True)
    plt.ylabel('Context Amplitude', fontsize=15, labelpad=15)
    plt.xlabel('Channel Index', fontsize=15, labelpad=15)
    mmcv.mkdir_or_exist(out_dir)
    if show:
        plt.show()
    else:
        fig_name = os.path.join(out_dir, 'act.png')
        plt.savefig(fig_name)


if __name__ == '__main__':
    main()
