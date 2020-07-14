import argparse
import os.path as osp
import os
from collections import defaultdict

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import init_dist, load_checkpoint

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcv import tensor2imgs
import matplotlib.pyplot as plt
import numpy as np



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
         'results / save the results) with the argument "--show" or "--show-dir"')

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
        hidden_outputs[name] = output

    return hook


def register_activation_hook(model):
    for module_name, module in model.module.named_modules():
        if 'layer3' in module_name and 'relu' in module_name:
            module.register_forward_hook(activation_hook(module_name))
            print(f'{module_name} is registered')


activations = dict()


def single_gpu_vis(model,
                   data_loader,
                   show=False,
                   out_dir=None):
    model.eval()
    register_activation_hook(model)

    dataset = data_loader.dataset
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
                    1000, hidden_output.shape[-1])
            activations[name].scatter_add_(0, gt_label.unsqueeze(
                1).expand_as(hidden_output), hidden_output)

        hidden_outputs.clear()

        for _ in range(batch_size):
            prog_bar.update()
    result = {}
    for name in activations:
        layer_act = activations[name]

        num_class, num_neuron = layer_act.size()
        dead_neuron_class = torch.tensor(num_class).cuda()
        dead_neuron_confidence = torch.tensor(0.).cuda()

        selected_class = []
        selectivity_index = []
        for neuron_idx in range(num_neuron):
            neuron_act = layer_act[:, neuron_idx]

            # In the case of mean activations of a neuron are all zero across whole classes
            # Simply consider that neuron as dead neuron.
            if neuron_act.nonzero().size(0) == 0:
                continue
            class_selected = neuron_act.argmax()
            mu_max = neuron_act[class_selected]
            mu_mmax = (neuron_act.sum() - mu_max).div(num_class - 1)
            class_confidence = (mu_max - mu_mmax).div(mu_max + mu_mmax)

            selected_class.append(class_selected)
            selectivity_index.append(class_confidence)

        selected_class = torch.stack(selected_class, 0)
        selectivity_index = torch.stack(selectivity_index, 0)

        result[name] = dict()
        result[name]['selected_class'] = selected_class.cpu().numpy()
        result[name]['selectivity_index'] = selectivity_index.cpu().numpy()

    mmcv.dump(result, osp.join(out_dir, 'result.pkl'))
    num_plots = len(result)
    colormap = plt.cm.jet
    plt.figure(figsize=(10, 10))
    plt.gca().set_prop_cycle(color=
        [colormap(i) for i in np.linspace(0, 0.9, num_plots)])

    labels = []
    for name in result:
        selectivity_index = result[name]['selectivity_index']
        selectivity_index_hist = np.histogram(selectivity_index * 100,
                                              bins=100, normed=True)
        x = np.arange(len(selectivity_index_hist[0])) / len(
            selectivity_index_hist[0])
        y = selectivity_index_hist[0]
        plt.fill_between(x, y, step="pre", alpha=0.6)
        plt.plot(x, y)
        labels.append(name)

    plt.legend(labels, ncol=1, loc='upper right',
               columnspacing=2.0, labelspacing=1,
               handletextpad=0.5, handlelength=1.5,
               fancybox=True, shadow=True)
    plt.ylabel('PDF', fontsize=15, labelpad=15)
    plt.xlabel('Selectivity Index', fontsize=15, labelpad=15)

    mmcv.mkdir_or_exist(out_dir)
    if show:
        plt.show()
    else:
        fig_name = os.path.join(out_dir, 'histogram.png')
        plt.savefig(fig_name)



if __name__ == '__main__':
    main()
