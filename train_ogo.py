#!/usr/bin/env python
# encoding: utf-8
# Author: Kennard Ng
# Create Time: 2020/10/29 16:44
# TODO:

import argparse
import copy
import os
import time
import warnings

import dgl
import numpy as np
import psutil
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from gcc.contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from gcc.contrastive.memory_moco import MemoryMoCo
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
)
from gcc.datasets.data_util import batcher, labeled_batcher
from gcc.models import GraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear
from ogo.ogo_models import OddGraphOutPredictionHead
from ogo.datasets.graph_dataset import OddGraphOutDataset
from ogo.datasets.utils import ogo_batcher


def parse_option():

    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb-freq", type=int, default=250, help="tb frequency")
    parser.add_argument("--save-freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch-size", type=int, default=32, help="batch_size")
    parser.add_argument("--num-workers", type=int, default=12, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=6, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=2000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")

    # optimization
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam', 'adagrad'], help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.0, help="decay rate for learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

    # resume
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # augmentation setting
    parser.add_argument("--aug", type=str, default="1st", choices=["1st", "2nd", "all"])

    parser.add_argument("--exp", type=str, default="")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="dgl",
                        choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde", "h-index-rand-1", "h-index-top-1", "h-index"] + GRAPH_CLASSIFICATION_DSETS)

    # model definition
    parser.add_argument("--model", type=str, default="gin", choices=["gat", "mpnn", "gin"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=5, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")

    # loss function
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)
    parser.add_argument("--num_graphs", type=int, default=6)
    parser.add_argument("--fusion_method", type=str, default="cosine-sod")

    # random walk
    parser.add_argument("--rw-hops", type=int, default=256)
    parser.add_argument("--subgraph-size", type=int, default=128)
    parser.add_argument("--restart-prob", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--freq-embedding-size", type=int, default=16)
    parser.add_argument("--degree-embedding-size", type=int, default=16)

    # specify folder
    parser.add_argument("--model-path", type=str, default=None, help="path to save model")
    parser.add_argument("--tb-path", type=str, default=None, help="path to tensorboard")
    parser.add_argument("--load-path", type=str, default=None, help="loading checkpoint at test time")

    # memory setting
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")

    # finetune setting
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")

    # GPU setting
    parser.add_argument("--gpu", default=None, type=int, nargs='+', help="GPU id to use.")

    # cross validation
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--fold-idx", type=int, default=0, help="random seed.")
    parser.add_argument("--cv", action="store_true")
    # fmt: on

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def option_update(opt):
    opt.model_name = "ogo/{}_ogo{}_{}_{}_layer_{}_lr_{}_decay_{}_bsz_{}_hid_{}_samples_{}_nce_t_{}_nce_k_{}_rw_hops_{}_restart_prob_" \
                     "{}_aug_{}_ft_{}_deg_{}_pos_{}_momentum_{}_ngraphs_{}".format(
        opt.exp,
        opt.moco,
        opt.dataset,
        opt.model,
        opt.num_layer,
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.hidden_size,
        opt.num_samples,
        opt.nce_t,
        opt.nce_k,
        opt.rw_hops,
        opt.restart_prob,
        opt.aug,
        opt.finetune,
        opt.degree_embedding_size,
        opt.positional_embedding_size,
        opt.alpha,
        opt.num_graphs
    )

    if opt.load_path is None:
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
    else:
        opt.model_folder = opt.load_path

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    return opt


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def train_finetune(
    epoch,
    train_loader,
    model,
    output_layer,
    criterion,
    optimizer,
    output_layer_optimizer,
    sw,
    opt,
):
    """
    one epoch training for moco
    """
    n_batch = len(train_loader)
    model.train()
    output_layer.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    f1_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    epoch_f1_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, y = batch

        graph_q.to(torch.device(opt.gpu))
        y = y.to(torch.device(opt.gpu))

        bsz = graph_q.batch_size

        # ===================forward=====================

        feat_q = model(graph_q)

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)
        out = output_layer(feat_q)

        loss = criterion(out, y)

        # ===================backward=====================
        optimizer.zero_grad()
        output_layer_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        torch.nn.utils.clip_grad_value_(output_layer.parameters(), 1)
        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear(
            global_step / (opt.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        for param_group in output_layer_optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()
        output_layer_optimizer.step()

        preds = out.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")

        # ===================meters=====================
        f1_meter.update(f1, bsz)
        epoch_f1_meter.update(f1, bsz)
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        graph_size.update(graph_q.number_of_nodes() / bsz, bsz)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "f1 {f1.val:.3f} ({f1.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    f1=f1_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            sw.add_scalar("ft_loss", loss_meter.avg, global_step)
            sw.add_scalar("ft_f1", f1_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("lr", lr_this_step, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            #  sw.add_scalar(
            #      "learning_rate", optimizer.param_groups[0]["lr"], global_step
            #  )
            loss_meter.reset()
            f1_meter.reset()
            graph_size.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg, epoch_f1_meter.avg


def test_finetune(epoch, valid_loader, model, output_layer, criterion, sw, opt):
    n_batch = len(valid_loader)
    model.eval()
    output_layer.eval()

    epoch_loss_meter = AverageMeter()
    epoch_f1_meter = AverageMeter()

    for idx, batch in enumerate(valid_loader):
        graph_q, y = batch

        graph_q.to(torch.device(opt.gpu))
        y = y.to(torch.device(opt.gpu))

        bsz = graph_q.batch_size

        # ===================forward=====================

        with torch.no_grad():
            feat_q = model(graph_q)
            assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)
            out = output_layer(feat_q)
        loss = criterion(out, y)

        preds = out.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")

        # ===================meters=====================
        epoch_loss_meter.update(loss.item(), bsz)
        epoch_f1_meter.update(f1, bsz)

    global_step = (epoch + 1) * n_batch
    sw.add_scalar("ft_loss/valid", epoch_loss_meter.avg, global_step)
    sw.add_scalar("ft_f1/valid", epoch_f1_meter.avg, global_step)
    print(
        f"Epoch {epoch}, loss {epoch_loss_meter.avg:.3f}, f1 {epoch_f1_meter.avg:.3f}"
    )
    return epoch_loss_meter.avg, epoch_f1_meter.avg


def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )


def train_ogo(epoch: int, train_loader, model, prediction_head: OddGraphOutPredictionHead, optimizer, sw, opt):
    """ performs training for one epoch using odd graph out pretraining """
    num_batches = train_loader.dataset.total // opt.batch_size
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    graph_size = AverageMeter()
    gnorm_meter = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    criterion = nn.CrossEntropyLoss()
    num_graphs_per_batch = prediction_head.num_embeddings
    end = time.time()
    multinomial_weights = torch.ones(num_graphs_per_batch, dtype=torch.float) / num_graphs_per_batch
    multinomial_weights = multinomial_weights.to(torch.device(opt.gpu))
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graphs = batch
        graphs = graphs.to(torch.device(opt.gpu))

        num_total_graphs = graphs.batch_size
        batchsize = num_total_graphs // num_graphs_per_batch
        assert batchsize * num_graphs_per_batch == num_total_graphs

        graph_features = model(graphs)
        graph_features = graph_features.view(batchsize, num_graphs_per_batch, opt.hidden_size)

        targets = torch.multinomial(multinomial_weights, num_samples=batchsize, replacement=True)
        targets = targets.view(-1)

        # swap the graph with the negative graph; the negative graph can be at the same location.
        swap_idxs = targets + torch.arange(len(targets), device=torch.device(opt.gpu)) * num_graphs_per_batch
        flattened_features = graph_features.view(num_total_graphs, opt.hidden_size)
        swap_features = flattened_features[swap_idxs, :]
        flattened_features[swap_idxs, :] = graph_features[:, -1, :]  # replace with negative graph
        graph_features = flattened_features.view(batchsize, num_graphs_per_batch, opt.hidden_size)
        graph_features[:, -1, :] = swap_features  # replace with swapped graph

        # get the predictions
        logits = prediction_head(graph_features)

        # ===================backward=====================
        optimizer.zero_grad()
        loss = criterion(logits, targets)
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), opt.clip_norm)

        global_step = epoch * num_batches + idx
        lr_this_step = opt.learning_rate * warmup_linear(global_step / (opt.epochs * num_batches), 0.1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()

        # ===================prediction accuracy=====================
        predictions = torch.argmax(logits, dim=1)
        predictions = predictions.view(-1)
        is_corrects = torch.eq(predictions, targets)
        accuracy = torch.mean(is_corrects.float())

        # ===================meters=====================
        loss_meter.update(loss.item(), batchsize)
        epoch_loss_meter.update(loss.item(), batchsize)
        accuracy_meter.update(accuracy.item(), batchsize)
        graphs.number_of_nodes()
        graph_size.update(graphs.number_of_nodes() / num_total_graphs, num_total_graphs)
        gnorm_meter.update(grad_norm, 1)
        max_num_nodes = max(max_num_nodes, graphs.number_of_nodes())
        max_num_edges = max(max_num_edges, graphs.number_of_edges())

        # if opt.moco:
        #     moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "accuracy {prob.val:.3f} ({prob.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    num_batches,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    prob=accuracy_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global_step = epoch * num_batches + idx
            sw.add_scalar("moco_loss", loss_meter.avg, global_step)
            sw.add_scalar("moco_prob", accuracy_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, global_step)
            sw.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
            loss_meter.reset()
            accuracy_meter.reset()
            graph_size.reset()
            gnorm_meter.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg


# def main(args, trial):
def main(args):
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    checkpoint = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            pretrain_args = checkpoint["opt"]
            pretrain_args.fold_idx = args.fold_idx
            pretrain_args.gpu = args.gpu
            pretrain_args.finetune = args.finetune
            pretrain_args.resume = args.resume
            pretrain_args.cv = args.cv
            pretrain_args.dataset = args.dataset
            pretrain_args.epochs = args.epochs
            pretrain_args.num_workers = args.num_workers
            if args.dataset in GRAPH_CLASSIFICATION_DSETS:
                # HACK for speeding up finetuning on graph classification tasks
                pretrain_args.num_workers = 0
            pretrain_args.batch_size = args.batch_size
            args = pretrain_args
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    args = option_update(args)
    print(args)
    assert args.gpu is not None and torch.cuda.is_available()
    print("Use GPU: {} for training".format(args.gpu))
    assert args.positional_embedding_size % 2 == 0
    print("setting random seeds")

    mem = psutil.virtual_memory()
    print("before construct dataset", mem.used / 1024 ** 3)
    if args.finetune:
        if args.dataset in GRAPH_CLASSIFICATION_DSETS:
            dataset = GraphClassificationDatasetLabeled(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
            labels = dataset.dataset.data.y.tolist()
        else:
            dataset = NodeClassificationDatasetLabeled(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
            labels = dataset.data.y.argmax(dim=1).tolist()

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        assert (
            0 <= args.fold_idx and args.fold_idx < 10
        ), "fold_idx must be from 0 to 9."
        train_idx, test_idx = idx_list[args.fold_idx]
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        valid_dataset = torch.utils.data.Subset(dataset, test_idx)

    elif args.dataset == "dgl":
        train_dataset = OddGraphOutDataset(
            num_graphs=args.num_graphs,
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            dgl_graphs_file="./data/small.bin",
            num_copies=args.num_copies,
        )
    else:
        if args.dataset in GRAPH_CLASSIFICATION_DSETS:
            train_dataset = GraphClassificationDataset(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
        else:
            train_dataset = NodeClassificationDataset(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )

    mem = psutil.virtual_memory()
    print("before construct dataloader", mem.used / 1024 ** 3)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=ogo_batcher(),
        shuffle=True if args.finetune else False,
        num_workers=args.num_workers,
        worker_init_fn=None
        if args.finetune or args.dataset != "dgl"
        else worker_init_fn,
    )

    if args.finetune:
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            collate_fn=labeled_batcher(),
            num_workers=args.num_workers,
        )
    mem = psutil.virtual_memory()
    print("before training", mem.used / 1024 ** 3)

    # create model and optimizer
    # n_data = train_dataset.total
    n_data = None

    model = GraphEncoder(positional_embedding_size=args.positional_embedding_size,
                         max_node_freq=args.max_node_freq,
                         max_edge_freq=args.max_edge_freq,
                         max_degree=args.max_degree,
                         freq_embedding_size=args.freq_embedding_size,
                         degree_embedding_size=args.degree_embedding_size,
                         output_dim=args.hidden_size,
                         node_hidden_dim=args.hidden_size,
                         edge_hidden_dim=args.hidden_size,
                         num_layers=args.num_layer,
                         num_step_set2set=args.set2set_iter,
                         num_layer_set2set=args.set2set_lstm_layer,
                         norm=args.norm,
                         gnn_model=args.model,
                         degree_input=True)

    # create the prediction head.
    prediction_head = OddGraphOutPredictionHead(embedding_size=args.hidden_size, num_embeddings=args.num_graphs,
                                                method=args.fusion_method, batchnorm=True)

    model = model.cuda(args.gpu)
    prediction_head = prediction_head.cuda(args.gpu)

    # add optimizer for both prediction head and the graph encoder model
    params = list(model.parameters()) + list(prediction_head.parameters())
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params=params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params=params,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            params=params,
            lr=args.learning_rate,
            lr_decay=args.lr_decay_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        # print("=> loading checkpoint '{}'".format(args.resume))
        # checkpoint = torch.load(args.resume, map_location="cpu")
        # checkpoint = torch.load(args.resume)
        # args.start_epoch = checkpoint["epoch"] + 1
        assert checkpoint is not None and isinstance(checkpoint, dict)
        print(checkpoint)
        model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        prediction_head.load_state_dict(checkpoint["prediction-head"])

        print("=> loaded successfully '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        del checkpoint
        torch.cuda.empty_cache()

    sw = SummaryWriter(args.tb_folder)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        loss = train_ogo(epoch=epoch, train_loader=train_loader, model=model,
                         prediction_head=prediction_head, optimizer=optimizer,
                         sw=sw, opt=args)
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq == 0:
            print("==> Saving...")
            state = {
                "opt": args,
                "model": model.state_dict(),
                "prediction-head": prediction_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            save_file = os.path.join(
                args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
            # help release GPU memory
            del state

        # saving the model
        print("==> Saving...")
        state = {
            "opt": args,
            "model": model.state_dict(),
            "prediction-had": prediction_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        save_file = os.path.join(args.model_folder, "current.pth")
        torch.save(state, save_file)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()


if __name__ == "__main__":
    warnings.simplefilter("once", UserWarning)
    args = parse_option()

    if args.cv:
        gpus = args.gpu

        def variant_args_generator():
            for fold_idx in range(10):
                args.fold_idx = fold_idx
                args.num_workers = 0
                args.gpu = gpus[fold_idx % len(gpus)]
                yield copy.deepcopy(args)

        # f1 = Parallel(n_jobs=5)(
        #     delayed(main)(args) for args in variant_args_generator()
        # )
        f1 = [main(args) for args in variant_args_generator()]
        print(f1)
        print(f"Mean = {np.mean(f1)}; Std = {np.std(f1)}")
    else:
        args.gpu = args.gpu[0]
        main(args)
