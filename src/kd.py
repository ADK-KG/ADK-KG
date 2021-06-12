import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from ops import var_cuda, zeros_var_cuda
from collections import OrderedDict
from learn_framework import LFramework
import beam_search as search
from fact_network import get_conve_nn_state_dict, get_conve_kg_state_dict, \
    get_complex_kg_state_dict, get_distmult_kg_state_dict, get_TransE_kg_state_dict
import ops as ops
from ops import int_fill_var_cuda, var_cuda, zeros_var_cuda
import torch.nn as nn
CUDA_LAUNCH_BLOCKING=1
import os
import random
import shutil
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import eval
from ops import var_cuda, zeros_var_cuda
import ops as ops

def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, metrics, params):
    """
    Train the model on `num_steps` batches
    """

    model.train()
    teacher_model.eval()

    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), \
                                            labels_batch.cuda(async=True)
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            output_batch = model(train_batch)


            with torch.no_grad():
                output_teacher_batch = teacher_model(train_batch)
            if params.cuda:
                output_teacher_batch = output_teacher_batch.cuda(async=True)

            loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if i % params.save_summary_steps == 0:
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                       loss_fn_kd, metrics, params, model_dir, restore_file=None):
    """
    Train the model and evaluate every epoch.
    """
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    
    if params.model_version == "resnet18_distill":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "cnn_distill": 
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2) 

    for epoch in range(params.num_epochs):

        scheduler.step()

        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        train_kd(model, teacher_model, optimizer, loss_fn_kd, train_dataloader,
                 metrics, params)

        val_metrics = evaluate_kd(model, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        if is_best:
            logging.info("- Found new best")
            best_val_acc = val_acc

            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)