import sys
import datetime
import pathlib
import math

import numpy as np
import imageio
from PIL import Image
import scipy.stats as stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, jaccard_score

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import *
from utils import *

from dataloader import DataLoader
from networks.factory import model_factory

Image.MAX_IMAGE_PIXELS = None


def test(test_loader, model, epoch):
    # Setting network for evaluation mode.
    model.eval()

    track_cm = np.zeros((test_loader.dataset.num_classes, test_loader.dataset.num_classes))
    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):

            # Obtaining images, labels and paths for batch.
            inps, labels = data[0], data[1]

            # Casting to cuda variables.
            inps = Variable(inps).cuda()

            # Forwarding.
            outs = model(inps)

            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)
            prds = soft_outs.cpu().data.numpy().argmax(axis=1).flatten()
            labels = labels.flatten()

            # filtering out pixels
            coord = np.where(labels != test_loader.dataset.num_classes)
            labels = labels[coord]
            prds = prds[coord]

            track_cm += confusion_matrix(labels, prds, labels=[0, 1])

        acc = (track_cm[0][0] + track_cm[1][1]) / np.sum(track_cm)
        f1_s = f1_with_cm(track_cm)
        kappa = kappa_with_cm(track_cm)
        jaccard = jaccard_with_cm(track_cm)

        _sum = 0.0
        for k in range(len(track_cm)):
            _sum += (track_cm[k][k] / float(np.sum(track_cm[k])) if np.sum(track_cm[k]) != 0 else 0)
        nacc = _sum / float(test_loader.dataset.num_classes)

        print("---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(nacc) +
              " F1 Score= " + "{:.4f}".format(f1_s) +
              " Kappa= " + "{:.4f}".format(kappa) +
              " Jaccard= " + "{:.4f}".format(jaccard) +
              " Confusion Matrix= " + np.array_str(track_cm).replace("\n", "")
              )

        sys.stdout.flush()

    return acc, nacc, f1_s, kappa, track_cm


def test_full_map(test_loader, net, epoch, output_path):
    # Setting network for evaluation mode.
    net.eval()

    prob_im = np.zeros([test_loader.dataset.labels.shape[0],
                        test_loader.dataset.labels.shape[1],
                        test_loader.dataset.labels.shape[2], test_loader.dataset.num_classes], dtype=np.float32)
    occur_im = np.zeros([test_loader.dataset.labels.shape[0],
                         test_loader.dataset.labels.shape[1],
                         test_loader.dataset.labels.shape[2], test_loader.dataset.num_classes], dtype=int)

    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs, cur_maps, cur_xs, cur_ys = data

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs = net(inps_c)
            soft_outs = F.softmax(outs, dim=1)

            for j in range(outs.shape[0]):
                cur_map = cur_maps[j]
                cur_x = cur_xs[j]
                cur_y = cur_ys[j]

                soft_outs_p = soft_outs.permute(0, 2, 3, 1).cpu().detach().numpy()

                prob_im[cur_map][cur_x:cur_x + test_loader.dataset.crop_size,
                                 cur_y:cur_y + test_loader.dataset.crop_size, :] += soft_outs_p[j, :, :, :]
                occur_im[cur_map][cur_x:cur_x + test_loader.dataset.crop_size,
                                  cur_y:cur_y + test_loader.dataset.crop_size, :] += 1

        # normalize to remove non-predicted pixels - if there is one
        occur_im[np.where(occur_im == 0)] = 1

        # calculate predictions
        prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=-1)
        # pixels with classes not used in the prediction are converted into 0
        prob_im_argmax[np.where(test_loader.dataset.labels == 2)] = 0

        for k, img_n in enumerate(test_loader.dataset.images):
            # Saving predictions.
            imageio.imwrite(os.path.join(output_path, img_n + '_pred_epoch_' + str(epoch) + '.png'),
                            prob_im_argmax[k]*255)

        lbl = test_loader.dataset.labels.flatten()
        pred = prob_im_argmax.flatten()
        print(lbl.shape, np.bincount(lbl.flatten()), pred.shape, np.bincount(pred.flatten()))

        acc = accuracy_score(lbl, pred)
        conf_m = confusion_matrix(lbl, pred)
        f1_s_w = f1_score(lbl, pred, average='weighted')
        f1_s_micro = f1_score(lbl, pred, average='micro')
        f1_s_macro = f1_score(lbl, pred, average='macro')
        kappa = cohen_kappa_score(lbl, pred)
        jaccard = jaccard_score(lbl, pred)
        tau, p = stats.kendalltau(lbl, pred)

        _sum = 0.0
        for k in range(len(conf_m)):
            _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)
        nacc = _sum / float(test_loader.dataset.num_classes)

        print("---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(nacc) +
              " F1 score weighted= " + "{:.4f}".format(f1_s_w) +
              " F1 score micro= " + "{:.4f}".format(f1_s_micro) +
              " F1 score macro= " + "{:.4f}".format(f1_s_macro) +
              " Kappa= " + "{:.4f}".format(kappa) +
              " Jaccard= " + "{:.4f}".format(jaccard) +
              " Tau= " + "{:.4f}".format(tau) +
              " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
              )

        sys.stdout.flush()

    return acc, nacc, f1_s_w, kappa, conf_m


def train(train_loader, model, criterion, optimizer, epoch):
    # Setting network for training mode.
    model.train()

    # Average Meter for batch loss.
    train_loss = list()

    # Iterating over batches.
    for i, data in enumerate(train_loader):
        # Obtaining images, labels and paths for batch.
        inps, labels = data[0], data[1]

        # if the current batch does not have samples from all classes
        # print('out i', i, len(np.unique(labels.flatten())))
        # if len(np.unique(labels.flatten())) < 10:
        #     print('in i', i, len(np.unique(labels.flatten())))
        #     continue

        # Casting tensors to cuda.
        inps = Variable(inps).cuda()
        labs = Variable(labels).cuda()

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs = model(inps)

        # Computing loss.
        loss = criterion(outs, labs)

        if math.isnan(loss):
            print('-------------------------NaN-----------------------------------------------')
            print(inps.shape, labels.shape, outs.shape, np.bincount(labels.flatten()))
            print(np.min(inps.cpu().data.numpy()), np.max(inps.cpu().data.numpy()),
                  np.isnan(inps.cpu().data.numpy()).any())
            print(np.min(labels.cpu().data.numpy()), np.max(labels.cpu().data.numpy()),
                  np.isnan(labels.cpu().data.numpy()).any())
            print(np.min(outs.cpu().data.numpy()), np.max(outs.cpu().data.numpy()),
                  np.isnan(outs.cpu().data.numpy()).any())
            print('-------------------------NaN-----------------------------------------------')
            raise AssertionError

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            soft_outs = F.softmax(outs, dim=1)
            # Obtaining predictions.
            prds = soft_outs.cpu().data.numpy().argmax(axis=1).flatten()

            labels = labels.cpu().data.numpy().flatten()

            # filtering out pixels
            coord = np.where(labels != train_loader.dataset.num_classes)
            labels = labels[coord]
            prds = prds[coord]

            acc = accuracy_score(labels, prds)
            conf_m = confusion_matrix(labels, prds, labels=[0, 1])
            f1_s = f1_score(labels, prds, average='weighted')

            _sum = 0.0
            for k in range(len(conf_m)):
                _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i + 1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  " Normalized Accuracy= " + "{:.4f}".format(_sum / float(train_loader.dataset.num_classes)) +
                  " F1 Score= " + "{:.4f}".format(f1_s) +
                  " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
                  )
            sys.stdout.flush()

    return sum(train_loss) / len(train_loss), _sum / float(train_loader.dataset.num_classes)


def main():
    parser = argparse.ArgumentParser(description='main')
    # general options
    parser.add_argument('--operation', type=str, required=True, help='Operation [Options: Train | Test]')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to to save outcomes (such as images and trained models) of the algorithm.')

    # dataset options
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--training_images', type=str, nargs="+", required=True, help='Training image names.')
    parser.add_argument('--testing_images', type=str, nargs="+", required=True, help='Testing image names.')
    parser.add_argument('--crop_size', type=int, required=True, help='Crop size.')
    parser.add_argument('--stride_crop', type=int, required=True, help='Stride size')

    # model options
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['deeplab', 'fcnwideresnet'], help='Model to evaluate')
    parser.add_argument('--model_path', type=str, default=None, help='Model path.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=500, help='Number of epochs')

    # handling imbalanced data
    parser.add_argument('--loss_weight', type=float, nargs='+', default=[1.0, 1.0], help='Weight Loss.')
    parser.add_argument('--weight_sampler', type=str2bool, default=False, help='Use weight sampler for loader?')
    args = parser.parse_args()
    print(args)

    # Making sure output directory is created.
    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # writer for the tensorboard
    writer = SummaryWriter(os.path.join(args.output_path, 'logs'))

    if args.operation == 'Train':
        print('---- training data ----')
        train_set = DataLoader('Train', args.dataset_path, args.training_images, args.crop_size, args.stride_crop,
                               args.output_path)
        print('---- testing data ----')
        test_set = DataLoader('Test', args.dataset_path, args.testing_images, args.crop_size, args.stride_crop,
                              args.output_path)

        if args.weight_sampler is False:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
        else:
            class_loader_weights = 1. / np.bincount(train_set.gen_classes)
            samples_weights = class_loader_weights[train_set.gen_classes]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights),
                                                                     replacement=True)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                       num_workers=NUM_WORKERS, drop_last=False, sampler=sampler)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # Setting network architecture.
        model = model_factory(args.model_name, train_set.num_channels, train_set.num_classes).cuda()

        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(args.loss_weight),
                                        ignore_index=train_set.num_classes).cuda()

        # Setting optimizer.
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                               betas=(0.9, 0.99))
        # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        curr_epoch = 1
        best_records = []
        if args.model_path is not None:
            print('Loading model ' + args.model_path)
            best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
            model.load_state_dict(torch.load(args.model_path))
            # optimizer.load_state_dict(torch.load(args.model_path.replace("model", "opt")))
            curr_epoch += int(os.path.basename(args.model_path)[:-4].split('_')[-1])
            for i in range(curr_epoch):
                scheduler.step()
        model.cuda()

        # Iterating over epochs.
        print('---- training ----')
        for epoch in range(curr_epoch, args.epoch_num + 1):
            # Training function.
            t_loss, t_nacc = train(train_loader, model, criterion, optimizer, epoch)
            writer.add_scalar('Train/loss', t_loss, epoch)
            writer.add_scalar('Train/acc', t_nacc, epoch)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                acc, nacc, f1_s, kappa, track_cm = test_full_map(test_loader, model, epoch, args.output_path)
                writer.add_scalar('Test/acc', nacc, epoch)
                save_best_models(model, args.output_path, best_records, epoch, kappa)
                # patch_acc_loss=None, patch_occur=None, patch_chosen_values=None
            scheduler.step()
    elif args.operation == 'Test':
        print('---- testing data ----')
        test_set = DataLoader('Test', args.dataset_path, args.training_images, args.crop_size, args.stride_crop,
                              args.output_path)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # Setting network architecture.
        model = model_factory(args.model_name, test_set.num_channels, test_set.num_classes).cuda()

        best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
        index = 0
        for i in range(len(best_records)):
            if best_records[index]['kappa'] < best_records[i]['kappa']:
                index = i
        epoch = int(best_records[index]['epoch'])
        print("loading model_" + str(epoch) + '.pth')
        model.load_state_dict(torch.load(os.path.join(args.output_path, 'model_' + str(epoch) + '.pth')))
        model.cuda()

        test_full_map(test_loader, model, epoch, args.output_path)
    else:
        raise NotImplementedError("Process " + args.operation + "not found!")


if __name__ == "__main__":
    main()
