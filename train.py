import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
import time
import datetime
from tqdm import tqdm
from data_loader import MPIIFaceGaze2D, MPIIFaceGaze3D
from models_3d import AFFNet as affnet_3d
from models_resnet import GazeResNet as gaze_resnet
from utils_function import exp_smooth

parser = argparse.ArgumentParser(description='GazeEstimation')
parser.add_argument('--train_root', default='/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train', help='Training dataset directory')
parser.add_argument('--valid_root', default='/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/test', help='Training labels')
parser.add_argument('--network', default=None, help='Backbone network')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--max_epoch', default=50, type=int, help='Train for how many epochs')
parser.add_argument('--base_lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='Size of one batch')    # ResNet-18: bs=64; AFFNet: bs=256
parser.add_argument('--resume_weights', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')

opt = parser.parse_args()


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    print("Printing Network...")

    net = affnet_3d(res=True)

    net.train()
    net = net.to(device)

    print(net)

    start_epoch = 1

    if args.resume_weights is not None:
        print('Loading resume network...')
        state_dict = torch.load(args.resume_weights)
        net.load_state_dict(state_dict)
        start_epoch = args.resume_epoch
        assert start_epoch <= args.max_epoch, "'Resume Epoch' cannot exceed 'Max Epoch'!"
        print(f'Start from epoch {start_epoch}.')

    train_dataset = MPIIFaceGaze3D(args.train_root)
    print("Train Data Length:", len(train_dataset))
    train_dataset = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,)

    test_dataset = MPIIFaceGaze3D(args.valid_root)
    print("Test Data Length:", len(test_dataset))
    test_dataset = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,)

    save_folder = "weights/" + datetime.datetime.now().strftime('%Y-%m-%d')

    best_loss, best_epoch = 1000., 0

    optimizer = torch.optim.Adam(net.parameters(), args.base_lr,
                                weight_decay=0.0005)

    acc_time = 0
    time_before, time_now = 0, 0
    for epoch in range(start_epoch, args.max_epoch + 1):
        # Train
        train_remain = train_one_epoch(net, optimizer, train_dataset, device, epoch, args.max_epoch)
        # Evaluate
        best_loss, best_epoch, test_remain = evaluate(net, test_dataset, device, epoch, args.max_epoch, best_loss, best_epoch)
        # Time calculate
        time_now = train_remain + test_remain
        time_after = exp_smooth(time_before, time_now)
        time_before = time_now
        acc_time = acc_time + train_remain + test_remain
        remain = "TOTAL REMAIN TIME ESTIMATED - {:02d}:{:02d}, ALREADY USED - {:02d}:{:02d}\n".format(
            int(time_after * (args.max_epoch - epoch) // 60),
            int(round(time_after * (args.max_epoch - epoch) % 60)),
            int(acc_time // 60),
            int(round(acc_time % 60))
        )
        print(remain)
        # Save weights
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        torch.save(net.state_dict(), save_folder + "/epoch_" + str(epoch) + ".pth")


def train_one_epoch(net, optimizer, dataloader, device, epoch, max_epoch):
    net.train()
    criterion = torch.nn.SmoothL1Loss(reduction='sum')
    optimizer.zero_grad()
    acc_loss = torch.zeros(1).to(device)

    dataloader = tqdm(dataloader)
    length = len(dataloader)
    time_begin = time.time()
    for step, data in enumerate(dataloader):
        gt, leftEye_img, rightEye_img, face_img, vector = data
        gt, leftEye_img, rightEye_img, face_img, vector = gt.to(device), leftEye_img.to(device), rightEye_img.to(device), face_img.to(device), vector.to(device)
        # print("size  ", gt.size(), leftEye_img.size(), rightEye_img.size(), face_img.size(), rects.size())
        gaze = net(leftEye_img, rightEye_img, face_img, vector)
        # loss = criterion(gaze, gt)
        loss = angular_error(gt, gaze)
        loss.backward()
        acc_loss += loss.detach()
        # time estimation for current epoch
        epoch_RemainTime_min = int(((length - step - 1) * ((time.time() - time_begin) / (step + 1))) // 60)
        epoch_RemainTime_sec = int(round(((length - step - 1) * ((time.time() - time_begin) / (step + 1))) % 60))

        dataloader.desc = "[TRAIN EPOCH {}] || LOSS: {:.4f} || CURRENT EPOCH REMAIN TIME - {:02d}:{:02d} -".format(
            epoch,
            acc_loss.item() / (step + 1),
            epoch_RemainTime_min,
            epoch_RemainTime_sec
        )

        log_file = "log/train/" + datetime.datetime.now().strftime('%Y-%m-%d') + ".txt"
        log_line = "TRAIN EPOCH {}/{}, ITER {}/{}, LOSS {:.3f}\n".format(epoch, max_epoch,
                                                                         step + 1, length,
                                                                         acc_loss.item() / (step + 1))
        with open(log_file, 'a') as f:
            f.write(log_line)
        optimizer.step()
        optimizer.zero_grad()
    return time.time() - time_begin


@torch.no_grad()
def evaluate(net, dataloader, device, epoch, max_epoch, best_loss, best_epoch):
    net.eval()
    # criterion = torch.nn.SmoothL1Loss(reduction='sum')
    dataloader = tqdm(dataloader)
    acc_loss = torch.zeros(1).to(device)
    length = len(dataloader)
    time_begin = time.time()
    for step, data in enumerate(dataloader):
        gt, leftEye_img, rightEye_img, face_img, rects = data
        gt, leftEye_img, rightEye_img, face_img, rects = gt.to(device), leftEye_img.to(device), rightEye_img.to(device), face_img.to(device), rects.to(device)
        gaze = net(leftEye_img, rightEye_img, face_img, rects)
        # loss = criterion(gaze, gt)
        loss = angular_error(gaze, gt)
        acc_loss += loss
        # time estimation for current epoch
        epoch_RemainTime_min = str(int(((length - step - 1) * ((time.time() - time_begin) / (step + 1))) // 60)).rjust(2, '0')
        epoch_RemainTime_sec = str(int(round(((length - step - 1) * ((time.time() - time_begin) / (step + 1))) % 60))).rjust(2, '0')
        dataloader.desc = "[EVALUATE EPOCH {}] || BEST LOSS {:.3f} || BEST EPOCH {} || " \
                          "CURRENT EPOCH REMAIN TIME - {}:{} -".format(
            epoch,
            best_loss, best_epoch,
            epoch_RemainTime_min,
            epoch_RemainTime_sec
        )

    if (acc_loss / length) < best_loss:
        best_loss = acc_loss / length
        best_epoch = epoch

    if not isinstance(best_loss, float):
        best_loss = float(best_loss.item())

    log_file = "log/evaluate/" + datetime.datetime.now().strftime('%Y-%m-%d') + ".txt"
    log_line = "EVALUATE Epoch {}/{}, LOSS {:.3f}, BEST LOSS {:.3f}, BEST EPOCH {}\n".format(epoch, max_epoch,
                                                                                         (acc_loss.item() / length),
                                                                                         best_loss, best_epoch)
    with open(log_file, 'a') as f:
        f.write(log_line)

    return best_loss, best_epoch, time.time() - time_begin


def angular_error(gt, predict, reduction='sum'):
    assert gt.size()[0] == predict.size()[0]
    losses = 0
    for g, p in zip(gt, predict):
        cosine_sim = torch.matmul(g, p) / (torch.norm(g) * torch.norm(p))
        loss = 1 - cosine_sim
        losses += loss
    if reduction == 'sum':
        return losses
    if reduction == 'mean':
        return losses / gt.size()[0]


if __name__ == "__main__":
    train(opt)
