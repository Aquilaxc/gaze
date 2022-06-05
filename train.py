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
from utils_function import exp_smooth, angular_error, angular_error_np


parser = argparse.ArgumentParser(description='GazeEstimation')
parser.add_argument('--train_root', default=r'D:\code\gaze\data\train', help='Training dataset directory')
parser.add_argument('--valid_root', default=r'D:\code\gaze\data\test', help='Training labels')
parser.add_argument('--network', default=None, help='Backbone network')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--max_epoch', default=50, type=int, help='Train for how many epochs')
parser.add_argument('--base_lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='Size of one batch')    # ResNet-18: bs=64; AFFNet: bs=256
parser.add_argument('--resume_weights', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')

opt = parser.parse_args()


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    print("Printing Network...")

    net = affnet_3d(res=False)

    net.train()
    net = net.to(device)

    print(net)

    start_epoch = 1
    lr = args.base_lr

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

    optimizer = torch.optim.Adam(net.parameters(), lr,
                                weight_decay=0.0005)

    acc_time = 0
    time_before, time_now = 0, 0
    for epoch in range(start_epoch, args.max_epoch + 1):
        if epoch == 15 or epoch == 30:
            lr /= 10
            optimizer = torch.optim.Adam(net.parameters(), lr,
                                weight_decay=0.0005)
        if epoch == 20:     # Activate augment
            train_dataset = MPIIFaceGaze3D(args.train_root, augment=True)
            train_dataset = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers, )

        # Train
        train_remain = train_one_epoch(net, optimizer, train_dataset, device, epoch, args.max_epoch)
        # Evaluate
        best_loss, best_epoch, test_remain = evaluate(net, test_dataset, device, epoch, args.max_epoch, best_loss, best_epoch)
        # Time calculate
        time_now = train_remain + test_remain
        time_after = exp_smooth(time_before, time_now)
        time_before = time_now
        acc_time = acc_time + train_remain + test_remain
        remain = "TOTAL REMAIN TIME ESTIMATED - {:02d}:{:02d}, ALREADY USED - {:02d}:{:02d}, LR - {}\n".format(
            int(time_after * (args.max_epoch - epoch) // 60),
            int(round(time_after * (args.max_epoch - epoch) % 60)),
            int(acc_time // 60),
            int(round(acc_time % 60)),
            lr
        )
        print(remain)
        # Save weights
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        torch.save(net.state_dict(), save_folder + "/epoch_" + str(epoch) + ".pth")


def train_one_epoch(net, optimizer, dataloader, device, epoch, max_epoch):
    net.train()
    criterion = torch.nn.SmoothL1Loss(reduction='mean')
    optimizer.zero_grad()
    acc_loss = torch.zeros(1).to(device)

    dataloader = tqdm(dataloader)
    length = len(dataloader)
    time_begin = time.time()
    for step, data in enumerate(dataloader):
        gt, leftEye_img, rightEye_img, face_img, vector, _ = data
        gt, leftEye_img, rightEye_img, face_img, vector = gt.to(device), leftEye_img.to(device), rightEye_img.to(device), face_img.to(device), vector.to(device)
        # print("size  ", gt.size(), leftEye_img.size(), rightEye_img.size(), face_img.size(), rects.size())
        gaze = net(leftEye_img, rightEye_img, face_img, vector)
        # loss = criterion(gt, gaze)
        # loss = angular_error_np(gt.cpu().data.numpy(), gaze.cpu().data.numpy(), reduction='sum')
        loss = angular_error(gt, gaze, reduction='sum')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # acc_loss += loss.detach()
        # time estimation for current epoch
        epoch_RemainTime_min = int(((length - step - 1) * ((time.time() - time_begin) / (step + 1))) // 60)
        epoch_RemainTime_sec = int(round(((length - step - 1) * ((time.time() - time_begin) / (step + 1))) % 60))

        dataloader.desc = "[TRAIN EPOCH {}/{}] || LOSS: {:.4f} || CURRENT EPOCH REMAIN TIME - {:02d}:{:02d} -".format(
            epoch,
            opt.max_epoch,
            loss.item(), # acc_loss.item() / (step + 1),
            epoch_RemainTime_min,
            epoch_RemainTime_sec
        )

        log_file = "log/train/" + datetime.datetime.now().strftime('%Y-%m-%d') + ".txt"
        log_line = "TRAIN EPOCH {}/{}, ITER {}/{}, LOSS {:.3f}\n".format(epoch, max_epoch,
                                                                         step + 1, length,
                                                                         loss) # acc_loss.item() / (step + 1))
        with open(log_file, 'a') as f:
            f.write(log_line)
    return time.time() - time_begin


@torch.no_grad()
def evaluate(net, dataloader, device, epoch, max_epoch, best_loss, best_epoch):
    net.eval()
    criterion = torch.nn.SmoothL1Loss(reduction='mean')
    dataloader = tqdm(dataloader)
    acc_loss = torch.zeros(1).to(device)
    length = len(dataloader)
    time_begin = time.time()
    for step, data in enumerate(dataloader):
        gt, leftEye_img, rightEye_img, face_img, vector, pixel_gt = data
        gt, leftEye_img, rightEye_img, face_img, vector = gt.to(device), leftEye_img.to(device), rightEye_img.to(device), face_img.to(device), vector.to(device)
        gaze = net(leftEye_img, rightEye_img, face_img, vector)
        # loss = criterion(gt, gaze)
        loss = angular_error(gt, gaze)

        acc_loss += loss.detach()
        # time estimation for current epoch
        epoch_RemainTime_min = str(int(((length - step - 1) * ((time.time() - time_begin) / (step + 1))) // 60)).rjust(2, '0')
        epoch_RemainTime_sec = str(int(round(((length - step - 1) * ((time.time() - time_begin) / (step + 1))) % 60))).rjust(2, '0')
        dataloader.desc = "[EVALUATE EPOCH {}/{}] || BEST LOSS {:.3f} || BEST EPOCH {} || " \
                          "CURRENT EPOCH REMAIN TIME - {}:{} -".format(
            epoch,
            opt.max_epoch,
            best_loss, best_epoch,
            epoch_RemainTime_min,
            epoch_RemainTime_sec
        )

    if (acc_loss.item() / length) < best_loss:
        best_loss = acc_loss.item() / length
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


if __name__ == "__main__":
    train(opt)
