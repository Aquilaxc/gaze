import os
import sys
import numpy as np
import data_processing_core as dpc
import gazeconversion as co
import cv2
from scipy import io as sio
import argparse


def read_file(path):
    with open(path) as infile:
        lines = infile.readlines()
        lines.pop(0)
        acc = lines.pop(-1)
    for line in lines:
        line = line.strip().split(" ")
        
        gt = line[2]
        name = line[0]
        prediction = line[1]
        
        gt = np.array(list(map(eval, gt.split(","))))
        prediction = np.array(list(map(eval, prediction.split(","))))

        yield name, gt, prediction


def convert_mpii_3T2(screen_pose, screen_size, logfile, annofolder, is3D):
    # Read Screen-Camera pose
    screen_pose = sio.loadmat(screen_pose)
    rvec = screen_pose["rvects"]
    tvec = screen_pose["tvecs"]
    rmat = cv2.Rodrigues(rvec)[0]

    # Convet pixel to mm
    screen = sio.loadmat(screen_size)
    w_pixel = screen["width_pixel"][0][0]
    h_pixel = screen["height_pixel"][0][0]
    w_mm = screen["width_mm"][0][0]
    h_mm = screen["height_mm"][0][0]
    w_ratio = w_mm / w_pixel
    h_ratio = h_mm / h_pixel

    # read gaze origin from annotation file
    person = os.path.split(logfile)[0]
    person = os.path.split(person)[1].split(".")[0]
    annotation = os.path.join(annofolder, f"{person}.label")
    with open(annotation) as infile:
        lines = infile.readlines()
        lines.pop(0)
         

    total_pixel = 0
    total_mm = 0
    for count, (name, gt, prediction) in enumerate(read_file(logfile)):
        annos = lines[count].strip().split(" ")
        if not is3D:
            gt = dpc.GazeTo3d(gt)
            prediction = dpc.GazeTo3d(prediction)

        Rvec = np.array(list(map(eval, annos[-3].split(","))))
        Svec = np.array(list(map(eval, annos[-2].split(","))))
        origin = np.array(list(map(eval, annos[-1].split(","))))

        Rmat = cv2.Rodrigues(Rvec)[0]
        Smat = np.diag(Svec)
        mat = np.dot(np.linalg.inv(Rmat), np.linalg.inv(Smat))

        pred_ccs = np.dot(mat, prediction.reshape((3, 1))).flatten()
        gt_ccs = np.dot(mat, gt.reshape((3,1))).flatten()
        origin_ccs = np.dot(mat, origin.reshape((3,1))).flatten()

        pred_point = co.Gaze3DTo2D(pred_ccs, origin_ccs, rmat, tvec)
        ccs_point = co.Gaze3DTo2D(gt_ccs, origin_ccs, rmat, tvec)
        mm_loss = np.sqrt(np.sum((pred_point - ccs_point)**2))
        total_mm += mm_loss

        pred_point[0] = pred_point[0]/w_ratio
        pred_point[1] = pred_point[1]/h_ratio

        ccs_point[0] = ccs_point[0]/w_ratio
        ccs_point[1] = ccs_point[1]/h_ratio

        pixel_loss = np.sqrt(np.sum((pred_point - ccs_point)**2))
        total_pixel += pixel_loss
    return total_pixel, total_mm, count+1


if __name__ == "__main__":

    parser  = argparse.ArgumentParser(description="Convert 3D gaze to 2D gaze on MPIIGaze")  
    
    parser.add_argument('--screenPose', type=str, default="/home/cyh/GazeDataset20200519/Original/MPIIFaceGaze/p00/Calibration/monitorPose.mat")

    parser.add_argument('--screenSize', type=str, default="/home/cyh/GazeDataset20200519/Original/MPIIFaceGaze/p00/Calibration/screenSize.mat")

    parser.add_argument('--logfolder', type=str, default= "/home/cyh/GazeBenchmark/exp/Implementation/Full-Face/evaluation/")

    parser.add_argument('--label', type=str, default= "/home/cyh/GazeDataset20200519/FaceBased/MPIIFaceGaze/Label/")

    parser.add_argument('--name', type=str, default= "20.log")

    args = parser.parse_args()
    acc, count = main(args)
    print(acc)


def main(args):
    total_pixel = 0
    total_mm = 0
    total_count = 0

    for i in ["p00", "p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10", "p11", "p12", "p13", "p14"]:
        logfile = os.path.join(args.logfolder, f"{i}.label/{args.name}")

        pixel_loss, mm_loss, count= convert_mpii_3T2(args.screenPose, 
                              args.screenSize, 
                              logfile, 
                              args.label,
                              args.is3D)

        total_pixel += pixel_loss
        total_mm += mm_loss
        total_count += count
    return total_pixel/total_count, total_mm/total_count, total_count 


            
            
        


