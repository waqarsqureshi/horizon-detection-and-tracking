#few changes are made in the code to work with images sizes that we have.
from __future__ import division
import numpy as np
import cv2
import os
import os.path as osp
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
import track
import detect

def readImage(img_path):
    ori_img = cv2.imread(img_path)
    if ori_img is None:
        return None
    return ori_img

def save_show(img_path,image,args):
    out_file = args.savedir 

    if out_file:
        out_file = osp.join(out_file, osp.basename(img_path))
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, image)
        #exit() #test point waqar

    if args.show:
        cv2.imshow('view', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit()
           


def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def process(args):
    #args.config
    args.show
    args.savedir
    ticks = 0
    lt = track.LaneTracker(2, 0.1, 500)
    ld = detect.LaneDetector(180)

    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        precTick = ticks
        ticks = cv2.getTickCount()
        dt = (ticks - precTick) / cv2.getTickFrequency()
        print("processing Image path: ",p)
        ori_img = readImage(p)
        if ori_img is None:
            print("Error: No Image file")
            break
        frame = cv2.resize(ori_img[245:590+245,140:1640+40],(1640,590), interpolation = cv2.INTER_AREA) #waqar
        print(frame.shape[0], frame.shape[1])
        
        predicted = lt.predict(dt)
        
        lanes = ld.detect(frame)
        print(lanes)
    
        if predicted is not None:
            cv2.line(frame,
                     (int(predicted[0][0]), int(predicted[0][1])),
                     (int(predicted[0][2]), int(predicted[0][3])),
                     (0, 0, 255), 5)
            cv2.line(frame,
                     (int(predicted[1][0]), int(predicted[1][1])),
                     (int(predicted[1][2]), int(predicted[1][3])),
                     (0, 0, 255), 5)
        if lanes is not None:
            lt.update(lanes)
        
        save_show(p,frame,args)

    cv2.destroyAllWindows()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    #parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)