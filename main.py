import os
import numpy as np
import cv2
import copy
import sys
import pywt
import scipy
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

import pdb

# 変化点抽出に適したウェーブレットの例 (他のを用いても良い)
complex_wtype = ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6',
                 'cmor', 'cmor0.5-0.5', 'cmor1.5-0.5', 'cmor2.5-0.5']

# スケール間の平均に注目
def compute_scale_mean(wtlist):
    aggregated_change = np.mean(np.stack(wtlist), axis=0)
    return aggregated_change

# スケール間の相関に注目
def compute_scale_correlation(wtlist):
    shape = wtlist[0].shape
    change_map = np.zeros(shape)

    for level in range(len(wtlist) - 1):
        numerator = (wtlist[level] * wtlist[level + 1])
        denominator = np.sqrt(wtlist[level]**2 + 1e-6) * np.sqrt(wtlist[level + 1]**2 + 1e-6)
        local_correlation = numerator / (denominator + 1e-6)
        change_map += 1 - local_correlation

    change_map /= (len(wtlist) - 1)

    return change_map

# x方向のwavelet変換
def xcwtlist (imagefile, scales=[1,2,4,8,16], wtype='morl'):
    img = cv2.imread(imagefile)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    wtlist = []
    for scale in scales:
        xwt = xcwt(gimg, scale, wtype)
        wtlist.append(xwt)
    return wtlist

def xcwt (gimg, scale, wtype='morl'):
    if wtype in complex_wtype:
        fimg = np.empty_like(gimg, dtype=np.complex64)
    else:
        fimg = np.empty_like(gimg, dtype=np.float64)
    for y in range(gimg.shape[0]):
        fimg[y,:],_ = pywt.cwt(gimg[y,:], [scale,], wtype)
    return fimg

# y方向のwavelet変換
def ycwtlist (imagefile, scales=[1,2,4,8,16], wtype='morl'):
    img = cv2.imread(imagefile)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    wtlist = []
    for scale in scales:
        ywt = ycwt(gimg, scale, wtype)
        wtlist.append(ywt)
    return wtlist

def ycwt (gimg, scale, wtype='morl'):
    if wtype in complex_wtype:
        fimg = np.empty_like(gimg, dtype=np.complex64)
    else:
        fimg = np.empty_like(gimg, dtype=np.float64)
    for x in range(gimg.shape[1]):
        fimg[:,x],_ = pywt.cwt(gimg[:,x], [scale,], wtype)
    return fimg


# スケールごとの結果を統合し、x, y方向の結果を統合
def xycwtlist_unify(imagefile, scales=[1, 2, 4, 8, 16], wtype='morl', how_scale="diff", how_xy="plus"):
    img = cv2.imread(imagefile)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_wtlist = []
    y_wtlist = []
    for scale in scales:
        x_wt = xcwt(gimg, scale, wtype)
        y_wt = ycwt(gimg, scale, wtype)
        x_wtlist.append(x_wt)
        y_wtlist.append(y_wt)

    if how_scale == "mean":
        x_change = compute_scale_mean(x_wtlist)
        y_change = compute_scale_mean(y_wtlist)
    elif how_scale == "correlation":
        x_change = compute_scale_correlation(x_wtlist)
        y_change = compute_scale_correlation(y_wtlist)

    if how_xy == "plus":
        combined_change = x_change + y_change
    elif how_xy == "product":
        combined_change = x_change * y_change
    elif how_xy == "max":
        combined_change = np.maximum(x_change, y_change)

    return [[x_wt, y_wt, combined_change]]

# まとめて，x, y方向の変換結果を得る
def xycwtlist (imagefile, scales=[1,2,4,8,16], wtype='morl'):
    img = cv2.imread(imagefile)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    wtlist = []
    for scale in scales:
        xwt = xcwt(gimg, scale, wtype)
        ywt = ycwt(gimg, scale, wtype)
        xywt = np.abs(xwt) + np.abs(ywt)
        wtlist.append([xwt, ywt, xywt])
    return wtlist

# 二値化して大まかに結果を表示
def show_wt_b (wt):
    wti = np.array(np.abs(wt), dtype='uint8')
    _, bimg_otsu = cv2.threshold(wti, 0, 255, cv2.THRESH_OTSU)
    return bimg_otsu

# ピーク (極大値)を表示
def show_wt_peaks (wt, threshold=None):
    iarr = wt
    oarr = np.zeros(iarr.shape, dtype='uint8')
    for i in range(iarr.shape[0]):
        peaks, _ = scipy.signal.find_peaks(np.abs(iarr[i]), threshold)
        for j in peaks:
            oarr[i,j] = 1
    return oarr

# 各ライン(y軸に平行なライン)の値をグラフ表示
def show_line (imagefile, line):
    img = cv2.imread(imagefile)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    myplot(gimg[line,:])

def show_xlines (wtlist, xy='x', line=300, plot_type='o', levels=None):
    if xy == 'x':
        n = 0
    elif xy == 'y':
        n = 1
    else:
        n = 2
    dline_list = [[normalize(np.abs(wtlist[level][n][line,:]))] \
                  for level in levels]
    ddata = np.transpose(np.concatenate(dline_list))
    myplot(ddata, plot_type=plot_type)

def normalize(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def myplot(data, plot_type='o'):
    if data.ndim == 1:
        plt.plot(data, plot_type)
        plt.show()
    elif data.ndim == 2:
        for idx, col in enumerate(data.T):
            plt.figure()
            plt.plot(col, plot_type)
            plt.title(f"level {idx}")
            plt.show()
    else:
        raise ValueError("Unsupported data dimension for plotting.")

# 表示関数のメイン
def show_wt (wtlist, type='raw', xy='a', levels=None,
             line=300, threshold=None, plot_type='s'):
    if levels is None:
        levels = range(len(wtlist))
    if isinstance(levels, int):
        levels = [levels,]
    if type in ['line', 'l']:
        show_xlines(wtlist, xy, line, plot_type, levels)
        type = 'raw'
        levels = [0,]
    for level in levels:
        wt = wtlist[level]
        if 'x' in xy:
            out = show_wt_level(wt[0], type, threshold=None)
            showimage(out, title=f'WT: {type=}, {xy=}, {level=}')
        if 'y' in xy:
            out = show_wt_level(wt[1], type, threshold=None)
            showimage(out, title=f'WT: {type=}, {xy=}, {level=}')
        if 'a' in xy:
            out = show_wt_level(wt[2], type, threshold=None)
            showimage(out, title=f'WT: {type=}, {xy=}, {level=}')

def show_wt_level(wt, type='raw', plot_type='o', threshold=None):
    if type in ['raw', 'r']:
        return np.abs(wt)
    elif type in ['peak', 'p']:
        return show_wt_peaks(wt, threshold)
    elif type in ['binarize', 'b']:
        return show_wt_b(wt)

def showimage(img, title=None):
    window_name = title if title else "Image"
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_path = "result.jpg"
    counter = 1

    while os.path.exists(save_path):
        save_path = f"result_{counter}.jpg"
        counter += 1
    
    cv2.imwrite(save_path, img)

def main():
    ww=xycwtlist('rabbit.jpg', wtype="morl", scales=[1,2,4,8,16,32])
    show_wt(ww)
    show_wt(ww, 'b')

    how_scale_candidates = ["mean", "correlation"]
    how_xy_candidates = ["plus", "product", "max"]
    for how_scale in how_scale_candidates:
        for how_xy in how_xy_candidates:
            ww = xycwtlist_unify('rabbit.jpg', scales=[1,2,4,8,16,32], wtype='morl', how_scale=how_scale, how_xy=how_xy)
            show_wt(ww)
            show_wt(ww, 'b')

if __name__ == '__main__':
    main()
