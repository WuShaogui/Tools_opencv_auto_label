# -- coding: utf-8 --
'''
@File  :label_json_to_datasets
@Author:Shaogui
@Date  :2020/10/17 10:39
@Desc  :将labelme笔记的图像分割数据"批量"转为mask格式--库导入版本
'''
import argparse
import glob
import json
import os
import os.path as ops

import cv2
import numpy as np
import yaml
from labelme import utils

from draw_label import draw_label


class LabelmeAnnotation:
    def __init__(self, json_dir, out_dir):
        self.json_dir = json_dir
        self.out_dir = out_dir
        print('deal json path:',self.json_dir)
        print('output file path:',self.out_dir)

    def export_mask_img(self, save_dir, lbl, lbl_names):
        '''export_mask_img 导出json中的mask

        Args:
            save_dir (str): 文件保存路径
            lbl ([1D ndarray]): 与图像大小对应的矩阵，每个像素对应存储类别信息
            lbl_names ([list(str)]): 类别列表
        '''
        mask = []
        for i in range(1, len(lbl_names)):  # 跳过第一个class（默认为背景）
            # 解析出每种标记对应的图，如：第一种对象标记为1，那么就单独取出第一种对象组成一张mask
            # (lbl == i)会把对应位置变成True，用astype转为int8
            # 此时对应位置置1，其余为0
            mask.append((lbl == i).astype(np.uint8))
        
        #如果不存在前景，直接保存空白背景
        if len(mask)==0:
            cv2.imwrite(ops.join(save_dir, 'label0.png'),np.zeros(lbl.shape, dtype=np.uint8))
            cv2.imwrite(ops.join('/home/wushaogui/DataRepo/IMAGE_SCENE_SEGMENTATION/JELLYROLL/v3.19/splitfig/masks',
            '/'.join(save_dir.split('/')[2:]).replace('.json','.png')),np.zeros(lbl.shape, dtype=np.uint8))
        else:
            mask = np.transpose(np.asarray(mask, np.uint8), [1, 2, 0])  # 转成[h,w,instance count]
            for i in range(len(lbl_names) - 1):
                _, mask[:, :, i] = cv2.threshold(mask[:, :, i], 0.5, 255, type=cv2.THRESH_BINARY)
                cv2.imwrite(ops.join(save_dir,'label%d.png'%i),mask[:, :, i])
                cv2.imwrite(ops.join('/home/wushaogui/DataRepo/IMAGE_SCENE_SEGMENTATION/JELLYROLL/v3.19/splitfig/masks',
            '/'.join(save_dir.split('/')[2:]).replace('.json','.png')),mask[:, :, i])

    def export_viz_img(self,save_dir, img, lbl, lbl_names):
        '''export_viz_img 导出json中的图片

        Args:
            save_dir (str): 文件保存路径
            img (3D ndarray): 图片
            lbl (1D array): 与图像大小对应的矩阵，每个像素对应存储类别信息
            lbl_names (list(str)): 类别列表
        '''
        # 取出标签的key和value
        captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]

        # 画出原图上面覆盖有lbl的多层的图（用处不大）
        lbl_viz = draw_label(lbl, img, captions)
        cv2.imwrite(ops.join(save_dir, 'label_viz.png'), lbl_viz)


    def export_label_txt(self,save_dir, lbl_names):
        '''export_label_txt 导出json中的类别信息

        Args:
            save_dir (str): 文件保存路径
            lbl_names (list(str)): 类别列表
        '''
        with open(ops.join(save_dir, 'label_names.txt'), 'w') as f:
            for lbl_name in lbl_names:
                f.write(lbl_name + '\n')


    def export_label_yaml(self,save_dir, lbl_names):
        '''export_label_yaml 导出json中的yaml

        Args:
            save_dir (str): 文件保存路径
            lbl_names (list(str)): 类别列表
        '''
        info = dict(label_names=lbl_names)
        with open(ops.join(save_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)

    def label_json_to_datasets(self):
        # 批量处理json文件
        for json_path in glob.glob(ops.join(self.json_dir,'*/*.json')):
            save_dir=json_path.replace(self.json_dir,self.out_dir)
            print(save_dir)
            if ops.isfile(json_path):
                # json数据
                data = json.load(open(json_path))

                # 把json文件中的二进制图片格式转化为array数组形式
                img = utils.img_b64_to_arr(data['imageData'])

                # 解析标签
                lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

                # 创建同结构文件夹
                if not ops.exists(save_dir):
                    os.makedirs(save_dir)

                # 存储原图，mask,标签
                # cv2.imwrite(ops.join(save_dir, 'img.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                self.export_mask_img(save_dir, lbl, lbl_names)
                # self.export_viz_img(save_dir, img, lbl, lbl_names)
                # self.export_label_txt(save_dir, lbl_names)
                # self.export_label_yaml(save_dir, lbl_names)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_dir')
    parser.add_argument('out_dir')
    args = parser.parse_args()

    labelmeannotation=LabelmeAnnotation(args.json_dir,args.out_dir)
    labelmeannotation.label_json_to_datasets()
