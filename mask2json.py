# -*- encoding: utf-8 -*-
'''
@File    :   mask2json.py
@Time    :   2021/01/09 11:09:02
@Author  :   Wu Shaogui
@Version :   1.0
@Contact :   wshglearn@163.com
@Desc    :   将mask转为labelme的json文件
'''
import json
from os import mkdir
import os.path as ops
from base64 import b64encode

import cv2
import numpy as np


class Mask2Json(object):
    def __init__(self,json_template_path,is_reduce=False):
        '''__init__ [summary]

        Args:
            json_template_path ([type]): [labelme json 的模板]
            is_reduce (bool, optional): [是否减少点集合的点数量]. Defaults to True.
        '''
        self.is_reduce=is_reduce
        #加载json模板
        with open(json_template_path,'r',encoding='utf-8') as fr:
            self.labelme_template=json.load(fr)
    
    def remove_points(self,contour,mindis=2):
        '''remove_points [opencv找到的边点集非常密集，需要通过该函数移除点集中距离小于mindis中的一个点]

        Args:
            contour ([3D ndarray]): [点集]
            mindis (int, optional): [小于该距离的点将被过滤]]. Defaults to 2.

        Returns:
            [2D ndarray]: [已移除靠近点的点集合]
        '''
        new_contour=contour.reshape((contour.shape[0],contour.shape[-1]))
        if (len(new_contour))<6:#点集数量太少，直接返回
            return new_contour
        
        ind=0
        while ind<len(new_contour):
            cur_dis=np.sqrt(np.sum((new_contour[ind+1:]-new_contour[ind])**2,axis=1))
            delnum=0
            for ind_p,_ in enumerate(cur_dis):
                if cur_dis[ind_p]<=mindis:
                    new_contour=np.delete(new_contour,ind+ind_p+1-delnum,0)
                    delnum+=1
            ind+=1
        return new_contour
    
    def mask_to_json(self,image_path,mask,min_points=0,mindis=5,save_json_dir=None):
        '''mask_to_json [生成json文件]

        Args:
            image_path ([str]): 原图文件路径,labelme的json文件包含b64encode编码的原文件，所以需要编码后存储该信息到json
            mask ([ndarray]): 对应的mask标注
            min_points(int): 点集数量小于该值得mask将被过滤
            save_json_path ([str], optional): 生成的json文件. Defaults to None.

        Returns:
            [bool]: [是否生成成功]
        '''

        # 获得每个mask的边缘点集
        mask=mask.astype(np.uint8)
        mask=mask[...,-1] if len(mask.shape)==3 else mask
        contours,_  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #原始mask

        # 以下是给模板赋值
        # self.labelme_template['version']='4.5.6'
        # self.labelme_template['flags']={}

        self.labelme_template['shapes']=[]
        for contour in contours:
            new_contour=contour.reshape((contour.shape[0],contour.shape[-1])) #去掉矩阵中间维度1
            if self.is_reduce:
                # new_contour=self.remove_points(contour,mindis=mindis)#使用自定义方法,移除点集中靠近的点
                
                new_contour=cv2.approxPolyDP(contour,2,True) #使用opencv求得近似多边形的点集合
                new_contour=np.reshape(new_contour,(new_contour.shape[0],new_contour.shape[2]))
            
            # 过滤后的边点集数量在min_points以上
            if len(new_contour)>=min_points:
                shape={}
                shape['label']='all'
                shape['points']=[]
                for point in new_contour:
                    assert len(point.shape)==1
                    shape['points'].append(point.tolist())
                shape['group_id']=None

                # 不同的形状，不同的解析方法，目前只能区分point，linestrip，polygon三种类型
                if len(shape['points'])==1:
                    shape['shape_type']='point'
                elif len(shape['points'])==2:
                    shape['shape_type']='linestrip'
                else:
                    shape['shape_type']='polygon'
                shape['flags']={}

                self.labelme_template['shapes'].append(shape)

        self.labelme_template['imagePath']=image_path.replace('/', '\\')
        self.labelme_template['imageData']=b64encode(open(image_path, "rb").read()).decode('utf-8')
        self.labelme_template['imageHeight']=mask.shape[0]
        self.labelme_template['imageWidth']=mask.shape[1]

        # 保存json
        if save_json_dir==None:
            save_json_path=ops.join('./','.'.join(ops.basename(image_path).split('.')[:-1])+'.json')
        else:
            save_json_path=ops.join(save_json_dir,'.'.join(ops.basename(image_path).split('.')[:-1])+'.json')
        json_content=json.dumps(self.labelme_template, ensure_ascii=False, indent=2, separators=(',',': '))
        try:
            with open(save_json_path, 'w+',encoding = 'utf-8') as fw:
                fw.write(json_content)
            return True
        except:
            return False

if __name__=='__main__':
    import glob
    import os

    # images_root='/home/wushaogui/DataRepo/IMAGE_SCENE_SEGMENTATION/JELLYROLL/v3.19/splitfig/'
    # clanames=['line','point','salient','sully','blue_point','nick','pit','scallops','shadow']

    mask2json=Mask2Json('./labelme4.5.6_template.json',is_reduce=True)
    # for claname in clanames:
    #     images_path=glob.glob(ops.join(images_root,'images',claname,'*.jpg'))
    #     save_json_dir=ops.join('./','jsons',claname)
    #     if not ops.exists(save_json_dir):
    #         os.makedirs(save_json_dir)
    #     for image_path in images_path:
    #         print(image_path)
    #         file_name=ops.basename(image_path)

    #         image=cv2.imread(image_path)
    #         mask=cv2.imread(image_path.replace('images','masks').replace('.jpg','.png'))
    #         if mask2json.mask_to_json(image_path,mask,save_json_dir=save_json_dir,mindis=6):
    #             print('gen done!')
    #         else:
    #             print('gen fail')

    # image_path='2020.11.21 19.37.07.121-201022111206102.bmp'
    # mask=cv2.imread('2020.11.21 19.37.07.121-201022111206102.mask.0.png')
    # if mask2json.mask_to_json(image_path,mask,save_json_dir='./'):
    #     print('gen done!')
    # else:
    #     print('gen faise')

    images_root='/EHOLLY_DATA/Labels/wushaogui/划痕凹坑附加数据集/roi'
    save_json_dir='/EHOLLY_DATA/Labels/飞拍/jsons/划痕凹坑'
    images_path=glob.glob(ops.join(images_root,'*[jpg,jpeg,png,bmp]'))
    for ind,image_path in enumerate(images_path):
        image_name=ops.basename(image_path)
        mask=cv2.imread(ops.join('/EHOLLY_DATA/Labels/飞拍/masks/划痕凹坑','.'.join(image_name.split('.')[:-1])+'.png'))
        if mask2json.mask_to_json(image_path,mask,save_json_dir=save_json_dir,mindis=6):
            print(ind,'gen done!')
        else:
            print(ind,'gen fail')

