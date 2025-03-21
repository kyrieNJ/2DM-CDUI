import numpy as np
import spacy
import pickle
# import Bert_representation
# from transformers import BertModel, BertConfig
import torch
import os
import json
from tqdm import tqdm
nlp = spacy.load('zh_core_web_sm')
import csv
import random


def processcsv(filename):
    dataset = {
        0:[],
        1:[],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
    }
    ana_dict = {}
    with open(filename, 'r', encoding='gbk') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in tqdm(reader):
            tmpdict = {}
            tmpdict['id'] = row[0]
            tmpdict['scene'] = row[1]
            tmpdict['mind'] = row[2]
            if tmpdict['scene']=='':
                continue
            tmpc_label = row[3]
            tmplabel = -1
            if tmpc_label == '非黑即白':
                tmplabel = 0
            elif tmpc_label == '情绪化推理':
                tmplabel = 1
            elif tmpc_label == '算命':
                tmplabel = 2
            elif tmpc_label == '乱贴标签':
                tmplabel = 3
            elif tmpc_label == '读心术':
                tmplabel = 4
            elif tmpc_label == '过度泛化':
                tmplabel = 5
            elif tmpc_label == '个人化归责':
                tmplabel = 6
            elif tmpc_label == '非扭曲':
                tmplabel = 7
            tmpdict['label'] = tmplabel
            dataset[tmplabel].append(tmpdict)

            if tmpdict['scene'] not in ana_dict.keys():
                ana_dict[tmpdict['scene']] = {}
                ana_dict[tmpdict['scene']][tmplabel]=[]
                ana_dict[tmpdict['scene']][tmplabel].append(tmpdict['mind'])
            else:
                if tmplabel not in ana_dict[tmpdict['scene']].keys():
                    ana_dict[tmpdict['scene']][tmplabel] = []
                    ana_dict[tmpdict['scene']][tmplabel].append(tmpdict['mind'])
                else:
                    ana_dict[tmpdict['scene']][tmplabel].append(tmpdict['mind'])

    return dataset,ana_dict

def tripleset_construct(alist,blist,clist,dlist,elist,flist,glist,hlist,CL_list):
    #取所有列表中的最大长度为每个正例中构造元组的循环总数
    maxlen=min(len(alist),len(blist),len(clist),len(dlist),len(elist),len(flist),len(glist),len(hlist))
    #每类依次循环当正例，其余为反例，构造时按0到maxlen的下标进行依次顺序遍历数组来构造元组，如果某一列表的长度不够则一直采用最后一个元素
    #每个元组有2个正例7个反例（即一个学习的表示，一个正例7个反例），0号位是正例1，1-3号位是顺序反例，4号位是正例2，5-8号位是其余顺序反例
    # abcdefg
    if len(alist)>=2:
        for aid in range(0,len(alist),2):
            for itag in range(0,maxlen):
                if itag>=len(blist):
                    b_mind=blist[len(blist)-1]
                else:
                    b_mind=blist[itag]

                if itag>=len(clist):
                    c_mind=clist[len(clist)-1]
                else:
                    c_mind=clist[itag]

                if itag>=len(dlist):
                    d_mind=dlist[len(dlist)-1]
                else:
                    d_mind=dlist[itag]

                if itag>=len(elist):
                    e_mind=elist[len(elist)-1]
                else:
                    e_mind=elist[itag]

                if itag>=len(flist):
                    f_mind=flist[len(flist)-1]
                else:
                    f_mind=flist[itag]

                if itag>=len(glist):
                    g_mind=glist[len(glist)-1]
                else:
                    g_mind=glist[itag]

                if itag>=len(hlist):
                    h_mind=hlist[len(hlist)-1]
                else:
                    h_mind=hlist[itag]

                if aid+1 >=len(alist):
                    a_mind1=alist[aid]
                    a_mind2=alist[len(alist)-2]
                else:
                    a_mind1=alist[aid]
                    a_mind2=alist[aid+1]
                CL_list.append([a_mind1,b_mind, c_mind, d_mind,a_mind2, e_mind, f_mind, g_mind,h_mind])

    # bacdefg
    if len(blist)>=2:
        for bid in range(0,len(blist),2):
            for itag in range(maxlen):
                if itag>=len(alist):
                    a_mind=alist[len(alist)-1]
                else:
                    a_mind=alist[itag]

                if itag>=len(clist):
                    c_mind=clist[len(clist)-1]
                else:
                    c_mind=clist[itag]

                if itag>=len(dlist):
                    d_mind=dlist[len(dlist)-1]
                else:
                    d_mind=dlist[itag]

                if itag>=len(elist):
                    e_mind=elist[len(elist)-1]
                else:
                    e_mind=elist[itag]

                if itag>=len(flist):
                    f_mind=flist[len(flist)-1]
                else:
                    f_mind=flist[itag]

                if itag>=len(glist):
                    g_mind=glist[len(glist)-1]
                else:
                    g_mind=glist[itag]

                if itag >= len(hlist):
                    h_mind = hlist[len(hlist) - 1]
                else:
                    h_mind = hlist[itag]

                if bid + 1 >= len(blist):
                    b_mind1 = blist[bid]
                    b_mind2 = blist[len(blist) - 2]
                else:
                    b_mind1 = blist[bid]
                    b_mind2 = blist[bid + 1]
                CL_list.append([b_mind1,a_mind, c_mind, d_mind,b_mind2, e_mind, f_mind, g_mind,h_mind])

    # cabdefg
    if len(clist)>=2:
        for cid in range(0, len(clist), 2):
            for itag in range(maxlen):
                if itag>=len(blist):
                    b_mind=blist[len(blist)-1]
                else:
                    b_mind=blist[itag]

                if itag>=len(alist):
                    a_mind=alist[len(alist)-1]
                else:
                    a_mind=alist[itag]

                if itag>=len(dlist):
                    d_mind=dlist[len(dlist)-1]
                else:
                    d_mind=dlist[itag]

                if itag>=len(elist):
                    e_mind=elist[len(elist)-1]
                else:
                    e_mind=elist[itag]

                if itag>=len(flist):
                    f_mind=flist[len(flist)-1]
                else:
                    f_mind=flist[itag]

                if itag>=len(glist):
                    g_mind=glist[len(glist)-1]
                else:
                    g_mind=glist[itag]

                if itag >= len(hlist):
                    h_mind = hlist[len(hlist) - 1]
                else:
                    h_mind = hlist[itag]

                if cid + 1 >= len(clist):
                    c_mind1 = clist[cid]
                    c_mind2 = clist[len(clist) - 2]
                else:
                    c_mind1 = clist[cid]
                    c_mind2 = clist[cid + 1]
                CL_list.append([ c_mind1,a_mind,b_mind, d_mind,c_mind2, e_mind, f_mind, g_mind,h_mind])

    # dabcefg
    for did in range(0, len(dlist), 2):
        for d_mind in dlist:
            for itag in range(maxlen):
                if itag>=len(blist):
                    b_mind=blist[len(blist)-1]
                else:
                    b_mind=blist[itag]

                if itag>=len(clist):
                    c_mind=clist[len(clist)-1]
                else:
                    c_mind=clist[itag]

                if itag>=len(alist):
                    a_mind=alist[len(alist)-1]
                else:
                    a_mind=alist[itag]

                if itag>=len(elist):
                    e_mind=elist[len(elist)-1]
                else:
                    e_mind=elist[itag]

                if itag>=len(flist):
                    f_mind=flist[len(flist)-1]
                else:
                    f_mind=flist[itag]

                if itag>=len(glist):
                    g_mind=glist[len(glist)-1]
                else:
                    g_mind=glist[itag]

                if itag >= len(hlist):
                    h_mind = hlist[len(hlist) - 1]
                else:
                    h_mind = hlist[itag]

                if did + 1 >= len(dlist):
                    d_mind1 = dlist[did]
                    d_mind2 = dlist[len(dlist) - 2]
                else:
                    d_mind1 = dlist[did]
                    d_mind2 = dlist[did + 1]
                CL_list.append([d_mind1,a_mind,b_mind, c_mind,d_mind2,  e_mind, f_mind, g_mind,h_mind])

    # eabcdfg
    if len(elist)>=2:
        for eid in range(0, len(elist), 2):
            for itag in range(maxlen):
                if itag>=len(blist):
                    b_mind=blist[len(blist)-1]
                else:
                    b_mind=blist[itag]

                if itag>=len(clist):
                    c_mind=clist[len(clist)-1]
                else:
                    c_mind=clist[itag]

                if itag>=len(dlist):
                    d_mind=dlist[len(dlist)-1]
                else:
                    d_mind=dlist[itag]

                if itag>=len(alist):
                    a_mind=alist[len(alist)-1]
                else:
                    a_mind=alist[itag]

                if itag>=len(flist):
                    f_mind=flist[len(flist)-1]
                else:
                    f_mind=flist[itag]

                if itag>=len(glist):
                    g_mind=glist[len(glist)-1]
                else:
                    g_mind=glist[itag]

                if itag >= len(hlist):
                    h_mind = hlist[len(hlist) - 1]
                else:
                    h_mind = hlist[itag]

                if eid + 1 >= len(elist):
                    e_mind1 = elist[eid]
                    e_mind2 = elist[len(elist) - 2]
                else:
                    e_mind1 = elist[eid]
                    e_mind2 = elist[eid + 1]
                CL_list.append([e_mind1,a_mind,b_mind, c_mind,e_mind2, d_mind,  f_mind, g_mind,h_mind])

    # fabcdeg
    if len(flist)>=2:
        for fid in range(0, len(flist), 2):
            for itag in range(maxlen):
                if itag>=len(blist):
                    b_mind=blist[len(blist)-1]
                else:
                    b_mind=blist[itag]

                if itag>=len(clist):
                    c_mind=clist[len(clist)-1]
                else:
                    c_mind=clist[itag]

                if itag>=len(dlist):
                    d_mind=dlist[len(dlist)-1]
                else:
                    d_mind=dlist[itag]

                if itag>=len(elist):
                    e_mind=elist[len(elist)-1]
                else:
                    e_mind=elist[itag]

                if itag>=len(alist):
                    a_mind=alist[len(alist)-1]
                else:
                    a_mind=alist[itag]

                if itag>=len(glist):
                    g_mind=glist[len(glist)-1]
                else:
                    g_mind=glist[itag]

                if itag >= len(hlist):
                    h_mind = hlist[len(hlist) - 1]
                else:
                    h_mind = hlist[itag]

                if fid + 1 >= len(flist):
                    f_mind1 = flist[fid]
                    f_mind2 = flist[len(flist) - 2]
                else:
                    f_mind1 = flist[fid]
                    f_mind2 = flist[fid + 1]
                CL_list.append([ f_mind1,a_mind,b_mind, c_mind,f_mind2, d_mind, e_mind, g_mind,h_mind])

    # gabcdef
    if len(glist)>=2:
        for gid in range(0, len(glist), 2):
            for itag in range(maxlen):
                if itag>=len(blist):
                    b_mind=blist[len(blist)-1]
                else:
                    b_mind=blist[itag]

                if itag>=len(clist):
                    c_mind=clist[len(clist)-1]
                else:
                    c_mind=clist[itag]

                if itag>=len(dlist):
                    d_mind=dlist[len(dlist)-1]
                else:
                    d_mind=dlist[itag]

                if itag>=len(elist):
                    e_mind=elist[len(elist)-1]
                else:
                    e_mind=elist[itag]

                if itag>=len(flist):
                    f_mind=flist[len(flist)-1]
                else:
                    f_mind=flist[itag]

                if itag>=len(alist):
                    a_mind=alist[len(alist)-1]
                else:
                    a_mind=alist[itag]

                if itag >= len(hlist):
                    h_mind = hlist[len(hlist) - 1]
                else:
                    h_mind = hlist[itag]

                if gid + 1 >= len(glist):
                    g_mind1 = glist[gid]
                    g_mind2 = glist[len(glist) - 2]
                else:
                    g_mind1 = glist[gid]
                    g_mind2 = glist[gid + 1]
                CL_list.append([ g_mind1,a_mind,b_mind, c_mind,g_mind2, d_mind, e_mind, f_mind,h_mind])

    if len(hlist)>=2:
        for hid in range(0,len(hlist),2):
            for itag in range(0,maxlen):
                if itag>=len(blist):
                    b_mind=blist[len(blist)-1]
                else:
                    b_mind=blist[itag]

                if itag>=len(clist):
                    c_mind=clist[len(clist)-1]
                else:
                    c_mind=clist[itag]

                if itag>=len(dlist):
                    d_mind=dlist[len(dlist)-1]
                else:
                    d_mind=dlist[itag]

                if itag>=len(elist):
                    e_mind=elist[len(elist)-1]
                else:
                    e_mind=elist[itag]

                if itag>=len(flist):
                    f_mind=flist[len(flist)-1]
                else:
                    f_mind=flist[itag]

                if itag>=len(glist):
                    g_mind=glist[len(glist)-1]
                else:
                    g_mind=glist[itag]

                if itag>=len(alist):
                    a_mind=alist[len(alist)-1]
                else:
                    a_mind=alist[itag]

                if hid+1 >=len(hlist):
                    h_mind1=hlist[hid]
                    h_mind2=hlist[len(hlist)-2]
                else:
                    h_mind1=hlist[hid]
                    h_mind2=hlist[hid+1]
                CL_list.append([h_mind1,a_mind,b_mind, c_mind, h_mind2,d_mind, e_mind, f_mind, g_mind])

    return CL_list
def processall4CL():

    pathname='../datasets/CD/C2D2_dataset.csv'

    #tmpdataset为以扭曲标签为键，具有该标签的文本为值的字典；ana_dict为以场景为键，值是一个字典，以每类标签为建，对应思维为值的字典
    tmpdataset,ana_dict=processcsv(pathname)

    CLtrain_dataset={}
    setcount=0
    for ikey in  tqdm(ana_dict.keys()):
        tmp_cl_list=[]
        #采样只取具有所有标签的场景数据
        if len(ana_dict[ikey])==8:
            # keylist为0-7标签的列表，为了在采样构造函数中控制正负样顺序（无用）
            keylist=[]
            for akey in ana_dict[ikey]:
                keylist.append(akey)
            #采样构造函数，输入为该场景下每个标签对应的思维合集的列表，返回的是采样的9元组集合包含，一个学习例，一个正例，7个反例
            tmp_cl_list=tripleset_construct(ana_dict[ikey][keylist[0]],ana_dict[ikey][keylist[1]],ana_dict[ikey][keylist[2]],
                                            ana_dict[ikey][keylist[3]],ana_dict[ikey][keylist[4]],ana_dict[ikey][keylist[5]],
                                            ana_dict[ikey][keylist[6]],ana_dict[ikey][keylist[7]],tmp_cl_list)
            CLtrain_dataset[ikey]=tmp_cl_list
            setcount+=len(tmp_cl_list*9)

    print(setcount)
    with open("../datasets/CD/CL_dataset_8min.json", "w",encoding='utf-8-sig') as file:
        json.dump(CLtrain_dataset, file)

    with open("../datasets/CD/CL_dataset_8min.json", "r",encoding='utf-8-sig') as file:
        data1 = json.load(file)
    print(len(data1))

processall4CL()