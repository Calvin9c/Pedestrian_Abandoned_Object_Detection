import scipy.io
import torch
import numpy as np
#import time
import os

#######################################################################
# Evaluate
def evaluate(qf,gf):
    query = qf.view(-1,1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    return score

######################################################################

# def main():
#     result = scipy.io.loadmat('pytorch_result.mat')
#     query_feature = torch.FloatTensor(result['query_f'])

#     gallery_feature = torch.FloatTensor(result['gallery_f'])

#     query_feature = query_feature.cuda()
#     gallery_feature = gallery_feature.cuda()

#     print(query_feature.shape)

#     ap = 0.0

#     for i in range(1):
#         score = evaluate(query_feature[i],gallery_feature)
#     print(score)

