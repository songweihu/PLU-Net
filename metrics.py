import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio
import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # print("SR GT", SR, GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # print("SR GT", SR, GT)

    # TP : True Positive
    # FN : False Negative

    # if SR is not None:
    #     print(1)
    # print(GT==False)
    TP = (SR==1)&(GT==1)
    FN = (SR==0)&(GT==1)
    # print("TP FN", TP, FN)
   
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    # print("SE", SE)
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = (SR==0)&(GT==0)
    FP = (SR==1)&(GT==0)

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = (SR==1)&(GT==1)
    FP = (SR==1)&(GT==0)


    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum(SR&GT)
    Union = torch.sum(SR|GT)
    # Inter = torch.sum(SR+GT)
    # Union = torch.sum(SR+GT>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum(SR&GT)
    # Inter = torch.sum(SR+GT)

    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC



class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

def get_iou(image_mask,predict):
    # image_mask = cv2.imread(mask_name,0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))
    #image_mask = mask
    # print(image.shape)
    height = predict.shape[0]
    weight = predict.shape[1]
    # print(height*weight)
    o = 0
    for row in range(height):
            for col in range(weight):
                if predict[row, col] < 0.5:  #由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
                    predict[row, col] = 0
                else:
                    predict[row, col] = 1
                if predict[row, col] == 0 or predict[row, col] == 1:
                    o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
            for col in range(weight_mask):
                if image_mask[row, col] < 125:   #由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
                    image_mask[row, col] = 0
                else:
                    image_mask[row, col] = 1
                if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                    o += 1
    predict = predict.astype(np.int16)

    interArea = np.multiply(predict, image_mask)
    tem = predict + image_mask
    unionArea = tem - interArea
    inter = np.sum(interArea)
    union = np.sum(unionArea)
    iou_tem = inter / union

    # Iou = IOUMetric(2)  #2表示类别，肝脏类+背景类
    # Iou.add_batch(predict, image_mask)
    # a, b, c, d, e= Iou.evaluate()
    print('%s:iou=%f' % (mask_name,iou_tem))

    return iou_tem

def get_dice(image_mask,predict):
    # image_mask = cv2.imread(mask_name, 0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))
    height = predict.shape[0]
    weight = predict.shape[1]
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  # 由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:  # 由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1
            if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                o += 1
    predict = predict.astype(np.int16)
    intersection = (predict*image_mask).sum()
    dice = (2. *intersection) /(predict.sum()+image_mask.sum())
    return dice

def get_hd(image_mask,predict):
    # image_mask = cv2.imread(mask_name, 0)
    # print(mask_name)
    # print(image_mask)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))
    #image_mask = mask
    height = predict.shape[0]
    weight = predict.shape[1]
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  # 由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:  # 由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1
            if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                o += 1
    hd1 = directed_hausdorff(image_mask, predict)[0]
    hd2 = directed_hausdorff(predict, image_mask)[0]
    res = None
    if hd1>hd2 or hd1 == hd2:
        res=hd1
        return res
    else:
        res=hd2
        return res

def show(predict):
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            predict[row, col] *= 255
    plt.imshow(predict)
    plt.show()