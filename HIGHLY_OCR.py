# coding: utf-8
import time
import sys
# sys.setdefaultencoding() does not exist, here!
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
from PIL import Image
import cv2
import numpy as np
import pytesseract
import os
import jieba
import jieba.analyse
import shutil
import threading
from multiprocessing import Process 
# 加载结巴分词自定义词库
# 机种编号词库
# jieba.load_userdict('dict\\fg_item_code.txt') 
# 机种名称词库
# jieba.load_userdict('dict\\fg_item_desc.txt') 


# In[5]:

# 原图片预处理
def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations = 1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations = 1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations = 3)

    return dilation2


# In[6]:

def findTextRegion(img):
    region = []

    # 1. 查找轮廓
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt) 

        # 面积小的都筛选掉
        if(area < 2000):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        print "rect is: "
        print rect

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if(height > width * 1.32):
            continue

        region.append(box)
        # print ("box is",box)
    return region


# In[19]:

# 打开图片识别文字
def ocring(path,index,resultContainer):
    if os.path.isfile(path):
        imageObject=Image.open(path)
        result = pytesseract.image_to_string(imageObject,lang='chi_sim')
        # resultContainer.append({"order":index,"resultText":result})
        resultContainer[index] = result
        print path + ' text grabbed'

# In[24]:

# 提取文字
def grabText(folder):
    croppedImgList = os.listdir(folder)
    # 对列表进行初始化，并指定长度    
    results = [0] * len(croppedImgList)
    # manager = Manager()
    # results = manager.list()
    processes = []
    for i in xrange(len(croppedImgList)):
        path = os.path.join(folder,croppedImgList[i])
        #识别文字设置多线程        
        p = threading.Thread(target=ocring,args=(path,i,results))
        p.start()  
        processes.append(p)
    #加入线程池管理         
    for process in processes:
        process.join()
    return results


# In[25]:

# 进行文字分词辨识并输出
def textProcess(textArray,savingPath):
    # 去除识别结果中的空格
    fullString = ''.join(textArray).replace(' ','')
    # 保存未分词的结果
    unsegDir = savingPath + '\\unseg_result.txt'
    r = open(unsegDir,'wb')
    r.write(fullString)
    r.close()

    # 使用结巴分词进行处理
    segList = jieba.cut(fullString,HMM=False)
    # 保存结果
    segResult = '\r\n'.join(segList)
    finalResultDir = savingPath + '\\segged_result.txt'
    r = open(finalResultDir,'wb')
    r.write(segResult)
    r.close()

    # 2017-12-01
    # 尝试从原文中提取标签 - TF-IDF 算法
    tagList = jieba.analyse.extract_tags(segResult, topK=20, withWeight=False, allowPOS=())
    tagResult = '\r\n'.join(tagList)
    # 保存结果
    taggedDir = savingPath + '\\tagged_result.txt'
    r = open(taggedDir,'wb')
    r.write(tagResult)
    r.close()
    # 尝试从原文本中提取标签 - TextRank 算法
    tagList = jieba.analyse.textrank(segResult, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 
    tagResult = '\r\n'.join(tagList)
    # 保存结果
    taggedDir = savingPath + '\\ranked_result.txt'
    r = open(taggedDir,'wb')
    r.write(tagResult)
    r.close()

# In[26]:

def detect(filePath):
    # 0. 读取图像
    img = cv2.imread(filePath)
    # 1. 转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)
    # 4. 获取路径    
    rootPath = os.getcwd()
    fileNameOnly = filePath.replace('.','_')
   
    fullPath = rootPath + '\\cropped\\' + fileNameOnly
    if not os.path.exists(fullPath): 
        os.mkdir(fullPath)
    else:
        shutil.rmtree(fullPath)
        os.mkdir(fullPath)
    # 5. 用绿线画出这些找到的轮廓
    for index,box in enumerate(region):
        temp_img = cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        ys_sorted_index = np.argsort(ys)
        xs_sorted_index = np.argsort(xs)
        x1 = box[xs_sorted_index[0], 0]
        x2 = box[xs_sorted_index[3], 0]
        y1 = box[ys_sorted_index[0], 1]
        y2 = box[ys_sorted_index[3], 1]
        img_org2 = temp_img.copy()   
        img_plate = img_org2[y1:y2, x1:x2]
        temp_filename = fullPath + "\\" + str(index) + '.jpg'
        cv2.imwrite(temp_filename,img_plate)
        print temp_filename + " is positioned"
    
    results = grabText(fullPath)
    textProcess(results,fullPath)
    print 'All done'


# In[27]:

if __name__=="__main__":
    # 多进程测试 - 4张图 90s
    processes = []
    t = time.time()
    for file in ['img_raw.jpg','img_raw1.jpg','smtz.jpg','smtz1.jpg']:
        p = Process(target=detect,args=(file,))
        p.start()  
        processes.append(p)
    #加入线程池管理         
    for process in processes:
        process.join()
    print (time.time() - t," sec used")

    # # 多线程测试 - 4张图 80s
    # threads = []
    # t = time.time()
    # for file in ['img_raw.jpg','img_raw1.jpg','smtz.jpg','smtz1.jpg']:
    #     t = threading.Thread(target=detect,args=(file,))
    #     t.start()  
    #     threads.append(t)
    # #加入线程池管理         
    # for thread in threads:
    #     thread.join()
    # print (time.time() - t," sec used")

# In[ ]:



