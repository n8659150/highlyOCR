# coding: utf-8
import time
from PIL import Image
import cv2
import numpy as np
import pytesseract
import os
import jieba
import jieba.analyse
import shutil
import threading
# from multiprocessing import Process
import multiprocessing
from concurrent.futures import ThreadPoolExecutor as TPE
from concurrent.futures import ProcessPoolExecutor as PPE
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

def findTextRegion(img,threshold = 3000):
    region = []

    # 1. 查找轮廓
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt) 

        # 面积小的都筛选掉
        if(area < threshold):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # print ("rect is: ")
        # print (rect)

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
        print (path + ' 中的文字已识别')

# In[24]:
# 根据坐标画出区域，并切图保存
def drawTextArea(sourceImg,region,savingPath,color,lineWidth):
    print ("drawTextArea")
    for index,box in enumerate(region):
        # temp_img = cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        temp_img = cv2.drawContours(sourceImg, [box], 0, color, lineWidth)
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
        # temp_filename = fullPath + "\\" + str(index) + '.jpg'
        temp_filename = savingPath + "\\" + str(index) + '.jpg'
        # print (temp_filename)
        # cv2.imwrite(temp_filename,img_plate)
        cv2.imencode('.jpg',img_plate)[1].tofile(temp_filename)
        print (temp_filename + " 已定位")

# 提取文字
# 线程池设置
def grabText(folder):
    croppedImgList = os.listdir(folder)
    # 对列表进行初始化，并指定长度    
    results = [0] * len(croppedImgList)
    # manager = Manager()
    # results = manager.list()
    with TPE(multiprocessing.cpu_count()*4) as executor:
        for i in range(len(croppedImgList)):
            path = os.path.join(folder,croppedImgList[i])
            executor.submit(ocring,path,i,results)
    return results


# In[25]:

# 进行文字分词辨识并输出
def textProcess(textArray,savingPath):
    # 去除识别结果中的空格
    fullString = ''.join(textArray).replace(' ','')
    # 保存未分词的结果
    unsegDir = savingPath + '\\unseg_result.txt'
    r = open(unsegDir,'w')
    r.write(fullString)
    r.close()

    # 使用结巴分词进行处理
    segList = jieba.cut(fullString,HMM=False)
    # 保存结果
    segResult = '\r\n'.join(segList)
    finalResultDir = savingPath + '\\segged_result.txt'
    r = open(finalResultDir,'w')
    r.write(segResult)
    r.close()

    # 2017-12-01
    # 尝试从原文中提取标签 - TF-IDF 算法
    tagList = jieba.analyse.extract_tags(segResult, topK=20, withWeight=False, allowPOS=())
    tagResult = '\r\n'.join(tagList)
    # 保存结果
    taggedDir = savingPath + '\\tagged_result.txt'
    r = open(taggedDir,'w')
    r.write(tagResult)
    r.close()
    # 尝试从原文本中提取标签 - TextRank 算法
    tagList = jieba.analyse.textrank(segResult, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 
    tagResult = '\r\n'.join(tagList)
    # 保存结果
    taggedDir = savingPath + '\\ranked_result.txt'
    r = open(taggedDir,'w')
    r.write(tagResult)
    r.close()

# In[26]:
def cv_read(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8),-1)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img

def getAbsFolderPath(folder):
    return os.getcwd() + "\\" + folder + "\\" 

#图片压缩批处理  
def compressImage(srcPath,dstPath):  
    for filename in os.listdir(srcPath):  
        #如果不存在目的目录则创建一个，保持层级结构
        if not os.path.exists(dstPath):
                os.makedirs(dstPath)        

        #拼接完整的文件或文件夹路径
        srcFile=os.path.join(srcPath,filename)
        dstFile=os.path.join(dstPath,filename)
        print (srcFile)
        print (dstFile)

        #如果是文件就处理
        if os.path.isfile(srcFile):     
            #打开原图片缩小后保存，可以用if srcFile.endswith(".jpg")或者split，splitext等函数等针对特定文件压缩
            sImg=Image.open(srcFile)  
            w,h=sImg.size  
            print (w,h)
            dImg=sImg.resize((int(w/2),int(h/2)),Image.ANTIALIAS)  #设置压缩尺寸和选项，注意尺寸要用括号
            dImg.save(dstFile) #也可以用srcFile原路径保存,或者更改后缀保存，save这个函数后面可以加压缩编码选项JPEG之类的
            print (dstFile," compressed succeeded")

        #如果是文件夹就递归
        if os.path.isdir(srcFile):
            compressImage(srcFile,dstFile)




def detect(filePath):
    # 0. 读取图像
    # img = cv2.imread(filePath)
    # img = cv_read(filePath)
    img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    # 1. 转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)
    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)
    # 4. 获取路径    
    croppedFolderPath = getAbsFolderPath("cropped")
    fileNameOnly = os.path.basename(filePath).replace('.','_')
    fullPath = croppedFolderPath + fileNameOnly
    # 若文件夹不存在则新建文件夹
    if not os.path.exists(fullPath): 
        os.mkdir(fullPath)
    else:
        shutil.rmtree(fullPath)
        os.mkdir(fullPath)
    # 5.根据坐标画出区域，并切图保存
    drawTextArea(img,region,fullPath,(0,255,0),2)
    # 6.OCR识别
    results = grabText(fullPath)
    # 7.对识别结果进行分词
    textProcess(results,fullPath)

def multiProcessingDetect(fileNameList):
    for file in fileNameList:
        detect(file)

def submitOCRExecutor(ocrFunc,*fileListToBeScanned):
    for i in range(len(fileListToBeScanned)):     
        if not fileListToBeScanned[i]:
            continue
        else:
            executor.submit(ocrFunc,fileListToBeScanned[i])

if __name__=="__main__":
    
    startTime = time.time()
    imagesFolderPath_1 = getAbsFolderPath("imgP1")
    imagesFolderFileList_1 = os.listdir("imgP1")
    imgP1Dir = list(map(lambda x: imagesFolderPath_1 + x,imagesFolderFileList_1))
    imagesFolderPath_2 = getAbsFolderPath("imgP2")
    imagesFolderFileList_2 = os.listdir("imgP2")
    imgP2Dir = list(map(lambda x: imagesFolderPath_2 + x,imagesFolderFileList_2))
    imagesFolderPath_3 = getAbsFolderPath("imgP3")
    imagesFolderFileList_3 = os.listdir("imgP3")
    imgP3Dir = list(map(lambda x: imagesFolderPath_3 + x,imagesFolderFileList_3))
    imagesFolderPath_4 = getAbsFolderPath("imgP4")
    imagesFolderFileList_4 = os.listdir("imgP4")
    imgP4Dir = list(map(lambda x: imagesFolderPath_4 + x,imagesFolderFileList_4))
    #开4个进程同时处理，每个进程包含8个线程 
    with PPE(multiprocessing.cpu_count()*2) as executor:

      # compressImage("./imgP1","./imgP1_compressed")
        submitOCRExecutor(multiProcessingDetect,imgP1Dir,imgP2Dir,imgP3Dir,imgP4Dir)
    endTime = time.time()
    print (endTime - startTime," sec used")



