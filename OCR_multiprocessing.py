# coding: utf-8
import HIGHLY_OCR as HO
from multiprocessing import Process
import time
# 加载结巴分词自定义词库
# 机种编号词库
# jieba.load_userdict('dict\\fg_item_code.txt') 
# 机种名称词库
# jieba.load_userdict('dict\\fg_item_desc.txt') 

if __name__=="__main__":
    # 多进程测试
    processes = []
    startTime = time.time()
    for file in ['img_raw.jpg','img_raw1.jpg','smtz.jpg','smtz1.jpg']:
        process = Process(target=HO.detect,args=(file,))
        process.start()  
        processes.append(process)
    #加入线程池管理         
    for process in processes:
        process.join()
    endTime = time.time()
    t_used = endTime - startTime
    print '%d sec used' %t_used