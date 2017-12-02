# coding: utf-8
import HIGHLY_OCR as HO
import threading
import time
# 加载结巴分词自定义词库
# 机种编号词库
# jieba.load_userdict('dict\\fg_item_code.txt') 
# 机种名称词库
# jieba.load_userdict('dict\\fg_item_desc.txt') 

if __name__=="__main__":
    # 多线程测试
    threads = []
    startTime = time.time()
    for file in ['img_raw.jpg','img_raw1.jpg','smtz.jpg','smtz1.jpg']:
        thread = threading.Thread(target=HO.detect,args=(file,))
        thread.start()
        threads.append(thread)
    #加入线程池管理         
    for thread in threads:
        thread.join()
    endTime = time.time()
    t_used = endTime - startTime
    print '%d sec used' %t_used