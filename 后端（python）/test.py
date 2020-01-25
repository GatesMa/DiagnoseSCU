import os

from flask import Flask, request
import pymysql as pymysql
import Util

import cv2 #！！！！！
from skimage.feature import local_binary_pattern #！！！！！
from pylab import * #！！！！！
import dlib #！！！！！
from imutils.face_utils import FaceAligner #！！！！！
import numpy as np #！！！！！
import os
from skimage.io import imread, imsave #！！！！！

from libsvm.python.svmutil import * #！！！！！

from PIL import Image
import matplotlib.pyplot as plt   #！！！！！
from glob import  glob
import PIL.Image as img

from urllib import request

np.set_printoptions(threshold=100000000)
np.set_printoptions(suppress=True)



app = Flask(__name__)


@app.route('/')
def hello_world():
    # update_num()
    return 'Diagnose! 面瘫诊断助手 ------> 后台～'



# 获取当前诊断的ID值
@app.route('/get_number')
def get_number() :
    # 连接数据库

    # 也可以使用关键字参数
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123qwe', db='diagnose', charset='utf8')
    conn.autocommit(1)  # conn.autocommit(True)

    # 使用cursor()方法获取操作游标
    cursor = conn.cursor()
    # 因该模块底层其实是调用CAPI的，所以，需要先得到当前指向数据库的指针。

    try:
        cursor.execute("SELECT * FROM number where num_key = 'Key'")        # 执行sql语句

        results = cursor.fetchall()  # 获取查询的所有记录
        # print("num_key", "num")
        # 遍历结果
        # for row in results:
        #     num_key = row[0]
        #     num = row[1]
        #     print(num_key, num)
        num = str(results[0][1])
        print('num:' + num)
        return num

    except:
        import traceback

        traceback.print_exc()
        # 发生错误时回滚
        conn.rollback()
    finally:
        # 关闭游标连接
        cursor.close()
        # 关闭数据库连接
        conn.close()


@app.route('/upload_img1', methods=['POST'])
def upload_img1() :
    f = request.files['file']
    print('save img1')

    id = request.values.get('id')

    print('id:', id)

    Dir_name = "WebTransUserData/" + str(id) + '/'

    os.makedirs(Dir_name, exist_ok=True)  # 以该名称建一个文件夹

    f.save(Dir_name + 'img1.jpg')
    return "1"

@app.route('/upload_img2', methods=['POST'])
def upload_img2() :
    f = request.files['file']
    print('save img2')

    id = request.values.get('id')

    print('id:', id)

    Dir_name = "WebTransUserData/" + str(id) + '/'

    os.makedirs(Dir_name, exist_ok=True)  # 以该名称建一个文件夹

    f.save(Dir_name + 'img2.jpg')
    return "1"

@app.route('/upload_video1', methods=['POST'])
def upload_video1() :
    f = request.files['file']
    print('save video1')

    id = request.values.get('id')

    print('id:', id)

    Dir_name = "WebTransUserData/" + str(id) + '/'

    os.makedirs(Dir_name, exist_ok=True)  # 以该名称建一个文件夹

    f.save(Dir_name + 'video1.mp4')

    return "1"

@app.route('/upload_video2', methods=['POST'])
def upload_video2() :
    f = request.files['file']
    print('save video2')

    id = request.values.get('id')

    print('id:', id)

    Dir_name = "WebTransUserData/" + str(id) + '/'

    os.makedirs(Dir_name, exist_ok=True)  # 以该名称建一个文件夹

    f.save(Dir_name + 'video2.mp4')

    return "1"


def update_num() :
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123qwe', db='diagnose', charset='utf8')
    conn.autocommit(1)  # conn.autocommit(True)

    # 使用cursor()方法获取操作游标
    cursor = conn.cursor()
    # 因该模块底层其实是调用CAPI的，所以，需要先得到当前指向数据库的指针。

    try:
        cursor.execute("SELECT * FROM number where num_key = 'Key'")  # 执行sql语句

        results = cursor.fetchall()  # 获取查询的所有记录
        # print("num_key", "num")
        # 遍历结果
        # for row in results:
        #     num_key = row[0]
        #     num = row[1]
        #     print(num_key, num)
        num = results[0][1]

        num = num + 1

        cursor.execute("UPDATE number set num = " + str(num) + " where num_key = 'Key'")  # 执行sql语句
        conn.commit()

        print('num:' + str(num))
        return str(num)

    except:
        import traceback

        traceback.print_exc()
        # 发生错误时回滚
        conn.rollback()
    finally:
        # 关闭游标连接
        cursor.close()
        # 关闭数据库连接
        conn.close()


# 接受参数num，并处理相应文件夹下的数据，返回面瘫等级
@app.route('/diag_and_return/<id>', methods=['GET'])
def diag_and_return(id):


    resp = request.urlopen('http://localhost:9004/diag_and_return/' + str(id))
    print(resp.read().decode())

    #if os.path.exists("WebTransUserData/"+ str(id) +"/img1.jpg") and os.path.exists("WebTransUserData/"+ str(id) +"/img2.jpg") and os.path.exists("WebTransUserData/"+ str(id) +"video1.mp4") and os.path.exists("WebTransUserData/"+ str(id) +"video2.mp4"):
    # res = Util.finalTest("WebTransUserData/"+ str(id) +"/img1.jpg", "WebTransUserData/"+ str(id) +"/img2.jpg"
    #                      , "WebTransUserData/"+ str(id) +"/video1.mp4", "WebTransUserData/"+ str(id) +"/video2.mp4", id)
   # else:
       # res = -1

    # 如果诊断成功， 数据库id++
    #if res != -1:

    # update_num()
    # return str(res);

#----####################################3----#######------


##########################################################

if __name__ == '__main__':
    print("APP RUN")

    app.run(
        port = 9004,
        debug = True
    )

