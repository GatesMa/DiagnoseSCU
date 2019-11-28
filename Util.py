import cv2
from skimage.feature import local_binary_pattern
from pylab import *
import dlib
from imutils.face_utils import FaceAligner
import numpy as np
import os
from skimage.io import imread, imsave

from libsvm.python.svmutil import *
from libsvm.python.svm import *
from libsvm.python.commonutil import *
from PIL import Image
import matplotlib.pyplot as plt
from glob import  glob
import PIL.Image as img


np.set_printoptions(threshold=100000000)
np.set_printoptions(suppress=True)


def save_img(url, user_id, type):
    """
    将视频解析成桢图片 传入视频名称
    :param url: 视频url，字符串类型
    :param num: 第几个用户
    :param type: 示齿还是抬眉  0 轻微示齿 1 严重示齿 2 抬眉
    :return:
    """
    Dir_name = "UserData\\" + str(user_id) + "\\" + str(type)
    print("Dir:" + Dir_name)
    os.makedirs(Dir_name, exist_ok=True)  # 以该名称建一个文件夹
    vc = cv2.VideoCapture(url)
    c = 1
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
        print("打开了视频：" + url + "，开始处理")
    else:
        rval = False
        print("视频：" + url + "无法打开")
    while rval:  # 循环读取视频帧

        rval, frame = vc.read()
        cv2.imwrite(Dir_name + '/' + str(c) + '.jpg', frame)  # 存储为图像
        print(Dir_name + '/' + str(c) + '.jpg' + "   save success")
        c = c + 1
        cv2.waitKey(1)
    vc.release()
    print('ALL DONE！')

def HOG_features(im):
    """
    提取图片HOG特征
    :param im: openCV格式的数据 - 用cv2.imread
    :return:
    # """
    # hog = cv2.HOGDescriptor()
    # winStride = (16, 16)
    # padding = (0, 0)
    # hist = hog.compute(im, winStride, padding)
    # hist = hog.compute(im)
    # hist = hist.reshape((-1,))
    # return hist

    winSize = (16, 16)
    blockSize = (16, 16)
    blockStride = (1, 1)
    cellSize = (16, 16)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    winStride = (1, 1)
    padding = (0, 0)
    test_hog = hog.compute(im, winStride, padding).reshape((-1,))

    cv2.normalize(test_hog, test_hog).flatten()
    return test_hog

'''test'''
def face_crop_gray(image):
    '''
    传入图片路径，将图片中的人脸识别出来，再进行切割（灰色黑白）
    :param image: 图片路径，字符串
    :return: HOG
    '''
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    frame = cv2.imread(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    face_id = 1
    count = 1
    for (x, y, w, h) in faces:
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('image', gray)

##############################-----------lbp特征及统计---------################################
def get_Uniform(image, step, width, height):
    """
    获取图片LBP特征
    :param image_src: 图片路径  e.g. 1.jpg
    :param step:  步长
    :param width: 块宽度
    :param height: 块高度
    :return:
    """
    # settings for LBP
    radius = 1  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数

    # 读取图像
    #image = cv2.imread(image_src)
    # image = face_align(image_src)
    # print('image:')
    # print(image)

    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 获取lbp特征数据

    # cv2.imshow('resize0', image)
    # cv2.waitKey()

    lbp = local_binary_pattern(image, n_points, radius)
    cv2.imwrite('lbp.jpg', lbp)
    cv2.imshow('lbp', lbp)
    cv2.waitKey()

    # 转int
    lbp = np.array(lbp).astype(np.int)
    # 转nparray
    array = np.array(lbp)

    # 从（1，1）开始，每隔步长取一个width*height的矩阵，统计直方图
    startW = 1  # 开始横坐标
    startH = 1  # 开始纵坐标
    step = step  # 步长
    width = width  # 取的矩阵为5列
    height = height  # 取的矩阵为3行
    result = []

    for i in range(0, 15):
        wNow = startW + step * i
        hNow = startH + step * i
        tempArray = array[wNow:wNow + height, hNow:hNow + width]
        # 先写回，得到opencv的数据格式
        cv2.imwrite('temp.jpg', tempArray)
        # 再读出来
        img = cv2.imread('temp.jpg')
        # 统计直方图
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist).flatten()

        hist = [i for item in hist for i in item]
        # 拼接
        result.extend(hist)

    result = np.array(result)
    return result

def get_Uniform_2(image, step, width, height):
    """
    获取图片LBP特征
    :param image_src: 图片路径  e.g. 1.jpg
    :param step:  步长
    :param width: 块宽度
    :param height: 块高度
    :return:
    """
    # settings for LBP
    radius = 1  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(image, n_points, radius, 'default')

    max_bins = int(lbp.max() + 1)

    hist , _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins));
    # 转int
    # lbp = np.array(lbp).astype(np.int)
    # # 转nparray
    # array = np.array(lbp)

    # # 从（1，1）开始，每隔步长取一个width*height的矩阵，统计直方图
    # startW = 1  # 开始横坐标
    # startH = 1  # 开始纵坐标
    # step = step  # 步长
    # width = width  # 取的矩阵为5列
    # height = height  # 取的矩阵为3行
    # result = []

    # # 15个块
    # for i in range(0, 15):
    #     wNow = startW + step * i
    #     hNow = startH + step * i
    #     tempArray = array[wNow:wNow + height, hNow:hNow + width]
    #     # 先写回，得到opencv的数据格式
    #     cv2.imwrite('temp.jpg', tempArray)
    #     # 再读出来
    #     img = cv2.imread('temp.jpg')
    #     # 统计直方图
    #     hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #     cv2.normalize(hist, hist).flatten()

    #     hist = [i for item in hist for i in item]
    #     # 拼接
    #     result.extend(hist)

    # result = np.array(result)
    return hist


# 原始LBP特征
def LBP(src):
    '''
    :param src:灰度图像
        img = cv2.imread('normal_face.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    :return:
    '''
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()

    lbp_value = np.zeros((1,8), dtype=np.uint8)
    neighbours = np.zeros((1,8), dtype=np.uint8)
    for x in range(1, width-1):
        for y in range(1, height-1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = lbp
    hist = cv2.calcHist([dst], [0], None, [256], [0, 255])
    cv2.normalize(hist, hist).flatten()
    return hist
#----------------------------------
# 计算跳变次数的函数
def getHopCnt(num):
    '''
    :param num:8位的整形数，0-255
    :return:
    '''
    if num > 255:
        num = 255
    elif num < 0:
        num = 0

    num_b = bin(num)
    num_b = str(num_b)[2:]

    # 补0
    if len(num_b) < 8:
        temp = []
        for i in range(8-len(num_b)):
            temp.append('0')
        temp.extend(num_b)
        num_b = temp

    cnt = 0
    for i in range(8):
        if i == 0:
            former = num_b[-1]
        else:
            former = num_b[i-1]
        if former == num_b[i]:
            pass
        else:
            cnt += 1

    return cnt
# 归一化函数
def img_max_min_normalization(src, min=0, max=255):
    height = src.shape[0]
    width = src.shape[1]
    if len(src.shape) > 2:
        channel = src.shape[2]
    else:
        channel = 1

    src_min = np.min(src)
    src_max = np.max(src)

    if channel == 1:
        dst = np.zeros([height, width], dtype=np.float32)
        for h in range(height):
            for w in range(width):
                dst[h, w] = float(src[h, w] - src_min) / float(src_max - src_min) * (max - min) + min
    else:
        dst = np.zeros([height, width, channel], dtype=np.float32)
        for c in range(channel):
            for h in range(height):
                for w in range(width):
                    dst[h, w, c] = float(src[h, w, c] - src_min) / float(src_max - src_min) * (max - min) + min

    return dst
# uniform LBP的主体函数:其中norm表示是否要进行归一化，因为使用uniform LBP出来的值是0-58的，显示效果可能不明显。
def uniform_LBP(src, norm=True):
    '''
    :param src:原始图像
    :param norm:是否做归一化到【0-255】的灰度空间
    :return:
    '''
    table = np.zeros((256), dtype=np.uint8)
    temp = 1
    for i in range(256):
        if getHopCnt(i) <= 2:
            table[i] = temp
            temp += 1
    height = src.shape[0]
    width = src.shape[1]
    dst = np.zeros([height, width], dtype=np.uint8)
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = table[lbp]

    hist = cv2.calcHist([dst], [0], None, [59], [0, 58])
    cv2.normalize(hist, hist).flatten()
    return hist

    # if norm is True:
    #     return img_max_min_normalization(dst)
    # else:
    #     return dst

#----------------------------------
# 先定义计算旋转后灰度值的函数，以保证旋转不变的结果
def value_rotation(num):
    value_list = np.zeros((8), np.uint8)
    temp = int(num)
    value_list[0] = temp
    for i in range(7):
        temp = ((temp << 1) | int(temp / 128)) % 256
        value_list[i+1] = temp
    return np.min(value_list)
# 均匀模式+旋转不变模式LBP
def rotation_invariant_uniform_LBP(src):
    table = np.zeros((256), dtype=np.uint8)
    temp = 1
    for i in range(256):
        if getHopCnt(i) <= 2:
            table[i] = temp
            temp += 1

    height = src.shape[0]
    width = src.shape[1]
    dst = np.zeros([height, width], dtype=np.uint8)
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = table[lbp]

    # dst = img_max_min_normalization(dst)
    # print(dst)
    for x in range(width):
        for y in range(height):
            dst[y, x] = int(value_rotation(dst[y, x]))
    # return dst
    # print(dst)
    hist = cv2.calcHist([dst], [0], None, [59], [0, 58])
    # print(hist)
    cv2.normalize(hist, hist).flatten()
    return hist
#----------------------------------
# 旋转不变的LBP
def rotation_invariant_LBP(src):
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            # 旋转不变值
            dst[y, x] = value_rotation(lbp)

    # return dst
    hist = cv2.calcHist([dst], [0], None, [256], [0, 255])
    # print(hist)
    cv2.normalize(hist, hist).flatten()
    return hist
#----------------------------------

###########################################################################################
# def get_landmarks_PRNet(image_src) :
#     """
#     获取一张人脸图片的68个特征点（无人脸时返回"no face"）PRNet方法
#     :param image_src:
#     :return:
#     """
#     image = imread(image_src)
#
#     # ---- init PRN
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU number, -1 for CPU
#     prn = PRN(is_dlib=True)
#
#     # use dlib to detect face
#     pos = prn.process(image)
#     # get landmarks
#     kpt = prn.get_landmarks(pos)
#
#     # 想要在图片中展示，将这段代码取消注释即可
#     # 3D vertices
#
#     # vertices = prn.get_vertices(pos)
#     #
#     # camera_matrix, pose = estimate_pose(vertices)
#     # image_pose = plot_pose_box(image, camera_matrix, kpt)
#     # cv2.imshow('sparse alignment', plot_kpt(image, kpt))
#     # cv2.imshow('dense alignment', plot_vertices(image, vertices))
#     # cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
#     # cv2.waitKey(0)
#
#     return kpt

def find_max_face(faces):
    """
    获取最大的人像
    :param faces: dlib处理过的人脸对象
    :return:
    """
    max_area = 0
    max_id = 0
    for i in range(len(faces)):
        area = faces[i].width()*faces[i].height()
        if(area > max_area):
            max_id = i
            max_area = area
    return max_id, max_area

def get_68_landmarks(im):
    """
    获取一张人脸图片的68个特征点（无人脸时返回"no face"）dlib方法 传入openCV格式 - RGB
    :param im: opencv格式矩阵
    :return:
    """
    # dlib检测器与预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data\\dlib\\shape_predictor_68_face_landmarks.dat')

    # img_rd = im
    img_gray = im

    # 人脸数以及每个人脸的特征点
    faces = detector(img_gray, 0)

    # lpf 加入的内容###########################
    max_id, max_area = find_max_face(faces)
    #########################################

    # print("68个特征点，人脸数：" + str(len(faces)))

    # 取特征点坐标
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img_gray, faces[max_id]).parts()])

    # 画点用
    '''
    for idx, point in enumerate(landmarks):
        # 68 点的坐标
        pos = (point[0, 0], point[0, 1])

        # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
        cv2.circle(img_rd, pos, 2, color=(139, 0, 0))
        # 利用 cv2.putText 写数字 1-68
        cv2.putText(img_rd, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

    cv2.namedWindow("image", 1)

    cv2.imshow("image", img_rd)
    cv2.waitKey(0)
    '''
    return img_gray, landmarks

def face_align(img_path):
    """
    用dlib对人脸图片进行对齐
    :param img_path: 图片路径 - str
    :return:
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data\\dlib\\shape_predictor_68_face_landmarks.dat')
    # 初始化 FaceAligner 类对象
    fa = FaceAligner(predictor)# , desiredFaceWidth=256

    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    if len(rects) == 0:
        print('图片中检测不到人脸')
        return
    face_aligned = fa.align(image, gray, rects[0])
    face_aligned = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("dataset/align.jpg", face_aligned)

    return face_aligned

def area_cut(image, landmarks, multiple1, multiple2, multiple3) :
    """
    根据特征点对图片区域进行截取
    :param image: opencv格式的对齐图片矩阵
    :param landmarks: 68个特征点
    :param multiple1: 眼睛放大倍数
    :param multiple2: 嘴巴放大倍数
    :param multiple3: 正常表情放大倍数
    :return:
    """

    array = np.array(image)
    # --------------------------------------------
    # 左眼 37 ～ 42
    max_x = -1
    min_x = 10000
    max_y = -1
    min_y = 10000
    for idx, point in enumerate(landmarks):

        if 36<=  idx <= 41:
            x, y = (point[0, 0], point[0, 1])
            max_x = max(max_x, x)
            min_x = min(min_x, x)
            max_y = max(max_y, y)
            min_y = min(min_y, y)
    x_add = int(((max_x - min_x) * (multiple1 - 2)) / 2)
    y_add = int(((max_y - min_y) * (multiple1 - 1)) / 2)
    leftEye = array[min_y - y_add:max_y + y_add, min_x - x_add:max_x + x_add]
    # cv2.imwrite('left_eye.jpg', leftEye)
    # cv2.imshow('resize0', leftEye)
    # cv2.waitKey()
    # --------------------------------------------
    # 右眼 43 ～ 48
    max_x = -1
    min_x = 10000
    max_y = -1
    min_y = 10000
    for idx, point in enumerate(landmarks):

        if 42 <= idx <= 47:
            x, y = (point[0, 0], point[0, 1])
            max_x = max(max_x, x)
            min_x = min(min_x, x)
            max_y = max(max_y, y)
            min_y = min(min_y, y)
    x_add = int(((max_x - min_x) * (multiple1 - 1)) / 2)
    y_add = int(((max_y - min_y) * (multiple1 - 1)) / 2)
    rightEye = array[min_y - y_add:max_y + y_add, min_x - x_add:max_x + x_add]
    # cv2.imwrite('right_eye.jpg', rightEye)
    # cv2.imshow('resize0', rightEye)
    # cv2.waitKey()
    # --------------------------------------------
    # 嘴巴 49 ～ 68
    max_x = -1
    min_x = 10000
    max_y = -1
    min_y = 10000
    for idx, point in enumerate(landmarks):

        if 48 <= idx <= 67:
            x, y = (point[0, 0], point[0, 1])
            max_x = max(max_x, x)
            min_x = min(min_x, x)
            max_y = max(max_y, y)
            min_y = min(min_y, y)
    x_add = int(((max_x - min_x) * (multiple2 - 1)) / 2)
    y_add = int(((max_y - min_y) * (multiple2 - 1)) / 2)
    mouth = array[min_y - y_add:max_y + y_add, min_x - x_add:max_x + x_add]
    # cv2.imwrite('mouth.jpg', mouth)
    # cv2.imshow('resize0', mouth)
    # cv2.waitKey()
    # --------------------------------------------
    # 正常表情 41 42、48 47、6、12，然后在这些点里找一个框
    max_x = -1
    min_x = 10000
    max_y = -1
    min_y = 10000
    for idx, point in enumerate(landmarks):
        if idx == 40 or idx == 41 or idx == 46 or idx == 47 or idx == 5 or idx == 11:
            x, y = (point[0, 0], point[0, 1])
            max_x = max(max_x, x)
            min_x = min(min_x, x)
            max_y = max(max_y, y)
            min_y = min(min_y, y)
    x_add = int(((max_x - min_x) * (multiple3 - 1)) / 2)
    y_add = int(((max_y - min_y) * (multiple3 - 1)) / 2)
    normal_face = array[min_y - y_add:max_y + y_add, min_x - x_add:max_x + x_add]
    # cv2.imwrite('normal_face.jpg', normal_face)
    # cv2.imshow('resize0', normal_face)
    # cv2.waitKey()
    return leftEye, rightEye, mouth, normal_face

def img_crop_teeth_hog(img_grey, landmarks):
    '''
    对一张图片进行切割，resize，获取图片的动作特征 - 嘴巴
    :param img_path:图片地址
    :return: 一张图片的动作特征，一个list
    '''
    # im = face_align(img_path)
    # im = img_path
    # im = img_path
    im = img_grey
    # 先获取特征点，划出人脸区域
    # img_gray, landmarks = get_68_landmarks(img_grey)
    max_x = -1
    min_x = 10000
    max_y = -1
    min_y = 10000
    for idx, point in enumerate(landmarks):
        x, y = (point[0, 0], point[0, 1])
        max_x = max(max_x, x)
        min_x = min(min_x, x)
        max_y = max(max_y, y)
        min_y = min(min_y, y)

    im = im[min_y:max_y, min_x:max_x]

    im = cv2.resize(im, (300, 300), interpolation=cv2.INTER_CUBIC)

    # cv2.imwrite('img_crop_hog.jpg', im)
    # cv2.imshow('resize0', im)
    # cv2.waitKey()

    #调整特征点的大小
    for idx, point in enumerate(landmarks):
        point[0, 0] -= min_x
        point[0, 1] -= min_y
        point[0, 0] = int(point[0, 0] * 300 / (max_x - min_x))
        point[0, 1] = int(point[0, 1] * 300 / (max_y - min_y))

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # for idx, point in enumerate(landmarks):
    #     # 68 点的坐标
    #     pos = (point[0, 0], point[0, 1])
    #
    #     # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
    #     cv2.circle(im, pos, 2, color=(139, 0, 0))
    #     # 利用 cv2.putText 写数字 1-68
    #     cv2.putText(im, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)
    #
    # cv2.namedWindow("image", 1)
    #
    # cv2.imshow("image", im)
    # cv2.waitKey(0)

    motion_features = []

    # print(len(landmarks))

    for idx, point in enumerate(landmarks):
        if 48 <= idx and idx <= 67:
            x, y = (point[0, 0], point[0, 1])
            # print(x,y)
            array_teeth = im[y - 8:y + 8, x - 8: x + 8] # 取16*16方块
            hog_array = HOG_features(array_teeth)
            for i in hog_array:
                motion_features.append(i)
            # print(len(hog_array))
            # print("-----------------------------")
    # print("motion_features:")
    # print(len(motion_features))
    return motion_features

def img_crop_eye_hog(img_grey, landmarks):
    '''
    对一张图片进行切割，resize，获取图片的动作特征 - 左眉毛
    :param img_path:图片地址
    :return: 一张图片的动作特征，一个list
    '''
    # im = face_align(img_path)
    im = img_grey
    # 先获取特征点，划出人脸区域
    # img_gray, landmarks = get_68_landmarks(im)
    max_x = -1
    min_x = 10000
    max_y = -1
    min_y = 10000
    for idx, point in enumerate(landmarks):
        x, y = (point[0, 0], point[0, 1])
        max_x = max(max_x, x)
        min_x = min(min_x, x)
        max_y = max(max_y, y)
        min_y = min(min_y, y)

    im = im[min_y-30:max_y+30, min_x-30:max_x+30] #因为这里，如果不进行加减，直接裁剪的话，后面取16的方块是有边界问题的
    # cv2.imwrite('img_crop_hog.jpg', im)
    # cv2.imshow('resize0', im)
    # cv2.waitKey()
    im = cv2.resize(im, (300, 300), interpolation=cv2.INTER_CUBIC)


    # cv2.imwrite('img_crop_hog.jpg', im)
    # cv2.imshow('resize0', im)
    # cv2.waitKey()

    # 调整特征点的大小
    # for idx, point in enumerate(landmarks):
    #     point[0, 0] -= min_x
    #     point[0, 1] -= min_y
    #     point[0, 0] = int(point[0, 0] * 300 / (max_x - min_x))
    #     point[0, 1] = int(point[0, 1] * 300 / (max_y - min_y))
    #

    # img_gray, landmarks = get_68_landmarks(im) #再求一次特征点

    motion_features = []

    for idx, point in enumerate(landmarks):
        if 17 <= idx and idx <= 21:
            x, y = (point[0, 0], point[0, 1])
            array_teeth = im[y - 8:y + 8, x - 8: x + 8]  # 取16*16方块
            hog_array = HOG_features(array_teeth)
            for i in hog_array:
                motion_features.append(i)
    # print(motion_features)
    return motion_features

############################################################################################

def getNormalFaceLevel(image_path):
    '''
    获得正常表情分类器的等级
    :param image: 正常表情图片
    :return:
    '''
    y = []
    x = []
    print("获取正常表情等级，提取下列图片的特征：" + image_path)

    model = svm_load_model('model\\z1.model')

    face_aligned = face_align(image_path)  # 对齐
    img_grey, landmarks = get_68_landmarks(face_aligned)  # 特征点和灰度open CV格式图片
    leftEye, rightEye, mouth, normal_face = area_cut(img_grey, landmarks, 5, 1.3, 1.0)  # 切割
    lbp = rotation_invariant_LBP(normal_face)

    d = {}
    num = 1
    for i in lbp:
        d[num] = i.item()
        num += 1

    x.append(d)  # x
    y.append(0)  #

    p_label, p_acc, p_val = svm_predict(y, x, model)
    print(p_label)
    return int(p_label[0])

def getEyeClosedLevel(image_path):
    '''
    获得正常表情分类器的等级
    :param image: 正常表情图片
    :return:
    '''
    y = []
    x = []
    print("获取闭眼表情等级，提取下列图片的特征：" + image_path)

    model = svm_load_model('model\\z2.model')

    face_aligned = face_align(image_path)  # 对齐
    img_grey, landmarks = get_68_landmarks(face_aligned)  # 特征点和灰度open CV格式图片
    leftEye, rightEye, mouth, normal_face = area_cut(img_grey, landmarks, 3.0, 1.3, 1.0)  # 切割

    # image_path = cv2.imread(image_path)
    #
    # image_path = cv2.cvtColor(image_path, cv2.COLOR_RGB2GRAY)
    lbp = rotation_invariant_LBP(leftEye)

    d = {}
    num = 1
    for i in lbp:
        d[num] = i.item()
        num += 1

    x.append(d)  # x
    y.append(0)  #

    p_label, p_acc, p_val = svm_predict(y, x, model)
    print(p_label)
    return int(p_label[0])

def getSeriousTeethLevel(video_path, user_id):
    save_img(video_path, user_id, 1)
    Dir_name = "UserData\\" + str(user_id) + "\\1"
    file_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(Dir_name))])
    print("共" + str(file_sum) + "帧")


    num1 = 1
    num2 = int(file_sum / 4)
    num3 = int(file_sum / 2)
    num4 = int(file_sum * 3 / 4)
    num5 = file_sum

    print("获取严重示齿，提取下列图片的特征：" + str(num1) + "," + str(num2) + "," + str(num3) + "," + str(num4) + "," + str(num5))

    # 一个文件夹数据
    num = 1

    image_path_list = []
    image_folder = Dir_name
    types = ('*.jpg', '*.png')

    yt = []
    xt = []

    d = {}

    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    k = 1
    for i, image_path in enumerate(image_path_list):
        if '' in image_path and ( k == num1 or k == num2 or k == num3 or k == num4 or k == num5):
            ########################接下来是图片旋转
            im = img.open(image_path)
            # ng = im.transpose(img.ROTATE_180) #旋转 180 度角。
            # ng = im.transpose(img.FLIP_LEFT_RIGHT) #左右对换。
            #ng = im.transpose(img.FLIP_TOP_BOTTOM)  # 上下对换。
            ng = im.transpose(Image.ROTATE_90) #旋转 90 度角。
            ng.save(image_path)
            ########################上面是图片旋转
            # print("正在对齐以下图片：" + image_path)
            face_aligned = face_align(image_path)  # 对齐
            # print('一张图片对齐完成')
            img_grey, landmarks = get_68_landmarks(face_aligned)  # 特征点和灰度open CV格式图片
            leftEye, rightEye, mouth, normal_face = area_cut(img_grey, landmarks, 5, 1.3, 1.0)  # 切割
            hog_array = img_crop_teeth_hog(img_grey, landmarks)
            for i in hog_array:
                d[num] = i.item()
                num += 1
        k += 1
    yt.append(1)
    xt.append(d)

    model = svm_load_model('model\\z4.model')
    p_label, p_acc, p_val = svm_predict(yt, xt, model)
    print(p_label)
    return int(p_label[0])

def getLightTeethLevel(video_path, num):
    save_img(video_path, num, 0)
    Dir_name = "UserData\\" + str(num) + "\\0"
    file_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(Dir_name))])
    print("共" + str(file_sum) + "帧")


    num1 = 1
    num2 = int(file_sum / 4)
    num3 = int(file_sum / 2)
    num4 = int(file_sum * 3 / 4)
    num5 = file_sum
    print("获取轻微示齿，提取下列图片的特征：" + str(num1) + "," + str(num2) + "," + str(num3) + "," + str(num4) + "," + str(num5))

    # 一个文件夹数据
    num = 1

    image_path_list = []
    image_folder = Dir_name
    types = ('*.jpg', '*.png')

    yt = []
    xt = []

    d = {}

    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    k = 1
    for i, image_path in enumerate(image_path_list):
        if '' in image_path and ( k == num1 or k == num2 or k == num3 or k == num4 or k == num5):
            ########################接下来是图片旋转
            im = img.open(image_path)
            # ng = im.transpose(img.ROTATE_180) #旋转 180 度角。
            # ng = im.transpose(img.FLIP_LEFT_RIGHT) #左右对换。
            # ng = im.transpose(img.FLIP_TOP_BOTTOM)  # 上下对换。
            ng = im.transpose(Image.ROTATE_90)  # 旋转 90 度角。
            ng.save(image_path)
            ########################上面是图片旋转
            face_aligned = face_align(image_path)  # 对齐
            img_grey, landmarks = get_68_landmarks(face_aligned)  # 特征点和灰度open CV格式图片
            leftEye, rightEye, mouth, normal_face = area_cut(img_grey, landmarks, 5, 1.3, 1.0)  # 切割
            hog_array = img_crop_teeth_hog(mouth)
            for i in hog_array:
                d[num] = i.item()
                num += 1
        k += 1
    yt.append(1)
    xt.append(d)

    model = svm_load_model('model\\z3.model')
    p_label, p_acc, p_val = svm_predict(yt, xt, model)
    print(p_label)
    return int(p_label[0])

def getEyeBrowLevel(video_path, num):
    save_img(video_path, num, 2)
    Dir_name = "UserData\\" + str(num) + "\\2"
    file_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(Dir_name))])
    print("共" + str(file_sum) + "帧")


    num1 = 1
    num2 = int(file_sum / 4)
    num3 = int(file_sum / 2)
    num4 = int(file_sum * 3 / 4)
    num5 = file_sum

    print("获取抬眉动态特征，提取下列图片的特征：" + str(num1) + "," + str(num2) + "," + str(num3) + "," + str(num4) + "," + str(num5))
    # 一个文件夹数据
    num = 1

    image_path_list = []
    image_folder = Dir_name
    types = ('*.jpg', '*.png')

    yt = []
    xt = []

    d = {}

    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    k = 1
    for i, image_path in enumerate(image_path_list):
        if '' in image_path and ( k == num1 or k == num2 or k == num3 or k == num4 or k == num5):
            ########################接下来是图片旋转
            im = img.open(image_path)
            # ng = im.transpose(img.ROTATE_180) #旋转 180 度角。
            # ng = im.transpose(img.FLIP_LEFT_RIGHT) #左右对换。
            # ng = im.transpose(img.FLIP_TOP_BOTTOM)  # 上下对换。
            ng = im.transpose(Image.ROTATE_90)  # 旋转 90 度角。
            ng.save(image_path)
            ########################上面是图片旋转
            face_aligned = face_align(image_path)  # 对齐
            img_grey, landmarks = get_68_landmarks(face_aligned)  # 特征点和灰度open CV格式图片
            leftEye, rightEye, mouth, normal_face = area_cut(img_grey, landmarks, 5, 1.3, 1.0)  # 切割
            hog_array = img_crop_teeth_hog(leftEye)
            for i in hog_array:
                d[num] = i.item()
                num += 1
        k += 1
    yt.append(1)
    xt.append(d)

    model = svm_load_model('model\\z5.model')
    p_label, p_acc, p_val = svm_predict(yt, xt, model)
    print(p_label)
    return int(p_label[0])


'''

正常表情分类器 + 闭眼分类器  ------>  
        等于2 示齿动作路径  0 -> 5, 1 -> 6
        小于2 示齿动作路径 + 抬眉动作路径  0 -> 1,  1 -> 2, 2 -> 2 ,3 -> 3, 4 -> 4
'''

'''
###################
后台逻辑：
假设收集的数据：
        正常表情：0.jpg 闭眼：1.jpg
        示齿: video1.mp4 抬眉：video2.mp4
###################
'''
def finalTest(img1, img2, video1, video2, User_Id):

    ########################## 正常表情分类器
    level1 = getNormalFaceLevel(img1)
    # print(level1)
    # print(type(level1))

    ########################## 闭眼分类器
    level2 = getEyeClosedLevel(img2)
    # print(level2)
    # print(type(level2))

    ########################## 第二阶段的处理
    if level1 + level2 == 2:
        #User_Id = 0  # 一个用户编号，现在默认0
        level4 = getSeriousTeethLevel(video1, User_Id)
        if level4 == 0:
            print("=======================================")
            print("facial paralysis（面瘫） level is 5!")
            print("=======================================")
            return 5
        else:
            print("=======================================")
            print("facial paralysis（面瘫） level is 6!")
            print("=======================================")
            return 6
    else:
        User_Id = 0  # 一个用户编号，现在默认0
        level3 = getLightTeethLevel(video1, User_Id)

        level5 = getEyeBrowLevel(video2, User_Id)

        if level3 + level5 == 0:
            print("=======================================")
            print("facial paralysis（面瘫） level is 1!")
            print("=======================================")
            return 0
        elif level3 + level5 == 1:
            print("=======================================")
            print("facial paralysis（面瘫） level is 2!")
            print("=======================================")
            return 2
        elif level3 + level5 == 2:
            print("=======================================")
            print("facial paralysis（面瘫） level is 2!")
            print("=======================================")
            return 2
        elif level3 + level5 == 3:
            print("=======================================")
            print("facial paralysis（面瘫） level is 3!")
            print("=======================================")
            return 3
        elif level3 + level5 == 4:
            print("=======================================")
            print("facial paralysis（面瘫） level is 4!")
            print("=======================================")
            return 4

