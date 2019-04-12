# -*-coding:utf8-*-
import os
import cv2
import time
import shutil

CASCADE_FACE = '/usr//local/Cellar/opencv/2.4.13.2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'
CASCADE_LEFT_EYE = '/usr//local/Cellar/opencv/2.4.13.2/share/OpenCV/haarcascades/haarcascade_lefteye_2splits.xml'
CASCADE_RIGHT_EYE = '/usr//local/Cellar/opencv/2.4.13.2/share/OpenCV/haarcascades/haarcascade_righteye_2splits.xml'


def get_image_path_list(dir_path, *suffix):
    path_array = []

    for r, ds, fs in os.walk(dir_path):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                file_name = os.path.join(r, fn)
                path_array.append(file_name)

    return path_array


def show_image(image_name, image_data):
    cv2.imshow(image_name, image_data)


def cutout_faces(src_path, tag_path, invalid_path, *suffix):
    try:
        count = 1
        image_paths = get_image_path_list(src_path, *suffix)

        face_cascade = cv2.CascadeClassifier(CASCADE_FACE)
        left_eye_cascade = cv2.CascadeClassifier(CASCADE_LEFT_EYE)
        right_eye_cascade = cv2.CascadeClassifier(CASCADE_RIGHT_EYE)

        for image_path in image_paths:
            image = cv2.imread(image_path)
            if type(image) != str:
                faces = face_cascade.detectMultiScale(image, 1.1, 5)
                if len(faces):
                    for (x, y, w, h) in faces:
                        # 设置人脸宽度大于128像素，去除较小的人脸
                        if w >= 128 and h >= 128:
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2, 8, 0)

                            left_eye = left_eye_cascade.detectMultiScale(image, 1.1, 5)
                            for (left_eye_x, left_eye_y, left_eye_w, left_eye_h) in left_eye:
                                cv2.rectangle(image, (left_eye_x, left_eye_y),
                                              (left_eye_x + left_eye_w, left_eye_y + left_eye_h),
                                              (255, 0, 0), 2, 8, 0)

                            left_eye = right_eye_cascade.detectMultiScale(image, 1.1, 5)
                            for (right_eye_x, right_eye_y, right_eye_w, right_eye_h) in left_eye:
                                cv2.rectangle(image, (right_eye_x, right_eye_y),
                                              (right_eye_x + right_eye_w, right_eye_y + right_eye_h),
                                              (255, 0, 0), 2, 8, 0)

                            # 以时间戳和读取的排序作为文件名称
                            list_str = [str(int(time.time())), str(count)]
                            file_name = ''.join(list_str)

                            '''第一种情况，宽图像'''
                            # face_x1 = x - w / 4
                            # face_x2 = face_x1 + w + w / 2
                            # face_y1 = y
                            # face_y2 = face_y1 + h

                            '''第二种情况，正方形图像'''
                            face_x1 = int(x - w / 4)
                            face_x2 = int(face_x1 + w + w / 2)
                            face_y1 = int(y - w / 4)
                            face_y2 = int(face_y1 + w + w / 2)

                            cutout_face = image[face_y1:face_y2, face_x1:face_x2]
                            face_224_224 = cv2.resize(cutout_face, (224, 224))

                            cv2.imshow("original", image)
                            cv2.imshow("cutout", cutout_face)
                            cv2.imshow("target", face_224_224)
                            cv2.waitKey(0)

                            # f = cv2.resize(image[face_y:face_h, face_x:face_w], (face_w - face_h, face_h - face_y))
                            # show_image(image[face_y:face_h, face_x:face_w])

                            # cv2.imwrite(tag_path + os.sep + '%s.jpg' % file_name, f)
                            count += 1

                            print(image_path + "have face")
                else:
                    shutil.move(image_path, invalid_path)

    except IOError:
        print("Error")

    else:
        print('Find ' + str(count - 1) + ' faces to Destination ' + tag_path)


if __name__ == '__main__':
    invalidPath = r'/Users/xiaxing/PycharmProjects/FaceCutout/data/noface'
    sourcePath = r'/Users/xiaxing/PycharmProjects/FaceCutout/data/original'
    targetPath = r'/Users/xiaxing/PycharmProjects/FaceCutout/data/faces'

    cutout_faces(sourcePath, targetPath, invalidPath, '.jpeg', '.jpg', '.JPG', 'png', 'PNG')
