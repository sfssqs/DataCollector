# -*-coding:utf8-*-
import os
import tensorflow as tf
from PIL import Image

cwd = '/Users/xiaxing/Desktop/DmsData/video_test/actions/'
suffix = {'.jpeg'}

classes = {'normal', 'calling'}

# 要生成的文件
writer = tf.python_io.TFRecordWriter("actions.tfrecords")
for index, name in enumerate(classes):
    class_path = cwd + name + '/'

    for img_name in os.listdir(class_path):
        # 每一个图片的路径
        img_path = class_path + img_name
        if os.path.splitext(img_path)[1] in suffix:
            img = Image.open(img_path)
            img = img.resize((224, 224))
            # 将图片转化为二进制格式
            img_raw = img.tobytes()

            # example对象对label和image数据进行封装
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }))

            # 序列化为字符串
            writer.write(example.SerializeToString())
writer.close()


# 读入actions.tfrecords
def read_and_decode(filename):
    # 生成一个queue队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    # 将image数据和label取出来
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # reshape为128*128的3通道图片
    img = tf.reshape(img, [224, 224, 3])
    # 在流中抛出img张量
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    # 在流中抛出label张量
    label = tf.cast(features['label'], tf.int32)
    return img, label


# 读入流中
filename_queue = tf.train.string_input_producer(["actions.tfrecords"])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

# 返回文件名和文件
# 取出包含image和label的feature对象
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                   })

image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [224, 224, 3])
label = tf.cast(features['label'], tf.int32)
label = tf.one_hot(label, 2, 1, 0)

# 开始一个会话
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(2000):
        # 在会话中取出image和label
        example, l = sess.run([image, label])
        # 这里Image是之前提到的
        img = Image.fromarray(example, 'RGB')
        # 存下图片
        tmp_path = cwd + 'temp/'
        img.save(tmp_path + str(i) + '_''Label_' + str(l) + '.jpg')
        print(example, l)
    coord.request_stop()
