import os
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

SUFFIX = '.jpeg', '.jpg', '.JPG', 'png', 'PNG'
IMAGE_WITH = 28
IMAGE_HEIGHT = 28


# convert int value to byte array, bit-endian
def int_to_byte(value):
    src = bytearray(4)
    src[0] = ((value >> 24) & 0xFF)
    src[1] = ((value >> 16) & 0xFF)
    src[2] = ((value >> 8) & 0xFF)
    src[3] = (value & 0xFF)
    return src


# write magic number to image ubyte header, 32bit
def write_image_magic_number(out):
    magic = bytearray(4)
    magic[0] = 0x00
    magic[1] = 0x00
    magic[2] = 0x08
    magic[3] = 0x03
    out.write(magic)


# write magic number to label ubyte header, 32bit
def write_label_magic_number(out):
    magic = bytearray(4)
    magic[0] = 0x00
    magic[1] = 0x00
    magic[2] = 0x08
    magic[3] = 0x01
    out.write(magic)


# write data length to file header, 32bit
def write_data_number(out, number):
    len_byte = int_to_byte(number)
    out.write(len_byte)


# convert input image to array, 128 * 128, Gray
def convert_image_to_array(src_path):
    out_img = Image.open(src_path).convert('L').resize((IMAGE_WITH, IMAGE_HEIGHT))
    out_img_arr = np.array(out_img)
    return out_img_arr


# filter images file from path
def get_image_path_list(dir_path):
    path_array = []
    for r, ds, fs in os.walk(dir_path):
        for fn in fs:
            if os.path.splitext(fn)[1] in SUFFIX:
                file_name = os.path.join(r, fn)
                path_array.append(file_name)

    return path_array


# show image array
def show_image_array(img_array):
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, )
    img = img_array.reshape(IMAGE_WITH, IMAGE_HEIGHT)
    ax.imshow(img, cmap='Greys', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


# generate image ubyte file
def generate_image_ubyte(out_path, out_file_name):
    out_file = os.path.join(out_path, out_file_name)
    out = open(out_file, 'wb+')
    # image magic number
    write_image_magic_number(out)
    # data size
    write_data_number(out, 10000)
    # image width
    write_data_number(out, IMAGE_WITH)
    write_data_number(out, IMAGE_HEIGHT)

    return out


# generate label ubyte file
def generate_label_ubyte(out_path, out_file_name):
    out_file = os.path.join(out_path, out_file_name)
    out = open(out_file, 'wb+')
    # label magic number
    write_label_magic_number(out)
    # data size
    write_data_number(out, 10000)
    return out


def main():
    print('this message is from main function')
    root_dir = '/Users/xiaxing/Desktop/DmsData/video_test/actions/'

    count = 0

    classess = []
    for path in os.listdir(root_dir):
        if path != ".DS_Store":
            classess.append(path)
    print classess

    out_image_ubyte = generate_image_ubyte('./', 'train_image.byte')
    out_label_ubyte = generate_label_ubyte('./', 'train_label.byte')

    for class_item in classess:
        image_dir = os.path.join(root_dir, class_item)
        print image_dir
        image_list = get_image_path_list(image_dir)

        for img_path in image_list:
            print (img_path)
            out_img_array = convert_image_to_array(img_path)
            out_image_ubyte.write(out_img_array)

            label_byte = bytearray(1)
            if class_item == classess[0]:
                label_byte[0] = 0x0
            else:
                label_byte[0] = 0x1
            out_label_ubyte.write(label_byte)
            count = count + 1

    print ('total image ' + str(count))
    out_image_ubyte.seek(4)
    out_image_ubyte.write(int_to_byte(count))
    out_image_ubyte.flush()
    out_image_ubyte.close()

    out_label_ubyte.seek(4)
    out_label_ubyte.write(int_to_byte(count))
    out_label_ubyte.flush()
    out_label_ubyte.close()


if __name__ == '__main__':
    main()
