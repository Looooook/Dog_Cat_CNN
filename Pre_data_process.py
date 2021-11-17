import os
import numpy as np
from PIL import Image


def file_rename(dog_type, file_path):
    """
    :param dog_type:种类
    :param file_path:文件路径
    :return:
    """
    type_counter = 0  # 种类计数
    for type in dog_type:
        # type是某种狗类名
        file_counter = 0
        sub_folder = os.listdir(file_path + type)
        for sub_class in sub_folder:
            file_counter += 1
            # print(file_counter)
            # print(type_counter)
            # print(sub_class)
            os.rename(file_path + type + '/' + sub_class,
                      file_path + type + '/' + str(type_counter) + '_' + str(file_counter) + sub_class.split('.')[
                          0].split('_')[-1] + '.jpg')
        type_counter += 1


def file_resize(output_folder, dog_type, file_path, width=100, height=100):
    for type in dog_type:
        files = os.listdir(file_path + type)
        for i in files:
            img_open = Image.open(file_path + type + '/' + i)
            conv_RGB = img_open.convert('RGB')
            resized_img = conv_RGB.resize((width, height), Image.BILINEAR)
            # print(i, os.path.basename(i))
            resized_img.save(os.path.join(output_folder, os.path.basename(i)) + '.jpg')


# 将图片转为数组
def read_image(train_folder, filename):
    img = Image.open(train_folder + filename)
    return np.array(img)


def dataset(train_folder):
    train_list_img = []
    train_list_label = []
    for file_n in os.listdir(train_folder):
        file_img_to_array = read_image(train_folder, file_n)
        train_list_img.append(file_img_to_array)
        train_list_label.append(int(file_n.split('_')[0]))
    train_list_img = np.array(train_list_img)
    train_list_label = np.array(train_list_img)
    print(train_list_img.shape)
    # print(train_list_img)


if __name__ == '__main__':
    dog_type = ['比格犬', '拉布拉多', '博美犬', '哈士奇']
    # file_rename(dog_type=dog_type, file_path='./DogRaw/')
    # file_resize(output_folder='./train_img/', file_path='./DogRaw/', dog_type=dog_type)
    dataset('./train_img/')
