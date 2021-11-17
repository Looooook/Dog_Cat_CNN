import numpy as np
from PIL import Image
import os
import torch


class Image_to_Numpy(object):
    def __init__(self, file_path, output_path, dog_types):
        self.file_path = file_path
        self.dog_types = dog_types
        self.output_path = output_path

    def image_rename(self):
        type_counter = 0
        for type in self.dog_types:
            instance_counter = 0
            for dog_instance in os.listdir(self.file_path + type):
                os.rename(self.file_path + type + '/' + dog_instance,
                          self.file_path + type + '/' + str(type_counter) + '_' + str(instance_counter) +
                          dog_instance.split('_')[-1] + '.jpg')
                instance_counter += 1
            type_counter += 1

    def image_resize(self, width=100, height=100):
        for type in self.dog_types:
            for dog_instance in os.listdir(self.file_path + type):
                img = Image.open(self.file_path + type + '/' + dog_instance)
                img_RGB = img.convert('RGB')
                resized_image = img_RGB.resize((width, height), Image.BILINEAR)
                resized_image.save(os.path.join(self.output_path, dog_instance))

    def imgs_to_np(self):
        list_images = []
        list_labels = []
        for dog_instance in os.listdir(self.output_path):
            img = Image.open(self.output_path + dog_instance)
            img = np.array(img)
            list_images.append(img)
            list_labels.append(int(dog_instance.split('_')[0]))
        # list_images此时还是列表
        np_images = np.array(list_images).transpose((0, 3, 1, 2))
        np_labels = np.array(list_labels)
        # print(np_images.shape)
        # / 255.0
        return torch.from_numpy(np_images).float() / 255, torch.from_numpy(np_labels).long()


if __name__ == '__main__':
    dog_types = ['博美犬', '哈士奇', '拉布拉多', '比格犬']
    image_to_numpy = Image_to_Numpy(file_path='./DogRaw/', output_path='./train_imgs/', dog_types=dog_types)
    # image_to_numpy.image_rename()
    # image_to_numpy.image_resize()
    np_image, np_labels = image_to_numpy.imgs_to_np()
    print(np_image.shape, np_labels)
