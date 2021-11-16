import os.path
import random

seed = 'qwerty'

name_list = []
# rename
# 1
name = ''.join([i for i in random.sample(seed, 6)])
print(name)
# 2
for i in range(7):
    choice = random.choice(seed)
    name_list.append(choice)
print(''.join(name_list))
# 3
name = random.randint(100, 10000000)
print(name)
# 获取文件名
path = '/mike/teacher/logo.jpg'
print(os.path.basename(path))
# 文件名分割
file_name = 'looooook.jpg'
print(file_name.split('.')[0])
