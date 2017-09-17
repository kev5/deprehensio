import os
# import matplotlib.pyplot as plt
# import cv2
# path = '/Users/apple/Downloads/PIC'
# dir_lists = []
# file_lists = []
#
# files = os.listdir(path)
# for f in files:
#     if os.path.isdir(path+'/'+f):
#         if f[0]!='.':
#             dir_lists.append(path+'/'+f)
#
# for d in dir_lists:
#     files = os.listdir(d)
#     for f in files:
#         if os.path.isfile(d+'/'+f):
#             if (f[0]!='.')&(f[-1]=='g'):
#                 file_lists.append(d+'/'+f)

# print file_lists
# pic = cv2.imread(file_lists[0])
# pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
# plt.imshow(pic)
# plt.show()

def get_files(path):

    dir_lists = []
    file_lists = []

    files = os.listdir(path)
    for f in files:
        if os.path.isdir(path + '/' + f):
            if f[0] != '.':
                dir_lists.append(path + '/' + f)

    for d in dir_lists:
        files = os.listdir(d)
        for f in files:
            if os.path.isfile(d + '/' + f):
                if (f[0] != '.') & (f[-1] == 'g'):
                    file_lists.append(d + '/' + f)

    return file_lists