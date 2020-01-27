import numpy as np
file = open('points3D.txt')
pos = []
index = []
for line in file:
    line_split = line.split()
    x = float(line_split[1])
    y = float(line_split[2])
    z = float(line_split[3])
    pos.append(np.array(([x, y, z]), dtype=np.double))
    index.append(int(line_split[0]))
file.close()

# read keypoints
data = np.loadtxt('images.txt')
data = data.reshape(-1, 1, 3)

kpQ = []
PointIndex = []
for i in range(len(data)):
    if data[i][0][2] != -1:
        kpQ.append((data[i][0][0], data[i][0][1]))
        PointIndex.append(data[i][0][2])

# get 3d pos of keypoints
kpQ_3d = []
for p in PointIndex:
    kpQ_3d .append(pos[index.index(p)])

# print(kpQ_3d[:5])
# print(pos[0])
# print(type(pos[0]))
# print(pos[index.index(183057)])

file = open('superData.txt', 'w')
for i in range(len(kpQ)):
    all_str = str(kpQ[i][0]) + ' ' + str(kpQ[i][1]) + ' ' + str(kpQ_3d[i]
                                                                [0]) + ' ' + str(kpQ_3d[i][1]) + ' ' + str(kpQ_3d[i][2])
    file.write(all_str)
    file.write('\n')
file.close()  # 关闭文件
