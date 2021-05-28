import numpy as np
import random
from tensorflow import keras
from tqdm import tqdm
import mathematics.Algebra as al
import matplotlib.pyplot as plt

def Sigmoid(pp):
    return 1/al.plus(1,np.exp(-pp))


def star(in_put):
    Neurons[0].chine_ind(in_put)
    inputer = Neurons[0].the_op
    # 第一層
    Neurons[1].chine_ind(inputer)
    ship = Neurons[1].get_ReLU()
    # 第二層

    Neurons[2].chine_ind(ship)
    output = Neurons[2].the_op
    # 第三層
    return ship, output


class gerdre:
    def __init__(self, data_in, data_out):
        self.w=np.zeros(data_in*data_out)
        self.b=np.zeros(data_out)
        for x in range(data_in*data_out):
            self.w[x]=random.randint(-1,1)/2
        for x in range(data_out):
            self.b[x]=random.randint(-1,1)/2
        if data_in!=1:
            self.w = self.w.reshape([data_out,data_in])
    
    def chine_ind(self, ind):
        self.a = ind
        the_op = self.a * self.w  # (the_output)

        if the_op.ndim == 1:
            self.the_op = the_op + self.b
        else:
            self.the_op = the_op.sum(axis=1) + self.b

        ##改變輸入值

    def chine_wb(self, w, b):
        self.w -= w  # 2d
        self.b -= b  # 2d

        ##改變權重

    def get_ReLU(self):
        self.pp = self.the_op
        for x in range(self.the_op.shape[0]):
            if self.the_op[x] < 0:
                self.the_op[x] = 0
        return self.the_op


#######################################################################################################
data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_name = [0,1,2,3,4,5,6,7,8,9]

data = {"iprnum": 784, "shipnum":256, "output": 10, "learn": 0.05, "skip": 20}  # 数据

Neurons = [gerdre(1, data["iprnum"]),
           gerdre(data["iprnum"], data["shipnum"]),
           gerdre(data["shipnum"], data["output"])]
# 神经元物件

Neurons[0].chine_ind(train_images[0].ravel())
inputer = Neurons[0].the_op
# 第一層

Neurons[1].chine_ind(inputer)
ship = Neurons[1].get_ReLU()
# 第二層

Neurons[2].chine_ind(ship)
output = Neurons[2].the_op
# 第三層
# 神經元初始化


c_adf = [np.zeros([data["shipnum"], data["iprnum"]]),
         np.zeros([data["output"], data["shipnum"]])]

c_wdf = [np.zeros([data["skip"], data["iprnum"]]),
         np.zeros([data["skip"], data["shipnum"], data["iprnum"]]),
         np.zeros([data["skip"], data["output"], data["shipnum"]])]

c_bdf = [np.zeros([data["skip"], data["iprnum"]]),
         np.zeros([data["skip"], data["shipnum"]]),
         np.zeros([data["skip"], data["output"]])]

o_zdf = [np.zeros([data["iprnum"], 1]),
         np.zeros([data["shipnum"], 1]),
         np.zeros([data["output"], 1])]
c_odf = [np.zeros([data["iprnum"]]),
         np.zeros([data["shipnum"], 1]),
         np.zeros([data["output"], 1])]
# 導數初始化
y, n, loss_sum, a = 1, 0, 0, 0
time = tqdm(total=60000, desc="hi")

#######################################################################################################
while y <= 60000:

    in_put = train_images[y % 60000]
    in_put = in_put.ravel()/255  # 好像不用
    answer = np.zeros(data["output"])
    answer[train_labels[y%60000]]=1

    ship, output = star(in_put)
    soutput = Sigmoid(output)
    loss = -al.plus(answer*np.log(soutput),al.plus(1,-answer)*np.log(1-soutput))
    # 損失

    loss_sum += loss.sum()
    skip=al.Mply(data["skip"],10)
    if y % skip == 0:
        time.update(skip)
        print("\n\r", a / skip, loss_sum/skip, end="", flush=True)

        loss_sum, a = 0, 0

    c_odf[2][..., 0]=al.plus(soutput, -answer)
    # 微分第一次

    o_zdf[2][..., 0]=np.ones([data["output"]])
    # 微分第二次//這裡用的是Sigmoid和交叉

    z_wdf = Neurons[2].a  # 1d
    z_adf = Neurons[2].w  # 1d
    # 微分第三次
    ############################################################

    c_wdf[2][y % data["skip"]] = c_odf[2] * o_zdf[2] * z_wdf
    c_adf[1] = c_odf[2] * o_zdf[2] * z_adf
    p = c_odf[2] * o_zdf[2]  # (z_bdf)
    c_bdf[2][y % data["skip"]] = p.reshape(p.shape[0])
    # 輸出總微分

    if y % data["skip"] == 0:
        c_wdf[2] /= data["skip"]
        c_bdf[2] /= data["skip"]
        Neurons[2].chine_wb(c_wdf[2].sum(axis=0) * data["learn"],
                            c_bdf[2].sum(axis=0) * data["learn"])

    # 改變權重和偏差

    
    ###########第二層#########################################
    p = c_adf[1].sum(axis=0)  ##直向加總
    c_odf[1] = p.reshape([p.shape[0], 1])
    # 微分第一次
    for x in range(data["shipnum"]):
        if ship[x] == 0:
            o_zdf[1][x, 0] = 0  ###~~~~~~~~~~~
        else:
            o_zdf[1][x, 0] = 1
        # 微分第二次
    z_wdf = Neurons[1].a
    z_adf = Neurons[1].w
    # 微分第三次
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    c_wdf[1][y % data["skip"]] = c_odf[1] * o_zdf[1] * z_wdf
    c_adf[0] = c_odf[1] * o_zdf[1] * z_adf
    p = c_odf[1] * o_zdf[1]  # (z_bdf)
    c_bdf[1][y % data["skip"]] = p.reshape(p.shape[0])
    # 輸出總微分

    if y % data["skip"] == 0:
        c_wdf[1] /= data["skip"]
        c_bdf[1] /= data["skip"]
        Neurons[1].chine_wb(c_wdf[1].sum(axis=0) * data["learn"],
                            c_bdf[1].sum(axis=0) * data["learn"])
    # 改變權重和偏差


    ###########第三層###################################################

    c_odf[0] = c_adf[0].sum(axis=0)  ##直向加總
    # 微分第一次

    z_wdf = Neurons[0].a
    # z_adf = Neurons[0].w
    # 微分第二次

    c_wdf[0][y % data["skip"]] = c_odf[0] * z_wdf

    c_bdf[0][y % data["skip"]] = c_odf[0]
    # 輸出總微分

    if y % data["skip"] == 0:
        c_wdf[0] /= data["skip"]
        c_bdf[0] /= data["skip"]
        Neurons[0].chine_wb(c_wdf[0].sum(axis=0) * data["learn"],
                            c_bdf[0].sum(axis=0) * data["learn"])
    # 改變權重和偏差

    if np.argmax(soutput) == np.argmax(answer):
        a += 1
    y += 1

answer=np.zeros(10)
y,a,b=0,0,0

while y<1000:

    in_put=test_images[y]
    in_put=in_put.ravel()/255
    answer = np.zeros(data["output"])
    answer[test_labels[y]]=1

    ship, output = star(in_put)
    soutput = Sigmoid(output)
    #輸出第三層
    if np.argmax(soutput)==np.argmax(answer):
        a+=1
    y+=1
    print(y)
print(a/y)

for x in range(30):
    a=random.randint(0,10000)

    in_put = test_images[a]
    ship, output=star(in_put.ravel()/255)
    soutput = Sigmoid(output)
    
    plt.grid(False)
    plt.imshow(test_images[a],cmap=plt.cm.binary)
    plt.title("預測"+str(class_name[np.argmax(soutput)]))
    plt.show()
