import os,math,json,random,pandas
import numpy as np
from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
loss=[299]
def Sigmoid(pp):
    return 1/(1+np.exp(-pp))

def Sigmoid_df(pp):
    return pp*(1-pp)

def ReLU(pp):
    a=pp>0
    return pp*a

def double_ReLU(pp):
    a=pp>0
    return pp*(a*2-1)

def tanh(pp):
    a=(np.exp(2*pp)-1)/(np.exp(2*pp)+1)
    if np.any(a!=a):
        if x>0:
            a=np.ones(pp.shape)
        else:
            a=np.zeros(pp.shape)
    return a

def tanh_df(pp,stay="df"):
    if stay=="df":
        return 1-pp**2
    else:
        return 1-tanh(pp)**2

def tanh_df_df(pp,stay="df"):
    return 2*(tanh(pp)**3-tanh(pp))#plase give role x do not tanh

def entropy(answer,input):
    return -answer*np.log(input)-(1-answer)*np.log(1-input)

def entropy_loss(answer,input):
    return (input-answer)/input/(1-input)

def star(in_put):
    Neurons[0].main_run(in_put)
    inputer = Neurons[0].get_ReLU()
    # 第一層
    Neurons[1].main_run(inputer)
    ship = Neurons[1].get_ReLU()
    # 第二層

    Neurons[2].main_run(ship)
    output = Neurons[2].the_op
    # 第三層
    return inputer,ship, Sigmoid(output)

class LSTM :
    def __init__(self,LSTM_time,input_num,output_num,long_num,whole_time):
        if input_num<output_num:
            print("LSTM error")
        self.ipnum=input_num
        self.short_ipnum=long_num-input_num
        self.long_num=long_num
        self.opnum=output_num
        self.LSTM_time=LSTM_time
        self.whole_time=whole_time
        def randons(c,a):
            b=np.empty((c,a))
            for x in range(a*c):
                b[int(x/a),x%a]=random.randint(-100,100)/2000
            return b
        #[w,b]
        self.Forget=[randons(long_num,long_num).reshape((1,long_num,long_num)),randons(1,long_num)]
        self.storage_s=[randons(long_num,long_num).reshape((1,long_num,long_num)),randons(1,long_num)]
        self.storage_t=[randons(long_num,long_num).reshape((1,long_num,long_num)),randons(1,long_num)]
        self.output_s=[randons(long_num,long_num).reshape((1,long_num,long_num)),randons(1,long_num)]
        self.output_t=[randons(1,long_num),randons(1,long_num)]

        self.ip=np.zeros((whole_time,self.LSTM_time,1,self.long_num))
        self.s1=np.zeros((whole_time,self.LSTM_time,self.long_num,1))
        self.s2=np.zeros((whole_time,self.LSTM_time,self.long_num,1))
        self.s3=np.zeros((whole_time,self.LSTM_time,self.long_num,1))
        self.th=np.zeros((whole_time,self.LSTM_time,self.long_num,1))
        self.opth=np.zeros((whole_time,self.LSTM_time,self.long_num,1))
        
        self.long_ip1=np.zeros((whole_time,self.LSTM_time,self.long_num,1))
        self.long_ip2=np.zeros((whole_time,self.LSTM_time,self.long_num,1))
        self.long_ip3=np.zeros((whole_time,self.LSTM_time,self.long_num,1))

        self.ipdf=np.zeros((whole_time,self.long_num,1)) 
        self.long_ipdf=np.zeros((whole_time,self.long_num,1)) 
        self.Forget_d=[np.zeros((whole_time,long_num,long_num)),np.zeros((whole_time,long_num))]
        self.storage_ds=[np.zeros((whole_time,long_num,long_num)),np.zeros((whole_time,long_num))]
        self.storage_dt=[np.zeros((whole_time,long_num,long_num)),np.zeros((whole_time,long_num))]
        self.output_ds=[np.zeros((whole_time,long_num,long_num)),np.zeros((whole_time,long_num))]
        self.output_dt=[np.zeros((whole_time,long_num)),np.zeros((whole_time,long_num))]

    def get_wb(self,data):
        self.Forget[0]=np.array(data["Forget_w"])
        self.Forget[1]=np.array(data["Forget_b"])
        self.storage_s[0]=np.array(data["storage_sw"])
        self.storage_s[1]=np.array(data["storage_sb"])
        self.storage_t[0]=np.array(data["storage_tw"])
        self.storage_t[1]=np.array(data["storage_tb"])
        self.output_s[0]=np.array(data["output_sw"])
        self.output_s[1]=np.array(data["output_sb"])
        self.output_t[0]=np.array(data["output_tw"])
        self.output_t[1]=np.array(data["output_tb"])

    def run(self,x_ip):
        #x_ip,sahep=(whole_time,lstm_time,import_num)

        #x_ip=x_ip.reshape((self.whole_time,self.LSTM_time,self.ipnum))
        short_ip=np.zeros((self.whole_time,self.short_ipnum))
        long_ip=np.zeros((self.whole_time,self.long_num,1))
        output=np.empty((self.whole_time,self.LSTM_time,self.opnum))
        for x in range(self.LSTM_time):
            #print(x_ip[:,x])
            ip=np.hstack((x_ip[:,x],short_ip)).reshape((self.whole_time,1,self.long_num))# U
            self.ip[:,x]=ip

            p=Sigmoid((ip*self.Forget[0]).sum(axis=2)+self.Forget[1]).reshape((self.whole_time,self.long_num,1))#1sm

            self.s1[:,x]=p
            self.long_ip1[:,x]=long_ip
            self.long_ip2[:,x]=long_ip*p

            p=Sigmoid((ip*self.storage_s[0]).sum(axis=2)+self.storage_s[1]).reshape((self.whole_time,self.long_num,1))#2sm
            self.s2[:,x]=p

            pp=tanh((ip*self.storage_t[0]).sum(axis=2)+self.storage_t[1]).reshape((self.whole_time,self.long_num,1))
            self.th[:,x]=pp

            self.long_ip3[:,x]=self.long_ip2[:,x]+p*pp

            p=Sigmoid((ip*self.output_s[0]).sum(axis=2)+self.output_s[1]).reshape((self.whole_time,self.long_num,1))#3sm

            self.s3[:,x]=p

            pp=tanh(long_ip.reshape((self.whole_time,1,self.long_num))*self.output_t[0]+self.output_t[1]).reshape((self.whole_time,self.long_num,1))
            self.opth[:,x]=pp
            output_Untreated=pp*p

            short_ip=output_Untreated[:,self.ipnum:,0]
            long_ip=self.long_ip3[:,x]
            output[:,x]=output_Untreated[:,:self.opnum,0]
        return output

            #                           output
            #long_input                 ^
            #==>===x=======+>=====\==============>
            #      |       |      th    ^
#tanh==th   #      |   /===x      |     |
            #      |   |   |      |     |
#sigmoid==sm#      sm  sm  th  sm>x     | this 3sm and th is dnn(2,one input one output)
            #      |   |   |   |  |     |
#short_input#==>U==/===/===/===/  \======-->short_input
            #   ^
            #   |input  #ip.shape=(input,short_input)
    
    def get_df(self,x_df):
        short_df=np.zeros((self.whole_time,self.short_ipnum))
        x_df=x_df.reshape((self.whole_time,self.LSTM_time,self.opnum))
        for y in range(self.LSTM_time):
            x=self.LSTM_time-y-1
            
            Derivative=np.hstack((x_df[:,x],np.zeros((self.whole_time,self.ipnum-self.opnum)),short_df)).reshape((self.whole_time,self.long_num,1))
            #print(Derivative.shape,self.opth[:,x].shape,self.s3[:,x].shape,self.ip[:,x].shape,self.output_dsw.shape)
            #print(self.long_ip3[:,x].shape,self.long_ipdf.shape,self.th[:,x].shape,self.s2[:,x].shape)
            self.output_ds[0]+=Derivative*self.opth[:,x]*Sigmoid_df(self.s3[:,x])*self.ip[:,x]
            self.output_ds[1]+=(Derivative*self.opth[:,x]*Sigmoid_df(self.s3[:,x]))[...,0]
            self.ipdf=(Derivative*self.opth[:,x]*Sigmoid_df(self.s3[:,x])*self.output_s[0]).sum(axis=1)

            self.output_dt[0]+=(Derivative*self.s3[:,x]*tanh_df(self.opth[:,x])*self.long_ip3[:,x])[...,0]
            self.output_dt[1]+=(Derivative*self.s3[:,x]*tanh_df(self.opth[:,x]))[...,0]
            self.long_ipdf[...,0]+=(Derivative*self.s3[:,x]*tanh_df(self.opth[:,x]))[...,0]*self.output_t[0]

            self.storage_ds[0]+=self.long_ipdf*self.th[:,x]*Sigmoid_df(self.s2[:,x])*self.ip[:,x]
            self.storage_ds[1]+=(self.long_ipdf*self.th[:,x]*Sigmoid_df(self.s2[:,x]))[...,0]
            self.ipdf+=(self.long_ipdf*self.th[:,x]*Sigmoid_df(self.s2[:,x])*self.storage_s[0]).sum(axis=1)

            self.storage_dt[0]+=self.long_ipdf*self.s2[:,x]*tanh_df(self.th[:,x])*self.ip[:,x]
            self.storage_dt[1]+=(self.long_ipdf*self.s2[:,x]*tanh_df(self.th[:,x]))[...,0]
            self.ipdf+=(self.long_ipdf*self.s2[:,x]*tanh_df(self.th[:,x])*self.storage_t[0]).sum(axis=1)

            #print(self.ip[:,x])
            self.Forget_d[0]+=self.long_ipdf*self.long_ip1[:,x]*Sigmoid_df(self.s1[:,x])*self.ip[:,x]
            self.Forget_d[1]+=(self.long_ipdf*self.long_ip1[:,x]*Sigmoid_df(self.s1[:,x]))[...,0]
            self.ipdf+=(self.long_ipdf*self.long_ip1[:,x]*Sigmoid_df(self.s1[:,x])*self.Forget[0]).sum(axis=1)

            self.long_ipdf*=self.s1[:,x]
            short_df=self.ipdf[:,self.ipnum:]

    def chind_wb(self):
        #print(self.Forget_dw.sum(axis=0)[0])#####~~~~~~~~~~~

        self.Forget[0]-=self.Forget_d[0].sum(axis=0)/self.whole_time
        self.Forget[1]-=self.Forget_d[1].sum(axis=0)/self.whole_time
        self.storage_s[0]-=self.storage_ds[0].sum(axis=0)/self.whole_time
        self.storage_s[1]-=self.storage_ds[1].sum(axis=0)/self.whole_time
        self.storage_t[0]-=self.storage_dt[0].sum(axis=0)/self.whole_time
        self.storage_t[1]-=self.storage_dt[1].sum(axis=0)/self.whole_time
        self.output_s[0]-=self.output_ds[0].sum(axis=0)/self.whole_time
        self.output_s[1]-=self.output_ds[1].sum(axis=0)/self.whole_time
        self.output_t[0]-=self.output_dt[0].sum(axis=0)/self.whole_time
        self.output_t[1]-=self.output_dt[1].sum(axis=0)/self.whole_time

        self.Forget_d[0]*=0
        self.Forget_d[1]*=0
        self.storage_ds[0]*=0
        self.storage_ds[1]*=0
        self.storage_dt[0]*=0
        self.storage_dt[1]*=0
        self.output_ds[0]*=0
        self.output_ds[1]*=0
        self.output_dt[0]*=0
        self.output_dt[1]*=0
        self.long_ipdf*=0

    def return_data(self):
        return {"Forget_w":self.Forget[0].tolist(),"Forget_b":self.Forget[1].tolist(),
                "storage_sw":self.storage_s[0].tolist(),"storage_sb":self.storage_s[1].tolist(),
                "storage_tw":self.storage_t[0].tolist(),"storage_tb":self.storage_t[1].tolist(),
                "output_sw":self.output_s[0].tolist(),"output_sb":self.output_s[1].tolist(),
                "output_tw":self.output_t[0].tolist(),"output_tb":self.output_t[1].tolist()}



class gerdre:
    def __init__(self,data_innum,data_outnum,type,time):
        self.type=type
        self.time=time
        self.w=np.zeros(data_innum*data_outnum)
        self.b=np.zeros((1,data_outnum))
        self.dw=np.zeros((time,data_outnum,data_innum))
        self.db=np.zeros((time,data_outnum))
        self.a=np.zeros((time,data_innum,1))
        self.o_zdf=np.ones((time,data_outnum,1))
        for x in range(data_innum*data_outnum):
            self.w[x]=random.randint(-100,100)/600
        for x in range(data_outnum):
            self.b[0,x]=random.randint(-100,100)/100
        self.w = self.w.reshape((1,data_outnum,data_innum))

    def get_wb(self,data):
        self.w=np.array(data["w"])
        self.b=np.array(data["b"])

    def run(self,ip):
        self.a=ip.reshape((ip.shape[0],1,ip.shape[1]))#(10,60) #self.w (1,32,60)

        #the_op=np.zeros((self.time,self.w.shape[1],self.w.shape[2]))

        p = self.a * self.w
        the_op= p.sum(axis=2)+self.b
        #print(self.a.shape,self.w.shape,self.b.shape,p.shape,the_op.shape)
        if self.type=="Sigmoid":
            self.output=Sigmoid(the_op)
            p=Sigmoid_df(self.output)
            self.o_zdf=p.reshape(self.o_zdf.shape)
        elif self.type=="ReLU":
            self.output=ReLU(the_op)
            p=the_op>0
            self.o_zdf=p.reshape(self.o_zdf.shape)
        elif self.type=="double_ReLU":
            self.output=double_ReLU(the_op)
            p=(the_op>0)*2-1
            self.o_zdf=p.reshape(self.o_zdf.shape)
        elif self.type=="tanh_df_and_ReLU_13":
            self.output=ReLU(the_op)
            a=int(the_op.shape[1]/4)
            self.output[...,:a]=tanh_df(the_op[...,:a],stay="model")
            p=the_op>0
            p[:,:a]=tanh_df_df(the_op[:,:a])
            self.o_zdf=p.reshape(self.o_zdf.shape)
        elif self.type=="NULL":
            self.output=the_op
            #there wont to get o_zdf becouse the df is 1
        else :
            print("type is error")
        
        return self.output
        

    def get_df(self,Derivative):
        #Derivative (time,data_outnum,1)
        #o_zdf (time,data_outnum,1)
        #a (time,data_innum)
        #print(Derivative.shape,self.o_zdf.shape,self.a.shape,self.w.shape)
        Derivative=Derivative.reshape(np.hstack((Derivative.shape,1)))
        self.dw=Derivative*self.o_zdf*self.a
        self.da=Derivative*self.o_zdf*self.w
        self.db=Derivative*self.o_zdf
        self.db=self.db.reshape((self.db.shape[0],self.db.shape[1]))
        return self.da.sum(axis=1)


    def chine_wb(self):
        self.w -= self.dw.sum(axis=0)/self.time  # 2d
        self.b -= self.db.sum(axis=0)/self.time  # 2d
        ##改變權重

    def return_data(self):
        return {"w":self.w.tolist(),"b":self.b.tolist()}



#######################################################################################################
data = keras.datasets.mnist
(x_train,y_train), (x_test,y_test) = data.load_data()
x_train,x_test=x_train/255,x_test/255

class_name = [0,1,2,3,4,5,6,7,8,9]

data = {"short_ipnum":36,"long_ipnum":64,"LSTM_ipnum":28,"LSTM_opnum":10,"ipnum":280,"ship1num":256,
        "ship2num":128,"output":10,"learn":0.25,"skip":1,"run_time":10,"LSTM_time":28}  
        # 数据 just dnn learn 0.5 exact_learn 0.05 run_time 100

#print(x_train.reshape((x_train[0],x_train[1]*x_train[2])))




Neurons = [gerdre(data["ipnum"], data["ship1num"],"ReLU",data["run_time"]),
           gerdre(data["ship1num"], data["ship2num"],"ReLU",data["run_time"]),
           gerdre(data["ship2num"], data["output"],"Sigmoid",data["run_time"])]
# 神经元物件

fg=LSTM(data["LSTM_time"],data["LSTM_ipnum"],data["LSTM_opnum"],data["long_ipnum"],data["run_time"])
"""
with open("data.json",mode="r") as file :
    a=json.load(file)
    print(a["point"])
    Neurons[0].get_wb(a["dnn0"])
    Neurons[1].get_wb(a["dnn1"])
    Neurons[2].get_wb(a["dnn2"])
    fg.get_wb(a["LSTM"])
"""
#x_train=np.zeros((60000,28,28))
#x_train=np.ones((60000,28,28))
#test


y,point, loss_sum, a = 1, 0, 0, 0
stay=0
time = tqdm(total=60000, desc="hi")
skip=data["skip"]*10
history=np.zeros(y_train.shape[0])
answer=np.zeros((data["run_time"],data["output"]))
output=np.empty((data["run_time"],data["ipnum"]))
#######################################################################################################

for y in range(int(x_train.shape[0]/data["run_time"]/2)):
    role_y=y*data["run_time"]
    

    for x in range(data["run_time"]):
        answer[x,y_train[role_y+x]]=1
        answer[x,y_train[role_y-data["run_time"]+x]]=0
        #answer=np.ones((280))

    
    output=fg.run(x_train[role_y:role_y+data["run_time"]]).reshape((data["run_time"],data["ipnum"]))
    
    dnn_output0=Neurons[0].run(output)

    dnn_output1=Neurons[1].run(dnn_output0)

    dnn_output2=Neurons[2].run(dnn_output1)

    loss=(answer-dnn_output2)**2
    learn=data["learn"]

    Derivative=2*(dnn_output2-answer)

    nn_df2=Neurons[2].get_df(Derivative*learn)
    Neurons[2].chine_wb()

    nn_df1=Neurons[1].get_df(nn_df2)
    Neurons[1].chine_wb()

    nn_df0=Neurons[0].get_df(nn_df1)
    Neurons[0].chine_wb()
    if stay==0:
        fg.get_df(nn_df0*200)
    else:
        fg.get_df(nn_df0)
    fg.chind_wb()

    a+=loss.sum()
    
    if y%skip==0:
        print(y,role_y,a/skip/data["run_time"],point/skip/data["run_time"],"learn:",learn,stay)
        if stay==0 and point/skip/data["run_time"]>0.82:
            stay=1
        if stay==1 and point/skip/data["run_time"]>0.85:
            stay=2
            data["learn"]/=10
        a,point=0,0

    for x in range(data["run_time"]):
        if np.argmax(dnn_output2[x])==np.argmax(answer[x]):
            point+=1

print("OK")
point=0
history=np.empty((x_test.shape[0]))
for y in range(int(x_test.shape[0]/data["run_time"])):
    role_y=y*data["run_time"]
        
    for x in range(data["run_time"]):
        answer[x,y_train[role_y+x]]=1
        answer[x,y_train[role_y-data["run_time"]+x]]=0
    
    output=fg.run(x_test[role_y:role_y+data["run_time"]]).reshape((data["run_time"],data["ipnum"]))
    dnn_output0=Neurons[0].run(output)
    dnn_output1=Neurons[1].run(dnn_output0)
    dnn_output2=Neurons[2].run(dnn_output1)

    if y%skip==0:
        print(y/int(x_test.shape[0]/data["run_time"]))

    for x in range(data["run_time"]):
        history[role_y+x]=np.argmax(dnn_output2[x])
        if np.argmax(dnn_output2[x])==np.argmax(answer[x]):
            point+=1

print(point/int(x_test.shape[0]/data["run_time"]),"yo")

##write
with open("data5.json",mode="w") as file:
    data={"point":point/int(x_test.shape[0]/data["run_time"]),"data":[Neurons[0].type,Neurons[1].type,Neurons[2].type],
            "LSTM":fg.return_data(),"dnn0":Neurons[0].return_data(),
            "dnn1":Neurons[1].return_data(),"dnn2":Neurons[2].return_data()}
    json.dump(data,file)

"""for x in range(10):
    a=random.randint(0,800)
    plt.imshow(x_test[a+x],cmap=plt.cm.binary)
    plt.title("預測"+str(history[a+x]))
    plt.show()
"""
