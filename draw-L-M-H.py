import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator
def read_tensorboard_data(tensorboard_path, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val


L_dir = '1109-1523-Deltacom-F50-P5'
M_dir = '1109-1532-Deltacom-F100-P5'
H_dir = '1108-2249-Deltacom-F150-P5'



# 读取delay数据
d_l = np.load('logs/'+L_dir+'/delay_res.npy')
d_m = np.load('logs/'+M_dir+'/delay_res.npy')
d_h = np.load('logs/'+H_dir+'/delay_res.npy')

d_l = d_l / np.expand_dims(d_l[:,0],axis=-1)
d_m = d_m / np.expand_dims(d_m[:,0],axis=-1)
d_h = d_h / np.expand_dims(d_h[:,0],axis=-1)

d_l = np.average(d_l,axis=0)
d_m = np.average(d_m,axis=0)
d_h = np.average(d_h,axis=0)

# 读取时间数据
t_l = np.load('logs/'+L_dir+'/time_res.npy')
t_m = np.load('logs/'+M_dir+'/time_res.npy')
t_h = np.load('logs/'+H_dir+'/time_res.npy')

t_l = np.average(t_l,axis=0)
t_l[-1]=0
t_m = np.average(t_m,axis=0)
t_m[-1]=0
t_h = np.average(t_h,axis=0)
t_h[-1]=0

print(t_l,t_m,t_h)
x= ['Opt','Pred','Seq','SP']
idx = np.arange(len(x))
print(idx)

acc_l = read_tensorboard_data('logs/'+L_dir,'accuracy')
acc_m = read_tensorboard_data('logs/'+M_dir,'accuracy')
acc_h = read_tensorboard_data('logs/'+H_dir,'accuracy')

acc_ax1 = plt.subplot(331)
accu_l= [j.value for j in acc_l]
accu_l = pd.DataFrame(accu_l)
acc_ax1.plot(accu_l,'cornsilk',accu_l.rolling(20).mean(),'coral', label='accuracy')
plt.ylabel('training accuracy')

acc_ax2 = plt.subplot(332,sharey=acc_ax1)
accu_m= [j.value for j in acc_m]
accu_m = pd.DataFrame(accu_m)
acc_ax2.plot(accu_m,'cornsilk',accu_m.rolling(20).mean(),'coral', label='accuracy')

acc_ax3 = plt.subplot(333,sharey=acc_ax1)
accu_h= [j.value for j in acc_h]
accu_h = pd.DataFrame(accu_h)
acc_ax3.plot(accu_h,'cornsilk',accu_h.rolling(20).mean(),'coral', label='accuracy')


t_ax1 = plt.subplot(334)
t_ax1.bar(idx,t_l,tick_label=x,color=['r', 'g', 'b','y'])
plt.ylim((0,2))
plt.ylabel('processing time')
t_ax2 = plt.subplot(335,sharey=t_ax1)
t_ax2.bar(idx,t_m,tick_label=x,color=['r', 'g', 'b','y'])
t_ax3 = plt.subplot(336,sharey=t_ax1)
t_ax3.bar(idx,t_h,tick_label=x,color=['r', 'g', 'b','y'])


d_ax1 = plt.subplot(337)
d_ax1.bar(idx,d_l,tick_label=x,color=['r', 'g', 'b','y'])
plt.ylim((0,2))
plt.ylabel('delay ratio')
d_ax2 = plt.subplot(338,sharey=d_ax1)
d_ax2.bar(idx,d_m,tick_label=x,color=['r', 'g', 'b','y'])
d_ax3 = plt.subplot(339,sharey=d_ax1)
d_ax3.bar(idx,d_h,tick_label=x,color=['r', 'g', 'b','y'])
plt.xlabel(['light','medium','heavy'])
plt.show()




# d_all = np.reshape([d_l,d_m,d_h],[3,4])
# print(d_all)
# d_all = d_all.T
# print(d_all)
#
# x_label = ['Light','Medium','Heavy']
#
# bar_width = 0.2  # 条形宽度
# index_opt = np.arange(len(x_label))
# index_pred = index_opt + bar_width
# index_seq = index_pred + bar_width
# index_sp = index_seq + bar_width
#
#
# plt.bar(index_opt, d_all[0],  width=bar_width, label='Opt')
# plt.bar(index_pred, d_all[1], width=bar_width, label='Pred')
# plt.bar(index_seq, d_all[2], width=bar_width, label='Seq')
# plt.bar(index_sp, d_all[3], width=bar_width, label='SP')
#
# plt.legend()
# plt.xticks(index_pred+bar_width/2, x_label)
# # my_y_ticks = np.arange(0.8, 1.6, 0.2)
# # plt.yticks(my_y_ticks)
# plt.ylim((0.8,2))
# plt.ylabel('Delay Ratio')
# plt.show()