import torch
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.manifold import TSNE 
from sklearn.cluster import KMeans 
import pdb

def cosine_schedule_warmup(total_step, value, final_value=0, warmup_step=0, warmup_value=0):
    if warmup_step > 0:
        warmup_schedule = np.linspace(warmup_value, value, warmup_step+2)[1:-1]
    else:
        warmup_schedule = np.array([])
    steps = np.arange(total_step - warmup_step)
    schedule = final_value + 0.5 * (value-final_value) * (1+np.cos(np.pi * steps / len(steps)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_step
    return schedule

class build_cosine_scheduler:
    def __init__(self, optimizer, lr, total_step, lr_warmup_step=0):
        init_lr = 0
        final_lr = lr * 1e-3
        self.lrs = cosine_schedule_warmup(total_step, lr, final_lr, lr_warmup_step, init_lr)
        self.optimizer = optimizer

    def step(self,idx):
        lr = self.lrs[idx]
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"]= lr
        self.lr=lr

class build_bicosine_scheduler:
    def __init__(self, optimizer, lr, total_step, lr_warmup_step=0):
        lr_promt = lr[0]
        lr_conv = lr[1]
        init_lr=0
        final_lr_promt = lr_promt * 1e-3
        final_lr_conv = lr_conv * 1e-3
        self.lrs_prompt = cosine_schedule_warmup(total_step, lr_promt, final_lr_promt, lr_warmup_step, init_lr)
        self.lrs_conv = cosine_schedule_warmup(total_step, lr_conv, final_lr_conv, lr_warmup_step, init_lr)
        self.optimizer = optimizer

    def step(self,idx):
        lr_promt = self.lrs_prompt[idx]
        lr_conv = self.lrs_conv[idx]
        for i, param_group in enumerate(self.optimizer.param_groups):
            # pdb.set_trace()
            if i==0:
                param_group["lr"] = lr_conv
            else:
                param_group["lr"] = lr_promt 
        self.lr_conv = lr_conv
        self.lr_prompt = lr_promt

def plot_tsne(features, labels, id):
    """
    features:(N*m)N*m大小特征,其中N代表有N个数据,每个数据m维
    label:(N)有N个标签
    """
    fig_path = "/home/ma-user/work/proda/visualization/tsne_{}.png".format(id)
    features = features.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    # import pandas as pd
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # import seaborn as sns
    # class_num = len(np.unique(labels)）#要分类的种类个数 eg:[0,1,2,3]这个就是为4



    # tsne_features = tsne.fit_transform(features)#将特征使用PCA降维至2维
    # print('tsne_features的shape:',tsne_features.shape)
    # # plt.scatter(tsne_features[:, 0], tsne_features[:, 1])#将对降维的特征进行可视化
    # # plt.show()
    # plt.savefig(fig_path)

    # sns.set()
    # df = pd.DataFrame()
    # df["y"] = labels
    # df["comp-1"] = tsne_features[:,0]
    # df["comp-2"] = tsne_features[:,1]


    # fig = sns.scatterplot(x="comp-1", y="comp-2",hue=df.y.tolist(),
    #               palette=sns.color_palette("hls", class_num),
    #               data=df).set(title="Bearing data T-SNE projection")

    # scatter_fig = fig.get_figure()
    # scatter_fig.savefig(fig_path, dpi = 400)

    tSNE = TSNE()
    word_embeddings = tSNE.fit_transform(features)
    classifier = KMeans(n_clusters=len(np.unique(labels)))
    classifier.fit(word_embeddings)
    labels = classifier.labels_
    min_left = min(word_embeddings[:, 0])
    max_right = max(word_embeddings[:, 0])
    min_bottom = min(word_embeddings[:, 1])
    max_top = max(word_embeddings[:, 1])
    # markers = ["bo","go",,"mo","yo","ko","bx","gx", "rx"]
    colors =["b","g","r","y", "k", "slategrey","slateblue","pink"]
    marks = ["o","o","o","o","o","o","o","o","o","o","x","x","x","x","x","x","x","x","x","x"]
    for i in range(len(word_embeddings)):
        plt.plot(word_embeddings[i][0], word_embeddings[i][1], marker=marks[labels[i]], color=colors[labels[i]])
    plt.axis([min_left, max_right, min_bottom, max_top])
    plt.savefig(fig_path)
    plt.clf()


def plot_histogram(image1,image2,n):
    # image1 = image1.reshape(image1.shape[0],-1).cpu()
    # image2 = image2.reshape(image2.shape[0],-1).cpu()
    image1 = image1.reshape(-1).cpu()
    image2 = image2.reshape(-1).cpu()
    image3 = torch.cat((image1,image2),0).detach().numpy()
    image1 = image1.detach().numpy()
    imagez = image2.detach().numpy()
    # bins = np.linspace(image3.min(),image3.max(),n)
    bins = np.linspace(-0.045,0.045,n)
    # for i in range(image1.shape[0]):
    # pdb.set_trace()
    i = 0
    j = 8
    # plt.ylim((0,15000))
    plt.ylim((0,400))
    # plt.hist(image1[i], bins, alpha=0.5, label='x_1')
    # plt.hist(image1[j], bins, alpha=0.5, label='x_2')
    plt.hist(image1, bins, alpha=0.5, label='Image features')
    plt.hist(image2, bins, alpha=0.5, label='Text features')
    plt.legend(loc='upper right',fontsize=15)
    # print("image",image1[i].mean(),image1[j].mean(),image1[i].mean()-image1[j].mean())
    fig_path = "/home/ma-user/work/proda/visualization/histogram_kl.png"
    plt.savefig(fig_path)
    plt.clf()
    # plt.ylim((0,15000))
    # plt.hist(image2[i], bins, alpha=0.5, label='adv_1')
    # plt.hist(image2[j], bins, alpha=0.5, label='adv_2')
    # plt.legend(loc='upper right')
    # print("text",image2[i].mean(),image2[j].mean(),image2[i].mean()-image2[j].mean())
    # fig_path = "/home/ma-user/work/proda/visualization/histogram_text0.png"
    # plt.savefig(fig_path)
    # plt.clf()
    # pdb.set_trace()

def cosine_loss(q,k):
    # pdb.set_trace()
    q = q.repeat(1,k.shape[1],1)
    # k = k.squeeze(1)
    # q = q/q.norm(dim=-1)
    k_norm = k.norm(dim=-1,keepdim=True)
    # pdb.set_trace()
    # k_norm = k.norm(dim=-1).unsqueeze(1).repeat(1,k.shape[1])
    k = k/k_norm
    cos = ((q*k)/(k.shape[0]*k.shape[1])).sum()
    return 1-cos
