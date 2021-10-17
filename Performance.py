"""

    Created on 24/09/21 11:30 AM 
    @author: Kartik Prabhu

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates

from matplotlib.lines import Line2D

def bar_plot(xlabels,barlabels, bar_data1, bar_data2, fileName, title, ylabel='IoU', xlabel='Dataset', ymin=0, ymax=0.5, fontsize = 9):
    plt.clf()
    x = np.arange(len(xlabels))  # the label locations
    width = 0.45  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, bar_data1, width, label=barlabels[0])
    rects2 = ax.bar(x + width/2, bar_data2, width, label=barlabels[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel, fontweight='bold',fontsize=14)
    ax.set_xlabel(xlabel, fontweight='bold',fontsize=14)
    ax.set_title(title,fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    plt.tick_params(labelsize=12)
    ax.set_ylim(ymin=ymin, ymax=ymax)

    ax.bar_label(rects1, padding=3,fontsize=fontsize)
    ax.bar_label(rects2, padding=3,fontsize=fontsize)
    plt.xticks(rotation=45)
    fig.tight_layout()
    ax.legend()
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #        ncol=2, mode="expand", borderaxespad=0.)
    # plt.show()
    plt.savefig(fileName, bbox_inches='tight',format='pgf')


def bar_plot_2(xlabels,barlabels, bar_data1, bar_data2, bar_data3, fileName, title="", ylabel='IoU', xlabel='Category', ymin=0, ymax=0.5, fontsize = 9):
    plt.clf()
    barWidth = 0.25
    fig, ax = plt.subplots()

    # Set position of bar on X axis
    r1 = np.arange(len(bar_data1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bar_data1, width=barWidth, edgecolor='white', label=barlabels[0])
    plt.bar(r2, bar_data2, width=barWidth, edgecolor='white', label=barlabels[1])
    plt.bar(r3, bar_data3, width=barWidth, edgecolor='white', label=barlabels[2])

    # Add xticks on the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bar_data1))], xlabels)
    ax.set_ylabel(ylabel, fontweight='bold',fontsize=14)
    ax.set_xlabel(xlabel, fontweight='bold',fontsize=14)
    ax.set_title(title,fontsize=14)
    plt.legend()
    plt.tick_params(labelsize=12)
    plt.xticks(rotation=45)
    # plt.show()
    plt.savefig(fileName, bbox_inches='tight',format='pgf')


def lineplot(labels, data_list, legends, fileName, ymin=0, ymax=0.5, xlabel="Datasets"):
    plt.clf()

    fig, ax = plt.subplots()
    xlables = [n for n in range(0,len(labels))]

    markers = list(Line2D.markers.keys())[2:]
    # plotting the line 1 points
    for data,legend,marker in zip(data_list,legends,markers):
        plt.plot(xlables, data, label = legend,marker=marker)

    ax.set_xticks(range(0,len(labels)))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_ylabel('IoU')
    ax.set_xlabel(xlabel)
    plt.xticks(rotation=45)
    # plt.show()
    plt.savefig(fileName, bbox_inches='tight',format='pgf')

def parallel(data,name,fileName,xlabel='Category',ylabel='IoU'):
    plt.clf()
    ax = parallel_coordinates(data,name,color=('#eb4034', '#0bb864', '#0b50b8'))
    plt.xticks(rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.grid(alpha=0.3)
    # plt.show()
    plt.savefig(fileName, bbox_inches='tight',format='pgf')

if __name__ == '__main__':

    # #Baseline
    labels = ["Pix3d(no aug)","Pix3d","s2r_v1(no aug)","s2r_v1","s2r_v2"]
    bar_data1 = [0.2723,0.3529,0.1739,0.1886,0.1896]
    bar_data2 = [0.3108,0.323,0.1289, 0.1876,0.1856]
    bar_labels =["Pix2Vox++","Pix2Vox"]

    data = [bar_data1,bar_data2]
    lineplot(labels, data, bar_labels,
                 "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/baseline_linegraph1.pgf")


    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/baseline_barplot1.pgf",
             "Baselines trained on Pix3D and S2R:3DFREE", ymax=0.4)

    labels = ["Pix3d(no aug)","Pix3d","s2r_v1(no aug)","s2r_v1","s2r_v2"]
    bar_data1 = [0.2777,0.3443,0.0577,0.0847,0.1297]
    bar_data2 = [0.3108,0.325,0.0679, 0.0571,0.1643]
    bar_labels =["Pix2Vox++","Pix2Vox"]

    data = [bar_data1,bar_data2]
    lineplot(labels, data, bar_labels,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/baseline_linegraph2.pgf")

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/baseline_barplot2.pgf",
             "Baselines trained on Pix3D and S2R:3DFREE", ymax=0.4)

    # #FineTuning
    labels = ["Pix3D","s2r_v1","s2r_v2","s2r_v1+pix3d","s2r_v2+pix3d"]
    bar_data1 = [0.3443,0.0847,0.1297,0.3139,0.3089]
    bar_data2 = [0.325,0.0571,0.1643,0.3125,0.3338]
    bar_labels =["Pix2Vox++","Pix2Vox"]
    data = [bar_data1,bar_data2]
    lineplot(labels, data, bar_labels,
                 "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetuning_linegraph1.pgf")

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetuning_barplot1.pgf",
             "Baselines trained on S2R:3DFREE and fine-tuned with Pix3D")

    # #Mixed training Pix2VoxPP
    labels = ["15%","25%","50%","75%","90%"]
    bar_data1 = [0.3508,0.3607,0.3636,0.3561,0.3459]
    bar_data2 = [0.3134,0.3556,0.3587,0.3494,0.3482]
    bar_labels =["V1 on Pix2Vox++","V2 on Pix2Vox++"]

    data = [bar_data1,bar_data2]
    lineplot(labels, data, bar_labels,
                 "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_linegraph1.pgf",
             ymin=0.28, ymax=0.38,xlabel="Percentage of real data(Pix3D) per mini-batch")
    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_barplot1.pgf",
             "Mixed training on Pix2Vox++ using 2 versions of S2R:3DFREE",ymin=0.25, ymax=0.4)

    bar_data1 = [0.3483,0.3432,0.3514,0.343,0.3389]
    bar_data2 = [0.3027,0.3513,0.3416,0.3413,0.3434]
    data = [bar_data1,bar_data2]
    bar_labels =["V1 on Pix2Vox","V2 on Pix2Vox"]
    lineplot(labels, data, bar_labels,
                 "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_linegraph2.pgf",
             ymin=0.25, ymax=0.38, xlabel="Percentage of real data(Pix3D) per mini-batch")

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_barplot2.pgf",
             "Mixed training on Pix2Vox using 2 versions of S2R:3DFREE",ymin=0.25, ymax=0.4)

    # #Abalation study
    labels = ["Pix3d(chair,no aug)","Pix3d(chair)","Textureless","Textureless+Light","Textured","Textured+Light","Multi-Object","Combined"]
    # bar_data1 = [0.3035,0.3308,0.2492,0.2435,0.2355,0.2196,0.2406]
    # bar_data2 = [0.2664,0.2907,0.1846,0.2415,0.213,0.1544,0.2226]
    bar_data1 = [0.2797,0.3305,0.1798,0.143,0.2217,0.2221,0.2372,0.2497]
    bar_data2 = [0.2694,0.2916,0.0748, 0.1026,0.112,0.0991,0.0943,0.133]
    bar_labels =["Pix2Vox++","Pix2Vox"]

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/ablation_barplot1.pgf",
             "Abalation study on chairs", ymax=0.4, fontsize = 7)

    data = [bar_data1,bar_data2]
    lineplot(labels, data, bar_labels,
                 "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/ablation_linegraph1.pgf")

    # #Abalation study with mixed training
    labels = ["Pix3d(chair)","Textureless","Textureless+Light","Textured","Textured+Light","Multi-Object","Combined"]
    bar_data1 = [0.3305,0.3648,0.3616,0.348,0.3425,0.3503,0.3774]
    bar_data2 = [0.2916,0.341,0.339,0.3404,0.3301,0.3431,0.3608]
    bar_labels =["Pix2Vox++","Pix2Vox"]

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/ablation_barplot2.pgf",
             "Abalation study on chairs with mixed training")

    data = [bar_data1,bar_data2]
    lineplot(labels, data, bar_labels,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/ablation_linegraph2.pgf")


    #Paralell coordinates per category
    #Mixed pix2vox
    labels = ["model","chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_labels =["Pix3D","S2R_V1","S2R_V2"]
    data1 = [0.2715,0.2756,0.268]
    data2 = [0.7287,0.7087,0.7479]
    data3 = [0.1831,0.2756,0.1790]
    data4 = [0.355,0.4072,0.362]
    data5 = [0.2004,0.2365,0.2404]
    data6 = [0.5703,0.652,0.6337]
    data7 = [0.2747,0.3007,0.3393]
    df = pd.DataFrame(list(zip(bar_labels,data1, data2,data3,data4,data5,data6,data7)),
                  columns =labels)
    parallel(df,name="model",fileName="/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_parallel_pix2vox.pgf")

    data1 = [0.2715,0.7287,0.1831,0.355,0.2004,0.5703,0.2747]
    data2 = [0.2756,0.7087,0.2756,0.4072,0.2365,0.652,.3007]
    data3 = [0.268,0.7479,0.1790,0.362,0.2404,0.6337,0.3393]
    labels = ["chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_plot_2(labels,bar_labels,data1,data2,data3,
               "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_barplot_pix2vox.pgf")

    #Paralell coordinates per category
    #Mixed pix2voxpp
    labels = ["model","chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_labels =["Pix3D","S2R_V1","S2R_V2"]
    data1 = [0.2887,0.3063,0.3077]
    data2 = [0.6782,0.7219,0.7582]
    data3 = [0.1888,0.1964,0.1859]
    data4 = [0.3739,0.397,0.3493]
    data5 = [0.2112,0.2533,0.2664]
    data6 = [0.618,0.6401,0.6194]
    data7 = [0.291,0.3051,0.3874]
    df = pd.DataFrame(list(zip(bar_labels,data1, data2,data3,data4,data5,data6,data7)),
                      columns =labels)
    parallel(df,name="model",fileName="/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_parallel_pix2voxpp.pgf")

    data1 = [.2887,.6782,.1888,.3739,.2112,.618,.291]
    data2 = [.3063,.7219,.1964,.397,.2533,.6401,.3051]
    data3 = [.3077,.7582,.1859,.3493,.2664,.6194,.3874]
    labels = ["chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_plot_2(labels,bar_labels,data1,data2,data3,
               "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_barplot_pix2voxpp.pgf")

    #Paralell coordinates per category
    #Finetuning pix2voxpp
    labels = ["model","chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_labels =["Pix3D","S2R_V1","S2R_V2"]
    data1 = [0.2887,0.2589,0.2605]
    data2 = [0.6782,0.6454,0.63]
    data3 = [0.1888,0.1715,0.1579]
    data4 = [0.3739,0.3382,0.3051]
    data5 = [0.2112,0.2096,0.185]
    data6 = [0.618,0.5897,0.5813]
    data7 = [0.291,0.2006,0.2388]

    df = pd.DataFrame(list(zip(bar_labels,data1, data2,data3,data4,data5,data6,data7)),
                      columns =labels)
    parallel(df,name="model",fileName="/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetune_parallel_pix2voxpp.pgf")

    data1 = [.2887,.6782,.1888,.3739,.2112,.618,.291]
    data2 = [.2589,.6454,.1715,.3382,.2096,.5897,.2006]
    data3 = [.2605,.63,.1579,.3051,.185,.5813,.2388]
    labels = ["chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_plot_2(labels,bar_labels,data1,data2,data3,
               "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetune_barplot_pix2voxpp.pgf")


    #Paralell coordinates per category
    #Mixed pix2vox
    labels = ["model","chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_labels =["Pix3D","S2R_V1","S2R_V2"]
    data1 = [0.2715,0.2534,0.2684]
    data2 = [0.7287,0.6806,0.6965]
    data3 = [0.1831,0.1735,0.1765]
    data4 = [0.355,0.3176,0.3722]
    data5 = [0.2004,0.2006,0.1963]
    data6 = [0.5703,0.5824,0.6202]
    data7 = [0.2747,0.2338,0.296]

    df = pd.DataFrame(list(zip(bar_labels,data1, data2,data3,data4,data5,data6,data7)),
                      columns =labels)
    parallel(df,name="model",fileName="/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetune_parallel_pix2vox.pgf")

    data1 = [.2715,.7287,.1831,.355,.2004,.5703,.2747]
    data2 = [.2534,.6806,.1735,.3176,.2006,.5824,.2338]
    data3 = [.2684,.6965,.1765,.3722,.1963,.6202,.296]
    labels = ["chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_plot_2(labels,bar_labels,data1,data2,data3,
           "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetune_barplot_pix2vox.pgf")
