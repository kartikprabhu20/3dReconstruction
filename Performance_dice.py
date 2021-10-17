"""

    Created on 24/09/21 11:30 AM 
    @author: Kartik Prabhu

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates

from matplotlib.lines import Line2D

def bar_plot(xlabels,barlabels, bar_data1, bar_data2, fileName, title, ylabel='F1 Score', xlabel='Dataset', ymin=0, ymax=0.5, fontsize = 9):
    plt.clf()
    x = np.arange(len(xlabels))  # the label locations
    width = 0.45  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, bar_data1, width, label=barlabels[0])
    rects2 = ax.bar(x + width/2, bar_data2, width, label=barlabels[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
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
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_title(title)
    plt.legend()
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
    ax.set_ylabel('F1 Score')
    ax.set_xlabel(xlabel)
    plt.xticks(rotation=45)
    # plt.show()
    plt.savefig(fileName, bbox_inches='tight',format='pgf')

def parallel(data,name,fileName,xlabel='Category',ylabel='F1 Score'):
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
    bar_data1 = [0.3782,0.4421,0.1054,0.1477,0.2166]
    bar_data2 = [0.4087,0.4281,0.1211,0.104,0.267]
    bar_labels =["Pix2Vox++","Pix2Vox"]

    data = [bar_data1,bar_data2]
    lineplot(labels, data, bar_labels,
                 "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/baseline_dice_linegraph1.pgf")

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/baseline_dice_barplot1.pgf",
             "Baselines trained on Pix3D and S2R:3DFREE")

    # #FineTuning
    labels = ["Pix3D","s2r_v1","s2r_v2","s2r_v1+pix3d","s2r_v2+pix3d"]
    bar_data1 = [0.4421,0.1477,0.21659,0.4177,0.4065]
    bar_data2 = [0.4281,0.104,0.267,0.4155,0.431]
    bar_labels =["Pix2Vox++","Pix2Vox"]
    data = [bar_data1,bar_data2]
    lineplot(labels, data, bar_labels,
                 "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetuning_dice_linegraph1.pgf")

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetuning_dice_barplot1.pgf",
             "Baselines trained on S2R:3DFREE and fine-tuned with Pix3D",ymin=0, ymax=0.6)

    # #Mixed training Pix2VoxPP
    labels = ["15%","25%","50%","75%","90%"]
    bar_data1 = [0.4493,0.4608,0.4644,0.4377,0.4472]
    bar_data2 = [0.4142, 0.4511,0.4543,0.4479,0.4386]
    bar_labels =["V1 on Pix2Vox++","V2 on Pix2Vox++"]
    #
    # bar_plot(labels,bar_labels,bar_data1,bar_data2,
    #          "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed1.pgf",
    #          "Mixed training on Pix2Vox++ using 2 versions of S2R:3DFREE")
    data = [bar_data1,bar_data2]
    lineplot(labels, data, bar_labels,
                 "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_dice_linegraph1.pgf",
             ymin=0.35, ymax=0.5,xlabel="Percentage of real data(Pix3D) per mini-batch")

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_dice_barplot1.pgf",
             "Mixed training on Pix2Vox++ using 2 versions of S2R:3DFREE",ymin=0.35, ymax=0.5)

    # #Mixed training Pix2Vox
    bar_data1 = [0.4476,0.4428,0.4499,0.4391,0.4355]
    bar_data2 = [0.406,0.4513,0.4352,0.4373,0.439]
    data = [bar_data1,bar_data2]
    bar_labels =["V1 on Pix2Vox","V2 on Pix2Vox"]
    lineplot(labels, data, bar_labels,
                 "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_dice_linegraph2.pgf",
             ymin=0.35, ymax=0.5, xlabel="Percentage of real data(Pix3D) per mini-batch")

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_dice_barplot2.pgf",
             "Mixed training on Pix2Vox using 2 versions of S2R:3DFREE",ymin=0.35, ymax=0.5)


    # #Abalation study
    labels = ["Pix3d(chair,no aug)","Pix3d(chair)","Textureless","Textureless+Light","Textured","Textured+Light","Multi-Object","Combined"]

    bar_data1 = [0.4047,0.4547,0.2933,0.242,0.3513,0.3519,0.3705,0.3851]
    bar_data2 = [0.3944,0.4175,0.1365,0.1795,0.1959,0.1769,0.168,0.2289]
    bar_labels =["Pix2Vox++","Pix2Vox"]
    data = [bar_data1,bar_data2]
    lineplot(labels, data, bar_labels,
                 "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/ablation_dice_linegraph1.pgf")

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/ablation_dice_barplot1.pgf",
             "Abalation study on chairs", fontsize = 7)
    #
    # #Abalation study with mixed training
    labels = ["Pix3d(chair)","Textureless","Textureless+Light","Textured","Textured+Light","Multi-Object","Combined"]
    bar_data1 = [0.4547,0.4879, 0.4821,0.4671,0.4369,0.4364,0.4962]
    bar_data2 = [0.4175,0.4659,0.4641,0.4672,0.4548,0.4665,0.4812]
    bar_labels =["Pix2Vox++","Pix2Vox"]
    # bar_plot(labels,bar_labels,bar_data1,bar_data2,
    #          "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/ablation2.pgf",
    #          "Abalation study on chairs with mixed training")
    data = [bar_data1,bar_data2]
    lineplot(labels, data, bar_labels,ymin=0.3, ymax=0.6,
             fileName="/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/ablation_dice_linegraph2.pgf")

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/ablation_dice_barplot2.pgf",
             "Abalation study on chairs with mixed training",ymin=0.3, ymax=0.6)

    #Paralell coordinates per category
    #Mixed pix2vox
    labels = ["model","chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_labels =["Pix3D","S2R_V1","S2R_V2"]
    data1 = [0.3883,0.3894,0.3752]
    data2 = [0.8013,0.7748,0.8061]
    data3 = [0.2456,0.2458,0.2347]
    data4 = [0.4656,0.519,0.4684]
    data5 = [0.2856,0.3228,0.3247]
    data6 = [0.6882,0.7603,0.7382]
    data7 = [0.3807,0.3977,0.427]

    df = pd.DataFrame(list(zip(bar_labels,data1, data2,data3,data4,data5,data6,data7)),
                  columns =labels)
    parallel(df,name="model",fileName="/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_dice_parallel_pix2vox.pgf")

    data1 = [0.3883,0.8013,0.2456,0.4656,0.2856,0.6882,0.3807]
    data2 = [0.3894,0.7748,0.2458,0.519,0.3228,0.7603,0.3977]
    data3 = [0.3752,0.8061,0.2347,0.4684,0.3247,0.7382,0.427]
    labels = ["chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_plot_2(labels,bar_labels,data1,data2,data3,
               "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_dice_barplot_pix2vox.pgf")


    #Paralell coordinates per category
    #Mixed pix2voxpp
    labels = ["model","chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_labels =["Pix3D","S2R_V1","S2R_V2"]
    data1 = [0.4016,0.4253,0.4211]
    data2 = [0.7452,0.789,0.8135]
    data3 = [0.251,0.2564,0.2422]
    data4 = [0.4783,0.5056,0.4485]
    data5 = [0.2973,0.3429,0.3535]
    data6 = [0.7234,0.7468,0.7222]
    data7 = [0.3881,0.4066,0.485]

    df = pd.DataFrame(list(zip(bar_labels,data1, data2,data3,data4,data5,data6,data7)),
                      columns =labels)
    parallel(df,name="model",fileName="/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_dice_parallel_pix2voxpp.pgf")

    data1 = [0.4016,0.7452,0.251,0.4783,0.2973,0.7234,0.3881]
    data2 = [0.4253,0.789,0.2564,0.5056,0.3429,0.7468,0.4066]
    data3 = [0.4211,0.8135,0.2422,0.4485,0.3535,0.7222,0.485]
    labels = ["chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_plot_2(labels,bar_labels,data1,data2,data3,
           "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed_dice_barplot_pix2voxpp.pgf")


    #Paralell coordinates per category
    #Finetuning pix2voxpp
    labels = ["model","chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_labels =["Pix3D","S2R_V1","S2R_V2"]
    data1 = [0.4016,0.3766,0.374]
    data2 = [0.7452,0.7284,0.7064]
    data3 = [0.251,0.2326,0.212]
    data4 = [0.4783,0.4488,0.4083]
    data5 = [0.2973,0.2952,0.2656]
    data6 = [0.7234,0.7048,0.6946]
    data7 = [0.3881,0.2826,0.3334]
    df = pd.DataFrame(list(zip(bar_labels,data1, data2,data3,data4,data5,data6,data7)),
                      columns =labels)
    parallel(df,name="model",fileName="/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetune_dice_parallel_pix2voxpp.pgf")

    data1 = [0.4016,0.7452,0.251,0.4783,0.2973,0.7234,0.3881]
    data2 = [0.3766,0.7284,0.2326,0.4488,0.2952,0.7048,0.2826]
    data3 = [0.374,0.7064,0.212,0.4083,0.26565,0.69468,0.3334]
    labels = ["chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_plot_2(labels,bar_labels,data1,data2,data3,
           "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetune_dice_barplot_pix2voxpp.pgf")


    #Paralell coordinates per category
    #Mixed pix2vox
    labels = ["model","chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_labels =["Pix3D","S2R_V1","S2R_V2"]
    data1 = [0.3883,0.3703,0.3798]
    data2 = [0.8013,0.7694,0.6646]
    data3 = [0.2456,0.2371,0.2311]
    data4 = [0.4656,0.4309,0.4822]
    data5 = [0.2856,0.2883,0.2757]
    data6 = [0.6882,0.6977,0.7327]
    data7 = [0.3807,0.335,0.39417]

    df = pd.DataFrame(list(zip(bar_labels,data1, data2,data3,data4,data5,data6,data7)),
                      columns =labels)
    parallel(df,name="model",fileName="/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetune_dice_parallel_pix2vox.pgf")

    data1 = [0.3883,0.8013,0.2456,0.4656,0.2856,0.6882,0.3807]
    data2 = [0.3703,0.7694,0.2371,0.4309,0.2883,0.6977,0.335]
    data3 = [0.3798,0.6646,0.2311,0.4822,0.2757,0.7327,0.39417]
    labels = ["chair(1250)","wardrobe(73)","table(585)","bed(301)","desk(216)","sofa(594)","bookcase(121)"]
    bar_plot_2(labels,bar_labels,data1,data2,data3,
               "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetune_dice_barplot_pix2vox.pgf")
