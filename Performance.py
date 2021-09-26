"""

    Created on 24/09/21 11:30 AM 
    @author: Kartik Prabhu

"""
import matplotlib.pyplot as plt
import numpy as np

def bar_plot(xlabels,barlabels, bar_data1, bar_data2, fileName, title):
    plt.clf()
    x = np.arange(len(xlabels))  # the label locations
    width = 0.45  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, bar_data1, width, label=barlabels[0])
    rects2 = ax.bar(x + width/2, bar_data2, width, label=barlabels[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('IoU')
    # ax.set_xlabel('Datasets')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    plt.xticks(rotation=45)
    fig.tight_layout()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
    plt.savefig(fileName, bbox_inches='tight',format='pgf')

if __name__ == '__main__':

    # #Baseline
    labels = ["Pix3d(no aug)","Pix3d","s2r_v1(no aug)","s2r_v1","s2r_v2"]
    bar_data1 = [0.3106,0.3328,0.1739,0.1943,0.1825]
    bar_data2 = [0.3108,0.323,0.1289, 0.1766,0.1856]
    bar_labels =["Pix2Vox++","Pix2Vox"]

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/baseline1.pgf",
             "Baseline comparison with different datasets")


    #FineTuning
    labels = ["Pix3D","s2r_v1","s2r_v2","s2r_v1+pix3d","s2r_v2+pix3d"]
    bar_data1 = [0.3328,0.1943,0.1825,0.3464,0.3569]
    bar_data2 = [0.323,0.1766,0.1856,0.3385,0.3129]
    bar_labels =["Pix2Vox++","Pix2Vox"]

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/finetuning1.pgf",
             "Baseline comparison with finetuning of different datasets")


    #Mixed training Pix2VoxPP
    labels = ["15%","25%","50%","75%","90%"]
    bar_data1 = [0.3508,0.3322,0.3501,0.3561,0.3459]
    bar_data2 = [0.3182,0.3424,0.3632,0.3519,0.3526]
    bar_labels =["V1 on Pix2Vox++","V2 on Pix2Vox++"]

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/mixed1.pgf",
             "Mixed training on Pix2Vox++ using 2 versions of S2R:3DFREE")

    #Abalation study
    labels = ["Pix3d(chair,no aug)","Pix3d(chair)","Textureless","Textureless+Light","Textured","Textured+Light","Multi-Object"]
    bar_data1 = [0.3035,0.3308,0.2492,0.2435,0.2355,0.2196,0.2406]
    bar_data2 = [0.2664,0.2907,0.1846,0.2415,0.213,0.1544,0.2226]
    bar_labels =["Pix2Vox++","Pix2Vox"]

    bar_plot(labels,bar_labels,bar_data1,bar_data2,
         "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/ablation1.pgf",
             "Abalation study on chairs")


    #Abalation study with mixed training
    labels = ["Textureless","Textureless+Light","Textured","Textured+Light","Multi-Object"]
    bar_data1 = [0.3648,0.3616,0.348,0.3425,0.3503]
    bar_data2 = [0.341,0.339,0.3404,0.3301,0.3431]
    bar_labels =["Pix2Vox++","Pix2Vox"]
    bar_plot(labels,bar_labels,bar_data1,bar_data2,
             "/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/performance/ablation2.pgf",
             "Abalation study on chairs with mixed training")
