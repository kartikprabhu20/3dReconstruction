"""

    Created on 14/09/21 1:03 AM 
    @author: Kartik Prabhu

"""
from collections import Counter, OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv(path):
    df = pd.read_csv(path)
    # data_top = df.head()
    # print(data_top)

    return df

def sum_colums(path, columnNames):
    df = read_csv(path)
    rowcount = df.shape[0]

    average = 0
    for name in columnNames:
        average += df[name].sum()/rowcount
        print (average)
    return average

def real_not_real(df,columnNames):

    total_real_count = 0
    total_not_real_count = 0

    for name in columnNames:
        # coldf = df[name]
        real_count = df[df[name] == 'Real'].shape[0]
        not_real_count = df[df[name] == 'Not Real'].shape[0]

        total_real_count += real_count
        total_not_real_count += not_real_count

    return  total_real_count, total_not_real_count

def horizontal_bar(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    print(labels)
    print(list(results.values()))
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax

def bar_plot(all_data, labels,fileName):

    fig, ax = plt.subplots()

    print(labels)
    print(len(all_data))
    # rectangular box plot
    medianprops = dict(linewidth=1.5, linestyle='-', color='#0e8f07')
    meanprops = dict(linewidth=1.5, color='#051478')

    bplot = ax.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,# fill with color
                       meanline =True,
                       showmeans =True,
                       medianprops=medianprops,
                       meanprops = meanprops,
                     labels=labels)  # will be used to label x-ticks
    ax.set_title('Rectangular box plot')



    # # plot violin plot
    # bplot = ax.violinplot(all_data,
    #                showmeans=True,
    #                showmedians=True)
    # ax.set_title('violin plot')


    # fill with colors
    colors = plt.get_cmap('Spectral')(
        np.linspace(0.1, 0.9, len(all_data)))

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.yaxis.grid(True)
    plt.xticks(rotation=45)
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Ranks')
    plt.savefig(fileName, bbox_inches='tight',format='pgf')


def question1(path, lists,names):
    df = read_csv(path)
    real_values = []
    notReal_values = []
    dict = {}
    for cat,name in zip(lists,names):
        real, notReal = real_not_real(df,cat)
        real_values.append(real)
        notReal_values.append(notReal)

        dict[name]=[real,notReal]
    print(dict)

    horizontal_bar(dict,["Real", "Not Real"])
    plt.savefig('/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/survey/question1.pgf', format='pgf')

    return  real_values,notReal_values


def one_to_ten(df, columnNames):
    df = read_csv(path)
    base = {"1":0, "2":0, "3":0, "4":0, "5":0,"6":0,"7":0,"8":0,"9":0,"10":0}
    for name in columnNames:
        distinctvalues = df[name].sort_values(ascending=True).value_counts()
        dictionary = dict(distinctvalues)
        a_counter = Counter(base)
        b_counter = Counter(dictionary)
        add_dict = a_counter + b_counter
        base = dict(add_dict)
    return base

def question2(path, lists,names,fileName,values):
    df = read_csv(path)
    dictionary = {}
    for cat,name in zip(lists,names):
        base = one_to_ten(df,cat)
        list1 = []
        for key in sorted(base):
            list1.append(base[key])
        dictionary[name]= list1
    print(dictionary)

    horizontal_bar(dictionary,values)
    plt.savefig(fileName, format='pgf')

def question3_barplot(path, lists,names,fileName,values):
    df = read_csv(path)
    dictionary = {}
    for cat,name in zip(lists,names):
        base = one_to_ten(df,cat)
        list1 = []
        for key in sorted(base):
            for i in range(base[key]):
                list1.append(key)
        dictionary[name]= list1
    print(dictionary)


    labels = list(dictionary.keys())
    print(labels)
    print(list(dictionary.values()))
    print(len(list(dictionary.values())))
    all_data = np.array(list(dictionary.values()))

    bar_plot(all_data,labels)
    # plt.savefig(fileName, format='pgf')

def question3(path,lists):
    average_list = []
    for cat in lists:
        average_list.append(sum_colums(path,cat)/3)

    return average_list

# def question2(path,lists):
#     average_list = []
#     for cat in lists:
#         average_list.append(sum_colums(path,cat))
#
#     return average_list

def getHistogram(names, values, fileName):
    plt.clf()
    xticks = [i for i in range(len(names))]
    plt.bar(xticks, height=values)
    plt.xticks(xticks, names)
    plt.savefig(fileName, format='pgf')

def getHistogram2(names, value1, value2, fileName):

    X_axis = np.arange(len(names))

    plt.bar(X_axis - 0.2, value1, 0.4, label = 'Real')
    plt.bar(X_axis + 0.2, value2, 0.4, label = 'Not Real')

    plt.xticks(X_axis, names)
    # plt.xlabel("Groups")
    plt.ylabel("Count")
    plt.title('Count of real and Not real')
    plt.legend()
    plt.savefig(fileName, format='pgf')

if __name__ == '__main__':
    path = '/Users/apple/OVGU/Thesis/code/3dReconstruction/Survey/Real_or_Not_Real2.csv'
    read_csv(path)

    threedfront_1 = ["1)Real or Not real?","10)Real or Not real?","19)Real or Not real?"]
    ai2thor_1 = ["2)Real or Not real?","11)Real or Not real?","20)Real or Not real?"]
    blenderproc_1 = ["3)Real or Not real?","12)Real or Not real?","21)Real or Not real?"]
    hyperism_1 = ["4)Real or Not real?","13)Real or Not real?","22)Real or Not real?"]
    interiornet_1 = ["5)Real or Not real?","14)Real or Not real?","23)Real or Not real?"]
    openrooms_1 = ["6)Real or Not real?","15)Real or Not real?","24)Real or Not real?"]
    pix3d_1 = ["7)Real or Not real?","16)Real or Not real?","25)Real or Not real?"]
    s2r_1 = ["8)Real or Not real?","17)Real or Not real?","26)Real or Not real?"]
    scenenet_1 = ["9)Real or Not real?","18)Real or Not real?","27)Real or Not real?"]


    threedfront_2 = ["1)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","10)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","11)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)"]
    ai2thor_2 = ["2)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","12)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","13)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)"]
    blenderproc_2 = ["3)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","14)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","15)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)"]
    hyperism_2 = ["4)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","16)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","17)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)"]
    interiornet_2 = ["5)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","18)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","19)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)"]
    openrooms_2 = ["6)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","20)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","21)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)"]
    pix3d_2 = ["7)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","22)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","23)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)"]
    s2r_2 = ["8)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","24)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","25)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)"]
    scenenet_2 = ["9)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","26)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)","27)Rate on scale of 1 to 10 ( 1-> not real, 10 -> real)"]

    threedfront_3 = ["1)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [A]",
                     "2)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [C]",
                     "3)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [C]"]
    ai2thor_3 = ["1)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [G]",
                 "2)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [G]",
                 "3)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [A]"]
    blenderproc_3 = ["1)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [C]",
                     "2)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [A]",
                     "3)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [G]"]
    hyperism_3 = ["1)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [D]",
                  "2)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [F]",
                  "3)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [B]"]
    interiornet_3 = ["1)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [E]",
                     "2)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [D]",
                     "3)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [I]"]
    openrooms_3 = ["1)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [F]",
                   "2)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [E]",
                   "3)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [H]"]
    pix3d_3 =["1)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [B]",
              "2)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [B]",
              "3)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [E]"]
    s2r_3 = ["1)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [H]",
             "2)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [I]",
             "3)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [D]"]
    scenenet_3 = ["1)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [I]",
                  "2)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [H]",
                  "3)Select a rank for each image. 1 -> Most real, 9 -> least real. (Note: Every Rank can be assigned to only a single image) [F]"]

    names = ["3DFRONT", "ai2THOR", "Blenderproc", "Hyperism", "InteriorNet", "OpenRooms","Pix3D", "S2R:3D-FREE", "SceneNet"]
    question3_list = [threedfront_3,ai2thor_3,blenderproc_3,hyperism_3,interiornet_3,openrooms_3,pix3d_3,s2r_3,scenenet_3]
    question2_list = [threedfront_2,ai2thor_2,blenderproc_2,hyperism_2,interiornet_2,openrooms_2,pix3d_2,s2r_2,scenenet_2]
    question1_list = [threedfront_1,ai2thor_1,blenderproc_1,hyperism_1,interiornet_1,openrooms_1,pix3d_1,s2r_1,scenenet_1]

    #section 1
    # real_values, notReal_values = question1(path,question1_list, names)

    #section 2
    # question2(path,question2_list,names,'/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/survey/question2.pgf',["1","2","3","4","5","6","7","8","9","10"])
    # averagelist = question3(path,question2_list)
    # getHistogram(names, averagelist,"/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/survey/question2_2.pgf")

    #section 3
    # question2(path,question3_list,names,'/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/survey/question3.pgf',["1","2","3","4","5","6","7","8","9"])
    #
    # averagelist = question3(path,question3_list)
    # getHistogram(names, averagelist,"/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/survey/question3_2.pgf")

    # question3_barplot(path,question3_list,names,'/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/survey/question3_3.pgf',["1","2","3","4","5","6","7","8","9","10"])
    #ran into some problem, copied these values from dictionary list generated from above line:question3_barplot

    bar_plot([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0], [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0], [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]]
             ,names,'/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/survey/question3_3.pgf')