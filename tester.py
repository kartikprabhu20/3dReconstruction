
import scipy
import sklearn
# from sklearn.feature_extraction import image
# from scipy.io import loadmat
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import kaolin as kal
from visualise import Visualise

if __name__ == '__main__':
    # images = loadmat('IMAGES.mat',variable_names='IMAGES',appendmat=True).get('IMAGES')
    # imgplot = plt.imshow(images[0])
    # plt.show()

    # datapath = '/Users/apple/OVGU/Thesis/Dataset/ModelNet40_small/'
    datapath = '/Users/apple/OVGU/Thesis/Dataset/ModelNet40_small/'

    mesh = Visualise(datapath).__mesh__()
    mesh.show()
    # kal.visualize.show(mesh)