
import scipy
import sklearn
# from sklearn.feature_extraction import image
# from scipy.io import loadmat
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import kaolin as kal
from visualise import Visualise
import json

if __name__ == '__main__':

    pix3d_json_path = '/Users/apple/OVGU/Thesis/Dataset/pix3d/pix3d.json'
    y = json.load(open(pix3d_json_path))
    print(y[1])

