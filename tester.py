# from sklearn.feature_extraction import image
# from scipy.io import loadmat
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import json


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class AnalysisDataAndFitLinearRegression:

    def _init_(self):
        self.version = 1

    def analyse_and_fit_lrm(self, path):
        # a path to a dataset is "./data/realest.csv"
        # dataset can be loaded by uncommenting the line bellow
        data = pd.read_csv(path)
        model_parameters = {}
        price_prediction = {}

        df = pd.read_csv(path, dtype=str)
        df_main= df.astype('float')
        col_names= pd.read_csv(path, nrows=0).columns
        # print(col_names)
        df=self.__listwise_deletion(df_main)

        taxDf=df_main.loc[(df_main.Bedroom == 4) & (df_main.Bathroom == 2)]['Tax']
        print(np.mean(taxDf))
        model_parameters['statistics']=np.array([np.mean(taxDf),np.std(taxDf),np.median(taxDf),np.min(taxDf),np.max(taxDf)])


        spaceDf= self.__listwise_deletion(df_main[df_main.Space > 800])
        spaceDf= spaceDf.sort_values('Price', ascending=False)
        model_parameters['data_frame']= spaceDf


        lotDf=df_main['Lot'].sort_values(ascending=True)

        print(df_main[df_main.Lot >= lotDf.quantile(.8)].shape[0])
        lotDf=df_main[df_main.Lot >= lotDf.quantile(.8)]
        model_parameters['number_of_observations'] = lotDf.shape[0]



        labels = df['Price']

        y_train= df.pop('Price')
        x_train = df

        # print(x_train)
        regression= LinearRegression()
        regression.fit(x_train, y_train)
        price_prediction['model_parameters']={'Intercept': regression.intercept_,
                                              'Bedroom': regression.coef_[0],
                                              'Space': regression.coef_[1],
                                              'Room':regression.coef_[2],
                                              'Lot': regression.coef_[3],
                                              'Tax': regression.coef_[4],
                                              'Bathroom': regression.coef_[5],
                                              'Garage': regression.coef_[6],
                                              'Condition': regression.coef_[7]}

        data=[[3,1500,8,40,1000,2,1,0]]
        df_test= pd.DataFrame(data, columns= ['Bedroom','Space','Room','Lot','Tax','Bathroom','Garage','Condition'])

        y_pred= regression.predict(df_test)
        price_prediction['price_prediction']= y_pred[0]


        return {
            'model_parameters': model_parameters,
            'price_prediction': price_prediction
        }

    def __listwise_deletion(self, data: pd.DataFrame):
        return data.dropna()

if __name__ == '__main__':

    # pix3d_json_path = '/Users/apple/OVGU/Thesis/Dataset/pix3d/pix3d.json'
    # y = json.load(open(pix3d_json_path))
    # print(y[1])

    a = AnalysisDataAndFitLinearRegression()
    dict =a.analyse_and_fit_lrm('/Users/apple/Downloads/realest.csv')
    print(dict)



