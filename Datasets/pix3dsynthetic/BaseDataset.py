"""

    Created on 01/06/21 11:40 AM
    @author: Kartik Prabhu

"""


class BaseDataset(object):

    def get_trainset(self):
        pass

    def get_testset(self):
        pass

    def get_train_test_split(self,filePath):
        pass