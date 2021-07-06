"""

    Created on 01/06/21 11:40 AM
    @author: Kartik Prabhu

"""


class BaseDataset(object):

    def get_trainset(self, transforms=None,images_per_category=0):
        pass

    def get_testset(self, transforms=None,images_per_category=0):
        pass

    def get_train_test_split(self,filePath):
        pass