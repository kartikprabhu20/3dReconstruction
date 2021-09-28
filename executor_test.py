"""

    Created on 25/09/21 10:19 AM 
    @author: Kartik Prabhu

"""


import Baseconfig
from PipelineManager import PipelineManager
from ModelManager import ModelType

if __name__ == '__main__':

    config = Baseconfig.config

    model_names = ["test_86_real","test_111","test_99","test_100"]
    model_types = [ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL]
    chair_dataset = [False, False,False, False]

    for model_name, model_type, chair in zip(model_names,model_types,chair_dataset):

        print("Testing for model name:" +model_name+" model type:"+model_type)
        config.pretrained_name = model_name
        config.main_name = model_name+"_test"
        config.model_type = model_type

        config.tensorboard_train = config.output_path + config.main_name  + '/tensorboard/tensorboard_training/'
        config.tensorboard_validation = config.output_path + config.main_name  + '/tensorboard/tensorboard_validation/'
        config.checkpoint_path = config.output_path + config.main_name + "/"
        config.pretrained_checkpoint_path = config.output_path + config.pretrained_name + "/"

        config.pix3d.test_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_test_chair.npy' if chair else config.root_path + '/Datasets/pix3d/splits/pix3d_test_2.npy'

        print("configuration: " + str(config))

        pipeline = PipelineManager(config).get_pipeline(pipeline_type=config.pipeline_type)
        pipeline.realdata_test()

        break
