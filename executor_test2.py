"""

    Created on 25/09/21 10:19 AM 
    @author: Kartik Prabhu

"""


import Baseconfig
from PipelineManager import PipelineManager
from ModelManager import ModelType

if __name__ == '__main__':

    config = Baseconfig.config
    config.load_model = True

    # model_names = ["test_86","test_123","test_126","test_111","test_124","test_127","test_123_1","test_126_1","test_124_1","test_127_1"]
    # model_types = [ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL]
    # chair_dataset = 10*[False]
    # thresholds = [0.05,0.05,0.05,0.07,0.05,0.10,0.07,0.05,0.10,0.05]
    #
    # model_names = model_names+["test_68_2","test_71_2","test_130","test_136",
    #                "test_112","test_114","test_90","test_89","test_95","test_97","test_109",
    #                "test_113","test_115","test_92","test_93","test_96","test_98","test_110",
    #                "test_122","test_118","test_107","test_105","test_116",
    #                "test_125","test_119","test_108","test_106","test_117"]
    # model_types = model_types+[ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,
    #                ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,
    #                ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,
    #                ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,
    #                ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL]
    # chair_dataset = chair_dataset+4*[False]+24*[True]
    # # [0.05, 0.075, .1, .2, .3, .4, .5, .6, .7]
    # thresholds = thresholds+[0.10,0.05,0.2,0.05,
    #               .3,.07,.1,.2,.3,.5,.3,
    #               .07,.1,.05,.5,.05,.5,.05,
    #               .2,.2,.2,.3,.4,
    #               .05,.1,0.05,.07,.1]

    # model_names = ["test_135","test_114","test_115"]
    # model_types = [ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP]
    # chair_dataset = [False]+2*[True]
    # thresholds = [0.1,0.07,0.2]

    # model_names = ["test_114","test_144","test_145","test_146","test_147"]
    # model_types = [ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL]
    # chair_dataset = [True]+ 4*[False]
    # thresholds = [0.07,0.05,0.05,0.05,0.05]
    model_names = ["test_148","test_149","test_150","test_151"]
    model_types = [ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL]
    chair_dataset = 4*[True]
    thresholds = [0.3,0.05,0.2,0.1]

    for model_name, model_type, chair, thresh in zip(model_names,model_types,chair_dataset, thresholds):
        try:
            print("Testing for model name:" +model_name+" model type:"+model_type)
            config.pretrained_name = model_name
            config.main_name = model_name+"_testing_best"
            config.model_type = model_type
            config.save_count = 70


            config.tensorboard_train = config.output_path + config.main_name  + '/tensorboard/tensorboard_training/'
            config.tensorboard_validation = config.output_path + config.main_name  + '/tensorboard/tensorboard_validation/'
            config.checkpoint_path = config.output_path + config.main_name + "/"
            config.pretrained_checkpoint_path = config.output_path + config.pretrained_name + "/"

            config.pix3d.test_indices = config.root_path+'/Datasets/pix3d/splits/pix3d_test_chair.npy' if chair else config.root_path+'/Datasets/pix3d/splits/pix3d_test_2.npy'

            print("configuration: " + str(config))
            config.TEST.VOXEL_THRESH_IMAGE = thresh

            pipeline = PipelineManager(config).get_pipeline(pipeline_type=config.pipeline_type)
            # pipeline.save_test_Images()
            pipeline.empty_test()

            pipeline = None
        except Exception as ex:
            print("Exception!!!!!!!!!!!!!!!!!!!")
            print(ex)


