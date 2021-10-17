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

    # model_names = ["test_118","test_119","test_115","test_114","test_107","test_108","test_105","test_106","test_116","test_117","test_95","test_96","test_89","test_93","test_90","test_92",
    #                "test_86","test_111","test_99","test_100","test_101","test_102","test_123","test_124","test_82","test_83","test_126","test_127",
    #                "test_81","test_67","test_68","test_78","test_79","test_80","test_71","test_69","test_72","test_73"]
    # model_types = [ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL
    #     ,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,
    #                ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,
    #                ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,]
    # chair_dataset = 16*[True]
    # chair_dataset += 12*[False]
    # chair_dataset += 10*[False]

    # model_names = ["test_64_1","test_65_1","test_67","test_68","test_71","test_69","test_72","test_73","test_82_1","test_83_1",
    #                "test_128","test_129","test_130","test_131", "test_132","test_100","test_64","test_65",
    #                "test_112","test_113","test_114","test_115","test_122","test_125","test_108"]
    # model_types = [ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,
    #                ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,
    #                ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL]
    # chair_dataset = 18*[False] + 7*[True]

    # model_names = ["test_67","test_68","test_71","test_69","test_72","test_73",
    #                "test_133","test_134","test_135","test_136","test_137",
    #                "test_114","test_115"]
    # model_types = [ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,
    #                ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,
    #                ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL]
    # chair_dataset = 11*[False]+2*[True]

    # model_names = ["test_123_1","test_124_1","test_126_1","test_127_1"]
    # model_types = [ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL]
    # chair_dataset = 4*[False]

    # model_names = ["test_112","test_100","test_108","test_72_2","test_71_2","test_73_2","test_69_2"]
    # model_types = [ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP]
    # chair_dataset = [True]+[False]+[True]+4*[False]

    # model_names = ["test_67_2","test_68_2","test_130_3"]
    # model_types = [ModelType.PIX2VOXELPP,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL]
    # chair_dataset = 3*[False]

    # model_names = ["test_141_","test_144","test_145","test_146","test_147"]
    # model_types = [ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL]
    # chair_dataset = [True]+ 4*[False]

    model_names = ["test_148","test_149","test_150","test_151"]
    model_types = [ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL,ModelType.PIX2VOXELPP,ModelType.PIX2VOXEL]
    chair_dataset = 4*[True]




    for model_name, model_type, chair in zip(model_names,model_types,chair_dataset):
        try:
            print("Testing for model name:" +model_name+" model type:"+model_type)
            config.pretrained_name = model_name
            config.main_name = model_name+"_testing_best"
            config.model_type = model_type

            config.tensorboard_train = config.output_path + config.main_name  + '/tensorboard/tensorboard_training/'
            config.tensorboard_validation = config.output_path + config.main_name  + '/tensorboard/tensorboard_validation/'
            config.checkpoint_path = config.output_path + config.main_name + "/"
            config.pretrained_checkpoint_path = config.output_path + config.pretrained_name + "/"

            config.pix3d.test_indices = config.root_path+'/Datasets/pix3d/splits/pix3d_test_chair.npy' if chair else config.root_path+'/Datasets/pix3d/splits/pix3d_test_2.npy'

            print("configuration: " + str(config))

            pipeline = PipelineManager(config).get_pipeline(pipeline_type=config.pipeline_type)
            pipeline.realdata_test()

            pipeline = None
        except Exception as ex:
            print("Exception!!!!!!!!!!!!!!!!!!!")
            print(ex)


