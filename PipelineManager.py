"""

    Created on 01/06/21 10:57 AM
    @author: Kartik Prabhu

"""
from DatasetManager import DatasetManager
from ModelManager import ModelManager
from ReconstructionPipeline import ReconstructionPipeline
from Synth2RealPipeline import Synth2RealPipeline


class PipelineType:
    RECONSTRUCTION = "RECONSTRUCTION"
    AUTOENCODER = "AUTOENCODER"
    CYCLEGAN = "CYCLEGAN"

class PipelineManager:

    def __init__(self, config):
        self.config = config
        self.datasetManager = DatasetManager(config)
        # self.model, self.optimizer = ModelManager(config).get_Model(config.model_type)


    def get_pipeline(self, pipeline_type):

        if pipeline_type == PipelineType.RECONSTRUCTION:
            return  ReconstructionPipeline(config=self.config, datasetManager=self.datasetManager)
        elif pipeline_type == PipelineType.CYCLEGAN:
            return Synth2RealPipeline(config=self.config, datasetManager=self.datasetManager)

        return  ReconstructionPipeline(config=self.config, datasetManager=self.datasetManager)
