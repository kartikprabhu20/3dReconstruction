"""

    Created on 01/06/21 10:57 AM
    @author: Kartik Prabhu

"""
from DatasetManager import DatasetManager
from ModelManager import ModelManager
from ReconstructionPipeline import ReconstructionPipeline

class PipelineType:
    RECONSTRUCTION = "RECONSTRUCTION"
    AUTOENCODER = "AUTOENCODER"

class PipelineManager:

    def __init__(self, config):
        self.config = config
        self.datasetManager = DatasetManager(config)
        self.model, self.optimizer = ModelManager(config).get_Model(config.model_type)


    def get_pipeline(self, pipeline_type):
        switcher = {
            PipelineType.RECONSTRUCTION: ReconstructionPipeline(model=self.model, optimizer=self.optimizer,config=self.config, datasetManager=self.datasetManager),
            # PipelineType.AUTOENCODER:
        }
        return switcher.get(pipeline_type, ReconstructionPipeline(model=self.model, optimizer=self.optimizer,config=self.config, datasetManager=self.datasetManager))

