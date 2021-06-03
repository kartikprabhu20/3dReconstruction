"""

    Created on 01/06/21 11:59 AM
    @author: Kartik Prabhu

"""
import Baseconfig
from PipelineManager import PipelineManager

if __name__ == '__main__':

    config = Baseconfig.config
    print("configuration: " + str(config))

    pipeline = PipelineManager(config).get_pipeline(pipeline_type=config.pipeline_type)
    pipeline.train()
