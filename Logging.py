"""

    Created on 03/05/21 6:18 PM 
    @author: Kartik Prabhu

"""

import logging
import os

class Logger:
    def __init__(self, model_name, logger_path):
        if not os.path.exists(logger_path):
            os.mkdir(logger_path)

        logger_path += "/" + model_name+".log"
        self.logger = logging.getLogger(model_name)
        hdlr = logging.FileHandler(logger_path)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.DEBUG)

    def get_logger(self):
        return self.logger