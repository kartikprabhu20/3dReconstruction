"""

    Created on 25/12/20 6:34 PM 
    @author: Kartik Prabhu

"""
import trimesh
import glob
import os


class Visualise():
    def __init__(self, file_path):
        self.file_path = file_path

    def __mesh__(self):
        mesh = trimesh.load(os.path.join(self.file_path, "bed/train/bed_0001.off"))
        # mesh.show()

        return mesh


