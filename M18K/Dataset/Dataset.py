from roboflow import Roboflow

class M18KDataset:
    def __int__(self):
        pass

    def download(self):

        rf = Roboflow(api_key="kTlC80dqA3CIoOuv8wob")
        project = rf.workspace("university-of-houston").project("mushroom-detection-snnme")
        dataset = project.version(1).download("coco")

        pass