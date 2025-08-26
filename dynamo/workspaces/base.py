from accelerate import Accelerator
from datasets.core import TrajectoryDataset


class Workspace:
    def __init__(self, cfg, work_dir):
        self.cfg = cfg
        self.work_dir = work_dir
        self.accelerator = Accelerator()
        self.dataset: TrajectoryDataset = None

    def set_models(self, encoder, projector):
        self.encoder = encoder
        self.projector = projector

    def set_dataset(self, dataset):
        self.dataset = dataset

    def run_offline_eval(self):
        return {"loss": 0}
