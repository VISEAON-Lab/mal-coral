from detectron2.engine.defaults import DefaultTrainer
from coral_arch.CoralMapDataset import coral_build_detection_train_loader
from detectron2.data.build import build_detection_train_loader
import logging
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel
from detectron2.checkpoint import DetectionCheckpointer

from config import *

class CoralDefaultTrainer(DefaultTrainer):
    # def __init__(self, cfg):
    #     """
    #     Args:
    #         cfg (CfgNode):
    #     """
    #     self.cfg = cfg.clone() #new
    #     logger = logging.getLogger("detectron2")
    #     if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
    #         setup_logger()
    #     # Assume these objects must be constructed in this order.
    #     model = self.build_model(self.cfg) #model = self.build_model(cfg)
    #     optimizer = self.build_optimizer(self.cfg, model) #optimizer = self.build_optimizer(cfg, model)
    #     data_loader = self.build_train_loader(self.cfg) #data_loader = self.build_train_loader(cfg)
    #
    #     # For training, wrap with DDP. But don't need this for inference.
    #     if comm.get_world_size() > 1:
    #         model = DistributedDataParallel(
    #             model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
    #         )
    #     super().__init__(model, data_loader, optimizer)
    #
    #     self.scheduler = self.build_lr_scheduler(cfg, optimizer)
    #     # Assume no other objects need to be checkpointed.
    #     # We can later make it checkpoint the stateful hooks
    #     self.checkpointer = DetectionCheckpointer(
    #         # Assume you want to save checkpoints together with logs/statistics
    #         model,
    #         cfg.OUTPUT_DIR,
    #         optimizer=optimizer,
    #         scheduler=self.scheduler,
    #     )
    #     self.start_iter = 0
    #     self.max_iter = cfg.SOLVER.MAX_ITER
    #
    #     self.register_hooks(self.build_hooks())


    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if TRAIN_WITH_MASK:
            return coral_build_detection_train_loader(cfg)
        else:
            return build_detection_train_loader(cfg)