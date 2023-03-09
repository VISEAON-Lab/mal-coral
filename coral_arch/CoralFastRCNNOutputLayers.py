from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from coral_arch.CoralFastRCNNOutputs import CoralFastRCNNOutputsImbalancedDataset


class CoralFastRCNNOutputLayers(FastRCNNOutputLayers):
    ""
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        return CoralFastRCNNOutputsImbalancedDataset(
            self.box2box_transform, scores, proposal_deltas, proposals, self.smooth_l1_beta
        ).losses()