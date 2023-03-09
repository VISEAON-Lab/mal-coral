import torch
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator import RPN
from detectron2.modeling.proposal_generator.rpn_outputs import RPNOutputs, find_top_rpn_proposals
from fvcore.nn import smooth_l1_loss
import torch.nn.functional as F


@PROPOSAL_GENERATOR_REGISTRY.register()
class CoralRPN(RPN):
    def forward(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)

        if self.training:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
        else:
            gt_labels, gt_boxes = None, None

        outputs = CoralRPNOutputs(
            self.box2box_transform,
            self.batch_size_per_image,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            gt_labels,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )

        return proposals, losses

class CoralRPNOutputs(RPNOutputs):
    def rpn_losses(
            self, gt_labels, gt_anchor_deltas, pred_objectness_logits, pred_anchor_deltas, smooth_l1_beta
    ):
        """
        Args:
            gt_labels (Tensor): shape (N,), each element in {-1, 0, 1} representing
                ground-truth objectness labels with: -1 = ignore; 0 = not object; 1 = object.
            gt_anchor_deltas (Tensor): shape (N, box_dim), row i represents ground-truth
                box2box transform targets (dx, dy, dw, dh) or (dx, dy, dw, dh, da) that map anchor i to
                its matched ground-truth box.
            pred_objectness_logits (Tensor): shape (N,), each element is a predicted objectness
                logit.
            pred_anchor_deltas (Tensor): shape (N, box_dim), each row is a predicted box2box
                transform (dx, dy, dw, dh) or (dx, dy, dw, dh, da)
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.

        Returns:
            objectness_loss, localization_loss, both unnormalized (summed over samples).
        """
        pos_masks = gt_labels == 1
        localization_loss = smooth_l1_loss(
            pred_anchor_deltas[pos_masks], gt_anchor_deltas[pos_masks], smooth_l1_beta, reduction="sum"
        )

        objectness_loss = F.binary_cross_entropy_with_logits(
            pred_objectness_logits[pos_masks],
            gt_labels[pos_masks].to(torch.float32),
            reduction="sum",
        )
        return objectness_loss, localization_loss

