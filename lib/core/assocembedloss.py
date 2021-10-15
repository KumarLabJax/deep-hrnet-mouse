import torch
import torch.nn as nn

"""
An implementation of the associative embedding loss function described in
"Associative Embedding: End-to-End Learning for Joint Detection and Grouping"
(Newell et al.)
"""

def _get_assoc_embed_values(assoc_embed_maps, pose_xy):
    """
    Extract the embedding values from the maps at the given
    XY pose coordinates.
    """

    pose_xy = pose_xy.round().long()

    instance_count, keypoint_count, xy_size = pose_xy.shape
    _, height, width = assoc_embed_maps.shape

    assert keypoint_count == 12
    assert xy_size == 2

    # we need to mask out points that are out of bounds
    keypoint_vis = (
        (pose_xy[..., 0] >= 0) & (pose_xy[..., 0] < width) &
        (pose_xy[..., 1] >= 0) & (pose_xy[..., 1] < height)
    )

    def gen_embed_vals():

        # TODO: there is probably a more efficient tensor indexing approach
        # to extract these values, but this should work

        for instance_i in range(instance_count):
            for keypoint_i in range(keypoint_count):
                if keypoint_vis[instance_i, keypoint_i].item():
                    curr_x, curr_y = pose_xy[instance_i, keypoint_i, :]
                    curr_embed = assoc_embed_maps[keypoint_i, curr_y, curr_x]
                    yield curr_embed
                else:
                    # since this point is out of bounds we just return zero,
                    # but respect the device and dtype of the map
                    yield torch.tensor(
                        0,
                        device=assoc_embed_maps.device,
                        dtype=assoc_embed_maps.dtype)

    embed_vals = torch.stack(list(gen_embed_vals()))
    embed_vals = embed_vals.reshape(instance_count, keypoint_count)

    return embed_vals, keypoint_vis


def _instance_grouping_term(embed_vals, keypoint_vis, reference_embeds):
    """
    This function implements the first sub-expression from the grouping loss
    in the associative embedding paper. This sub-expression incentivizes
    grouping within instances
    """
    instance_count = reference_embeds.size(0)

    if instance_count == 0:
        # there needs to be at least one instance for this term
        # to contribute to the loss
        return torch.tensor(
            0,
            device=reference_embeds.device,
            dtype=reference_embeds.dtype)

    else:
        squared_diffs = (reference_embeds.view(-1, 1) - embed_vals) ** 2
        squared_diffs[~keypoint_vis] = 0

        return torch.sum(squared_diffs) / instance_count


def _ref_embedding_separation_term(reference_embeds, sigma=1):

    """
    This function implements the second sub-expression from the grouping loss
    in the associative embedding paper. This sub-expression incentivizes
    separation between reference embedding values
    """

    instance_count = reference_embeds.size(0)

    if instance_count <= 1:
        # there needs to be at least two instances for this term
        # to contribute to the loss since it is based upon a difference
        # of reference embeddings.
        return torch.tensor(
            0,
            device=reference_embeds.device,
            dtype=reference_embeds.dtype)

    else:

        # calculate the squared difference between all combinations of the reference embedding
        ref_embed_combos = torch.combinations(reference_embeds, 2)
        num_combos = ref_embed_combos.size(0)
        ref_embed_combos[:, 1].mul_(-1)
        squared_diffs = ref_embed_combos.sum(1) ** 2

        # now we have squared diffs and we can calculate the sum of exponents
        # part of the subexpression
        sum_of_exps = torch.sum(torch.exp(-0.5 * sigma ** 2 * squared_diffs))

        # In the paper the denominator is N^2 because they calculate for all permutations.
        # Since we use all combinations rather than permutations we use a different
        # denominator to achieve the same result
        return sum_of_exps / (num_combos + instance_count / 2)


def balanced_bcelogit_loss(inf_maps, lbl_maps, fairness_quotient):

    assert 0.0 <= fairness_quotient <= 1.0

    total_len = 0
    for i, dim_len in enumerate(lbl_maps.size()):
        if i == 0:
            total_len = dim_len
        else:
            total_len *= dim_len

    raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        inf_maps,
        lbl_maps,
        reduction='none')

    # split the losses between true labels and false labels
    true_lbl_losses = raw_loss[lbl_maps == 1]
    true_lbl_count = len(true_lbl_losses)
    false_lbl_losses = raw_loss[lbl_maps == 0]
    false_lbl_count = len(false_lbl_losses)

    assert total_len == true_lbl_count + false_lbl_count

    if fairness_quotient == 0 or true_lbl_count == 0 or false_lbl_count == 0:
        # We've either been asked to apply zero fairness or
        # we're missing one of the true/false classes so we can't balance
        # the loss here
        return raw_loss.mean()
    elif fairness_quotient == 1:
        # return a balanced loss (where true and false cases contribute equally)
        return (true_lbl_losses.mean() + false_lbl_losses.mean()) / 2
    else:
        # we mix balanced and imbalanced losses according to the fairness quotient
        # TODO: there is a more efficient way to do this. We don't need to sum over
        #       the raw losses twice.
        balanced_loss = (true_lbl_losses.mean() + false_lbl_losses.mean()) / 2
        imbalanced_loss = raw_loss.mean()

        return balanced_loss * fairness_quotient + imbalanced_loss * (1.0 - fairness_quotient)



def weighted_bcelogit_loss(inf_maps, lbl_maps, pos_weight):

    total_len = 0
    for i, dim_len in enumerate(lbl_maps.size()):
        if i == 0:
            total_len = dim_len
        else:
            total_len *= dim_len

    raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        inf_maps,
        lbl_maps,
        reduction='none')

    # split the losses between true labels and false labels
    true_lbl_losses = raw_loss[lbl_maps == 1]
    true_lbl_count = len(true_lbl_losses)
    false_lbl_losses = raw_loss[lbl_maps == 0]
    false_lbl_count = len(false_lbl_losses)

    assert total_len == true_lbl_count + false_lbl_count

    if true_lbl_count == 0 or false_lbl_count == 0:
        # we're missing one of the true/false classes so we can't weight
        return raw_loss.mean()
    else:
        # return a weighted loss
        numerator = true_lbl_losses.sum() * pos_weight + false_lbl_losses.sum()
        denominator = true_lbl_count * pos_weight + false_lbl_count

        return numerator / denominator


class PoseEstAssocEmbedLoss(nn.Module):

    """
    Combines an MSE (L2) loss for the pose heatmaps with an associative embedding loss
    """

    def __init__(
            self,
            pose_heatmap_weight=1.0,
            assoc_embedding_weight=1.0,
            separation_term_weight=1.0,
            grouping_term_weight=1.0,
            sigma=1.0,
            pose_loss_func=None):

        super(PoseEstAssocEmbedLoss, self).__init__()

        self.pose_heatmap_weight = pose_heatmap_weight
        self.assoc_embedding_weight = assoc_embedding_weight
        self.separation_term_weight = separation_term_weight
        self.grouping_term_weight = grouping_term_weight
        self.sigma = sigma
        self.pose_loss_func = pose_loss_func

        self.loss_components = dict()

    def forward(self, inference_tensor, truth_labels):

        # we need to combine the embedding loss and the keypoint L2 loss
        est_device = inference_tensor.device

        # put all truth labels on the same device as the inference
        lbl_joint_heatmaps = truth_labels['joint_heatmaps'].to(
            device=est_device,
            non_blocking=True)
        lbl_pose_instances = truth_labels['pose_instances'].to(
            device=est_device,
            non_blocking=True)
        lbl_instance_count = truth_labels['instance_count'].to(
            device=est_device,
            non_blocking=True)
        pose_keypoint_count = lbl_joint_heatmaps.size(1)

        inf_joint_heatmaps = inference_tensor[:, :pose_keypoint_count, ...]
        inf_assoc_embed_map = inference_tensor[:, pose_keypoint_count:, ...]

        if self.pose_loss_func is not None:
            pose_loss = self.pose_loss_func(
                inf_joint_heatmaps,
                lbl_joint_heatmaps)
        else:
            pose_loss = nn.functional.mse_loss(
                inf_joint_heatmaps,
                lbl_joint_heatmaps)
        embed_loss = self.pose_assoc_embed_loss(
            inf_assoc_embed_map,
            lbl_pose_instances,
            lbl_instance_count)
        combined_loss = pose_loss * self.pose_heatmap_weight + embed_loss * self.assoc_embedding_weight

        self.loss_components['pose_loss'] = pose_loss.detach()
        self.loss_components['embed_loss'] = embed_loss.detach()
        self.loss_components['weighted_pose_loss'] = pose_loss.detach() * self.pose_heatmap_weight
        self.loss_components['weighted_embed_loss'] = embed_loss.detach() * self.assoc_embedding_weight
        self.loss_components['combined_loss'] = combined_loss.detach()

        return combined_loss

    def pose_assoc_embed_loss(
            self,
            batch_assoc_embed_maps,
            batch_target_poses_xy,
            batch_instance_counts):

        batch_size = batch_target_poses_xy.size(0)
        batch_losses = torch.zeros(
            batch_size,
            device=batch_assoc_embed_maps.device,
            dtype=batch_assoc_embed_maps.dtype)

        for sample_i in range(batch_size):

            # pull out the values corresponding to the current sample in the mini batch
            instance_count = batch_instance_counts[sample_i]
            assoc_embed_maps = batch_assoc_embed_maps[sample_i, ...]
            target_poses_xy = batch_target_poses_xy[sample_i, :instance_count, ...]

            # extract the embedding values at the "truth" XY coordinates for each
            # instance and calculate the reference embedding
            embed_vals, keypoint_vis = _get_assoc_embed_values(assoc_embed_maps, target_poses_xy)

            # remove instances with no visible keypoints
            keypoint_vis_counts = keypoint_vis.sum(1)
            visible_instances = keypoint_vis_counts > 0

            keypoint_vis_counts = keypoint_vis_counts[visible_instances]
            embed_vals = embed_vals[visible_instances, :]
            keypoint_vis = keypoint_vis[visible_instances, :]

            reference_embeds = embed_vals.sum(1) / keypoint_vis.sum(1).to(embed_vals)

            # we take the loss expression defined in the associative embedding paper
            # and break it down into two sub-expressions: an instance grouping part
            # and a reference embedding separation part. We also apply a weighting to
            # each of these loss components which is not in the paper's approach
            inst_grp_term = _instance_grouping_term(embed_vals, keypoint_vis, reference_embeds)
            if self.grouping_term_weight != 1:
                inst_grp_term *= self.grouping_term_weight

            sep_term = _ref_embedding_separation_term(reference_embeds, sigma=self.sigma)
            if self.separation_term_weight != 1:
                sep_term *= self.separation_term_weight

            batch_losses[sample_i] = inst_grp_term + sep_term

        return batch_losses.mean()
