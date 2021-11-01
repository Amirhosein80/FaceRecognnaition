import tensorflow as tf 
import torch

class TripletLossTF (tf.keras.losses.Loss):
    """
    claculate triplet loss for tensorflow freamwork.
    """
    def __init__(self, alpha=0.2):
        super(TripletLossTF, self).__init__()
        self.alpha = alpha

    def __call__(self, anchor_pred, postive_pred, negative_pred):
        distance_anc_pos = tf.norm((anchor_pred - postive_pred), axis=1)
        distance_anc_neg = tf.norm((anchor_pred - negative_pred), axis=1)
        return tf.reduce_sum(tf.maximum((distance_anc_pos - distance_anc_neg + self.alpha), 0))


class TripletLossTorch (torch.nn.Module):
    """
    claculate triplet loss for pytorch freamwork.
    """
    def __init__(self, alpha=0.2):
        super(TripletLossTorch, self).__init__()
        self.alpha = alpha 
        
    def forward (self, anchor_pred, postive_pred, negative_pred):
        distance_anc_pos = torch.nn.functional.pairwise_distance(anchor_pred, postive_pred)
        distance_anc_neg = torch.nn.functional.pairwise_distance(anchor_pred, negative_pred)
        return torch.sum(torch.maximum((distance_anc_pos - distance_anc_neg + self.alpha), torch.tensor(0).cuda()))