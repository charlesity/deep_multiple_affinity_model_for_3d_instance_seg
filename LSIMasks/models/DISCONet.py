from torch import nn
from torch import randn
import numpy as np
from torch import randn, argmax, tensor, gather, zeros, stack, cat

from confnets.blocks import SamePadResBlock

#DISCONet inspired by
from LSIMasks.models.latent_mask_model import MaskDecoder

class DISCONet(MaskDecoder):
    def __init__(self, latent_variable_size=32,
                 mask_shape=(5, 29, 29),
                 feature_maps=16,
                 target_index=0,
                 crop_slice_prediction=None,
                 mask_dws_fact=(1, 1, 1),
                 pred_dws_fact=(1, 1, 1),
                 sample_strides=(1, 1, 1),
                 limit_nb_decoded_masks_to=None,
                 max_random_crop=(0, 0, 0),  nb_mask_output = 4, multiple_decoder_option=None, probabilistic_model = True):
        super(DISCONet, self).__init__(latent_variable_size
                                                  ,mask_shape
                                                  ,feature_maps
                                                  ,target_index
                                                  ,crop_slice_prediction
                                                  ,mask_dws_fact
                                                  ,pred_dws_fact
                                                  ,sample_strides
                                                  ,limit_nb_decoded_masks_to,
                                    max_random_crop)
        self.nb_mask_output = nb_mask_output
        self.first_blocks = nn.Sequential(SamePadResBlock(feature_maps, f_inner=feature_maps,
                                           f_out=feature_maps,
                                           dim=3,
                                           pre_kernel_size=(3, 3, 3),
                                           kernel_size=(3, 3, 3),
                                           activation="ReLU",
                                           normalization="GroupNorm",
                                           nb_norm_groups=1,
                                           apply_final_activation=False,
                                           apply_final_normalization=False),SamePadResBlock(feature_maps, f_inner=feature_maps,
                                           f_out=feature_maps,
                                           dim=3,
                                           pre_kernel_size=(3, 3, 3),
                                           kernel_size=(3, 3, 3),
                                           activation="ReLU",
                                           normalization="GroupNorm",
                                           nb_norm_groups=1,
                                           apply_final_activation=False,
                                           apply_final_normalization=False), SamePadResBlock(feature_maps, f_inner=feature_maps,
                                         f_out=feature_maps,
                                         dim=3,
                                         pre_kernel_size=(3, 3, 3),
                                         kernel_size=(3, 3, 3),
                                         activation="ReLU",
                                         normalization="GroupNorm",
                                         nb_norm_groups=1,
                                         apply_final_activation=False,
                                         apply_final_normalization=False))
        self.second_blocks = nn.Sequential(SamePadResBlock(2*feature_maps, f_inner=2*feature_maps,
                                           f_out=2*feature_maps,
                                           dim=3,
                                           pre_kernel_size=(3, 3, 3),
                                           kernel_size=(3, 3, 3),
                                           activation="ReLU",
                                           normalization="GroupNorm",
                                           nb_norm_groups=1,
                                           apply_final_activation=False,
                                           apply_final_normalization=False),
                                           SamePadResBlock(2*feature_maps, f_inner=2*feature_maps,
                                           f_out=2*feature_maps,
                                           dim=3,
                                           pre_kernel_size=(3, 3, 3),
                                           kernel_size=(3, 3, 3),
                                           activation="ReLU",
                                           normalization="GroupNorm",
                                           nb_norm_groups=1,
                                           apply_final_activation=False,
                                           apply_final_normalization=False),
                                           SamePadResBlock(2*feature_maps, f_inner=2*feature_maps,
                                         f_out=1,
                                         dim=3,
                                         pre_kernel_size=(3, 3, 3),
                                         kernel_size=(3, 3, 3),
                                         activation="ReLU",
                                         normalization="GroupNorm",
                                         nb_norm_groups=1,
                                         apply_final_activation=False,
                                         apply_final_normalization=False))
        self.final_activation = nn.Sigmoid()

    def forward(self, encoded_variable):
        out_linear = self.linear_base(encoded_variable)
        N = out_linear.shape[0]
        reshaped = out_linear.view(N, -1, *self.min_path_shape)

        out = self.first_blocks(reshaped)
        z = randn((out.shape)).cuda()
        cat_inputs = cat((out, z), 1)
        out = self.second_blocks(cat_inputs)
        out = self.final_activation(out)
        return out