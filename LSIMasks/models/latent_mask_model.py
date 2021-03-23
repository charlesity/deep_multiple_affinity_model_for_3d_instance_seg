
from torch import nn
import numpy as np
from torch import randn, argmax, tensor, gather, zeros, stack, cat

from confnets.blocks import SamePadResBlock
from confnets.models import MultiScaleInputMultiOutputUNet
from copy import deepcopy
from pydoc import locate


from torchsummary import summary



class LatentMaskModel(MultiScaleInputMultiOutputUNet):
    """
    Basically a UNet model (allowing to predict embeddings at different depth levels)
    with few additional small MaskDecoders.
    """
    def __init__(self, *super_args, mask_decoders_kwargs=None, **super_kwargs):
        super(LatentMaskModel, self).__init__(*super_args, **super_kwargs)

        assert mask_decoders_kwargs is not None, "Specs for mask-decoders are required."
        global_kwargs = mask_decoders_kwargs.pop("global", {})
        self.nb_mask_decoders = nb_mask_decoders = len(mask_decoders_kwargs.keys())
        all_mask_decoder_kwargs = [deepcopy(global_kwargs) for _ in range(nb_mask_decoders)]

        assert len(all_mask_decoder_kwargs) == len(self.output_branches_specs), \
            "The passed mask_decoders_kwargs do not match with the number of output branches in the UNet model"

        for i in range(nb_mask_decoders):
            all_mask_decoder_kwargs[i].update(mask_decoders_kwargs[i])

            # Deduce the size of the latent_variable from the output_size of the UNet branch:
            all_mask_decoder_kwargs[i]["latent_variable_size"] = self.output_branches_specs[i]["out_channels"]

        # Build one Mask decoder per branch:
        self.mask_decoders = nn.ModuleList([
            MaskDecoder(**kwgs) for kwgs in all_mask_decoder_kwargs
        ])

class MaskDecoder(nn.Module):
    def __init__(self,
                 latent_variable_size=32,
                 mask_shape=(5, 29, 29),
                 feature_maps=16,
                 target_index=0,
                 crop_slice_prediction=None,
                 mask_dws_fact=(1, 1, 1),
                 pred_dws_fact=(1, 1, 1),
                 sample_strides=(1, 1, 1),
                 limit_nb_decoded_masks_to=None,
                 max_random_crop=(0, 0, 0)
                 ):
        """

        :param latent_variable_size:
        :param mask_shape:
        :param feature_maps: Feature maps of the convolutions in the resBlock of the decoder

        :param target_index:
                    Index of the associated target tensor loaded in the batch-list

        :param crop_slice_prediction:
                    String specifying if we should crop part of embedding-tensor predicted by the UNet
                    to avoid training on borders.

        :param mask_dws_fact:
                    Downscaling factor of the mask (with respect to the chosen target!)

        :param pred_dws_fact:
                     Downscaling factor of the predicted embedding (with respect to the given target!)
                     (This in most of the cases is 1)

        :param sample_strides: (list of tuples)
                     How often we should sample embedding vectors (to be decoded) in the output tensor from the UNet
                     (These numbers are given in the resolution of the chosen target)

        :param limit_nb_decoded_masks_to: (list of lists)
                    Randomly pick only a portion of the sampled embedding vectors. Both 'factor' and 'number' modes
                    are available. I can specify a list of values, if `sample_strides` is a list, one per stride.
                    Example: [[120, 'number']]

        :param max_random_crop: (tuple)
                    Apply a very small random crop to the embedding tensor from the Unet, to avoid artifacts due to the
                    striding happening always on the same pixels

        """
        super(MaskDecoder, self).__init__()

        mask_shape = tuple(mask_shape) if isinstance(mask_shape, list) else mask_shape

        mask_shape = tuple(mask_shape) if isinstance(mask_shape, list) else mask_shape
        assert isinstance(mask_shape, tuple)
        self.output_shape = mask_shape
        assert all(sh % 2 == 1 for sh in mask_shape), "Patch should have even dimensions"

        self.min_path_shape = mask_shape
        self.vectorized_shape = np.array(self.min_path_shape).prod()

        # Build layers:
        self.latent_variable_size = latent_variable_size
        self.linear_base = nn.Linear(latent_variable_size, self.vectorized_shape * feature_maps)

        self.feature_maps = feature_maps
        self.decoder_module = SamePadResBlock(feature_maps, f_inner=feature_maps,
                                              f_out=1,
                                              dim=3,
                                              pre_kernel_size=(3, 3, 3),
                                              kernel_size=(3, 3, 3),
                                              activation="ReLU",
                                              normalization="GroupNorm",
                                              nb_norm_groups=1,
                                              apply_final_activation=False,
                                              apply_final_normalization=False)
        self.final_activation = nn.Sigmoid()

        self.mask_shape = mask_shape
        assert all(i % 2 == 1 for i in mask_shape), "Mask shape should be odd"

        # Validate the rest of the parameters (only used during training):
        mask_dws_fact = tuple(mask_dws_fact) if isinstance(mask_dws_fact, list) else mask_dws_fact
        assert isinstance(mask_dws_fact, tuple)
        pred_dws_fact = tuple(pred_dws_fact) if isinstance(pred_dws_fact, list) else pred_dws_fact
        assert isinstance(pred_dws_fact, tuple)
        sample_strides = tuple(sample_strides) if isinstance(sample_strides, list) else sample_strides
        assert isinstance(sample_strides, tuple)
        max_random_crop = tuple(max_random_crop) if isinstance(max_random_crop, list) else max_random_crop
        assert isinstance(max_random_crop, tuple)

        self.target_index = target_index
        self.crop_slice_prediction = crop_slice_prediction
        self.mask_dws_fact = mask_dws_fact
        self.pred_dws_fact = pred_dws_fact
        self.sample_strides = sample_strides
        self.limit_nb_decoded_masks_to = limit_nb_decoded_masks_to
        self.max_random_crop = max_random_crop

    def forward(self, encoded_variable):
        x = self.linear_base(encoded_variable)
        N = x.shape[0]
        reshaped = x.view(N, -1, *self.min_path_shape)

        out = self.decoder_module(reshaped)
        out = self.final_activation(out)
        return out


# ----------------------------------
# Multiple output latent mask model
# ----------------------------------
class MultiOutputLatentMaskModel(MultiScaleInputMultiOutputUNet):
    def __init__(self, *super_args, multiple_output_class = '', mask_decoders_kwargs=None, **super_kwargs):
        super(MultiOutputLatentMaskModel, self).__init__(*super_args, **super_kwargs)

        assert mask_decoders_kwargs is not None, "Specs for mask-decoders are required."
        global_kwargs = mask_decoders_kwargs.pop("global", {})
        self.nb_mask_decoders = nb_mask_decoders = len(mask_decoders_kwargs.keys())
        all_mask_decoder_kwargs = [deepcopy(global_kwargs) for _ in range(nb_mask_decoders)]

        assert len(all_mask_decoder_kwargs) == len(self.output_branches_specs), \
            "The passed mask_decoders_kwargs do not match with the number of output branches in the UNet model"

        
        for i in range(nb_mask_decoders):
            all_mask_decoder_kwargs[i].update(mask_decoders_kwargs[i])

            # Deduce the size of the latent_variable from the output_size of the UNet branch:
            all_mask_decoder_kwargs[i]["latent_variable_size"] = self.output_branches_specs[i]["out_channels"]


        if multiple_output_class =='':
            # Build one Mask decoder per branch:
            self.mask_decoders = nn.ModuleList([
                MultipleOutputMaskDecoder(**kwgs) for kwgs in all_mask_decoder_kwargs
            ])
        else:
            # use probabilistic model specified in config
            class_ = locate(multiple_output_class)
            self.mask_decoders = nn.ModuleList([class_(**kwgs) for kwgs in all_mask_decoder_kwargs]) #only one will be available

            


class MultipleOutputMaskDecoder(MaskDecoder):
    def __init__(self,latent_variable_size=32,
                 mask_shape=(5, 29, 29),
                 feature_maps=16,
                 target_index=0,
                 crop_slice_prediction=None,
                 mask_dws_fact=(1, 1, 1),
                 pred_dws_fact=(1, 1, 1),
                 sample_strides=(1, 1, 1),
                 limit_nb_decoded_masks_to=None,
                 max_random_crop=(0, 0, 0), nb_mask_output = 4, probabilistic_model = False, multiple_decoder_option = 'channel'):
        super(MultipleOutputMaskDecoder, self).__init__(latent_variable_size
                                                  ,mask_shape
                                                  ,feature_maps
                                                  ,target_index
                                                  ,crop_slice_prediction
                                                  ,mask_dws_fact
                                                  ,pred_dws_fact
                                                  ,sample_strides
                                                  ,limit_nb_decoded_masks_to,
                                    max_random_crop)

        self.probabilistic_model = probabilistic_model
        self.multiple_decoder_option = multiple_decoder_option
        self.nb_mask_output = nb_mask_output

        if self.multiple_decoder_option == "channels":
            self.multiple_decoders = SamePadResBlock(self.feature_maps, f_inner=self.feature_maps,
                                                     f_out=nb_mask_output,
                                                     dim=3,
                                                     pre_kernel_size=(3, 3, 3),
                                                     kernel_size=(3, 3, 3),
                                                     activation="ReLU",
                                                     normalization="GroupNorm",
                                                     nb_norm_groups=1,
                                                     apply_final_activation=False,
                                                     apply_final_normalization=False)
            rd_input = randn([1, feature_maps, *mask_shape]) # use this to obtain the shape of tensor
            output_shape = self.multiple_decoders(rd_input).shape[1:]
            input_features_weight = np.prod(output_shape)
            self.prediction_weights = nn.Sequential(nn.Linear(input_features_weight, input_features_weight),nn.ReLU(),
                                                    nn.Linear(input_features_weight, nb_mask_output), nn.Softmax(dim=1))
        elif self.multiple_decoder_option == "branches":
            self.multiple_decoders = nn.ModuleList([SamePadResBlock(self.feature_maps, f_inner=self.feature_maps,
                                                      f_out=1,
                                                      dim=3,
                                                      pre_kernel_size=(3, 3, 3),
                                                      kernel_size=(3, 3, 3),
                                                      activation="ReLU",
                                                      normalization="GroupNorm",
                                                      nb_norm_groups=1,
                                                      apply_final_activation=False,
                                                      apply_final_normalization=False) for i in range(nb_mask_output)])

            rd_input = randn([1, feature_maps, *mask_shape]) # use this to obtain the shape of tensor
            output_shape = np.array(self.multiple_decoders[0](rd_input).shape[1:])
            output_shape[0] = len(self.multiple_decoders)
            input_features_weight = np.prod(output_shape)
            self.prediction_weights = nn.Sequential(nn.Linear(input_features_weight, input_features_weight),nn.ReLU(),
                                                    nn.Linear(input_features_weight, nb_mask_output), nn.Softmax(dim=1))
        #Bayesian option
        else:
            raise Exception ("Invalid parameter", "Parameters must be channels or branches")
        self.final_decoder_activation = nn.Sigmoid()


    def forward(self, encoded_variable, al_mode = False):
        out_linear = self.linear_base(encoded_variable)
        N = out_linear.shape[0]
        reshaped = out_linear.view(N, -1, *self.min_path_shape)
        if self.multiple_decoder_option == "channels":
            channel_based_m_outputs = self.multiple_decoders(reshaped)
            channel_based_m_outputs = self.final_decoder_activation(channel_based_m_outputs)
            reshape2 = channel_based_m_outputs.view(N, -1)
            predict_weights_prob = self.prediction_weights(reshape2)
            if self.training:
                return [channel_based_m_outputs, predict_weights_prob]
            else:
                if al_mode: #still return both outputs for active learing validation mode
                    return [channel_based_m_outputs, predict_weights_prob]
                else:
                    index = argmax(predict_weights_prob, dim=1)
                    out = zeros(N, 1, channel_based_m_outputs.shape[2], channel_based_m_outputs.shape[3],
                                channel_based_m_outputs.shape[4]).cuda()
                    for i, idx in enumerate(index):
                        out[i, 0, :,:,: ] = channel_based_m_outputs[i, idx, :,:,:]
                    return [out]
        elif self.multiple_decoder_option == "branches":
            m_outputs = [self.final_decoder_activation(decoder(reshaped)) for decoder in self.multiple_decoders]
            branch_based_m_outputs = cat(m_outputs, dim=1)  #concatenate along channels
            reshaped2 = branch_based_m_outputs.view(branch_based_m_outputs.shape[0], -1)
            predict_weights_prob = self.prediction_weights(reshaped2)
            if self.training:
                return [branch_based_m_outputs, predict_weights_prob]
            else:
                if al_mode: #still return both outputs for active learing validation mode
                    return [branch_based_m_outputs, predict_weights_prob]
                else:
                    index = argmax(predict_weights_prob, dim=1)
                    out = zeros(N, 1, branch_based_m_outputs.shape[2], branch_based_m_outputs.shape[3],
                                branch_based_m_outputs.shape[4]).cuda()
                    for i, idx in enumerate(index):
                        out[i, 0, :, :, :] = branch_based_m_outputs[i, idx, :, :, :]
                    return [out]
        else:
            raise Exception("Multiple prediction option must be 'channels' or 'branches'. Update the yml file")

                




