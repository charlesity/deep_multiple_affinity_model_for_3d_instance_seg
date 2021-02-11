
from confnets import MultiScaleInputMultiOutputUNet



class SamplingBayesianMultiScaleInputMultiOutputUNet(MultiScaleInputMultiOutputUNet):

    def __init__(self, *super_args, mask_decoders_kwargs=None, **super_kwargs):
        super(SamplingBayesianMultiScaleInputMultiOutputUNet, self).__init__(*super_args, **super_kwargs)


