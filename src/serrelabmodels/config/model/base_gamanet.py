name: base_gammanet
import_prepath: serrelabmodels.models.base_gammanet
import_class: BaseGN
args:
  base_ff: -model.vgg_16
  gn_params: [['conv2_2', 3],['conv3_3', 3],['conv4_3', 3],['conv5_3', 1],['conv4_3', 1],['conv3_3', 1],['conv2_2', 1]]
  timesteps: 8
  hidden_init: 'identity'
  attention: 'gala' # 'se', None
  attention_layers: 1 #2
  saliency_filter_size: 3 #5
  norm_attention: False
  normalization_fgru: InstanceNorm2d
  normalization_fgru_params: {'affine': True}
  normalization_gate: InstanceNorm2d
  normalization_gate_params: {'affine': True}
  force_alpha_divisive: False
  force_non_negativity: True
  multiplicative_excitation: True
  ff_non_linearity: 'ReLU'
  us_resize_before_block: True
  readout: True
  readout_feats: 1