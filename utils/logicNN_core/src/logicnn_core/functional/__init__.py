"""論理ゲートNNの独立した数学関数を公開する。"""

from .combinatorics import build_binomial_table, sample_unique_combinations, truth_table_from_id, unrank_combinations
from .logic import all_binary_logic_outputs, apply_binary_lut, apply_export_luts, apply_export_truth_tables
from .regularization import regularization_loss, rescale_weights_
from .sampling import gumbel_sigmoid, scale_gradient, temperature_sigmoid, temperature_softmax
from .walsh import fast_walsh_hadamard, light_basis, walsh_basis, weighted_light_basis_sum, weighted_walsh_basis_sum

__all__ = [
    "all_binary_logic_outputs",
    "apply_binary_lut",
    "apply_export_luts",
    "apply_export_truth_tables",
    "build_binomial_table",
    "fast_walsh_hadamard",
    "gumbel_sigmoid",
    "light_basis",
    "regularization_loss",
    "rescale_weights_",
    "sample_unique_combinations",
    "scale_gradient",
    "temperature_sigmoid",
    "temperature_softmax",
    "truth_table_from_id",
    "unrank_combinations",
    "walsh_basis",
    "weighted_light_basis_sum",
    "weighted_walsh_basis_sum",
]
