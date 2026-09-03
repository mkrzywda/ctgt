"""Search-space constants and leaderboard mappings."""

CONV_NAMES = ['GCN', 'GraphSAGE', 'GAT', 'GIN', 'SG']
HEAD_OPTIONS = [1, 2, 4, 8]
INIT_OPTIONS = ['uniform', 'xavier', 'kaiming', 'normal']
MIN_LAYERS, MAX_LAYERS = 1, 6
HID_MIN, HID_MAX = 16, 256
DROP_MIN, DROP_MAX = 0.0, 0.8
LR_MIN, LR_MAX = 1e-3, 1e-3
MAX_CLS_LAYERS = 3
SURROGATE_FEATURE_DIM = 16

LEADERBOARD_BENCHMARK_MAP = {
    'metal_opt':     'dft_3d_optb88vdw_bandgap',
    'metal_mbj':     'dft_3d_mbj_bandgap',
    'magnetic':      'dft_3d_magmom_oszicar',
    'slme':          'dft_3d_slme',
    'spillage':      'dft_3d_spillage',
    'seebeck_p':     'dft_3d_p_Seebeck',
    'powerfactor_n': 'dft_3d_n_powerfact',
}
