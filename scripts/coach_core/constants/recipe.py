"""Fixed dimensions of the supported Qwen-compatible training recipe."""

TARGET_LAYERS = [1, 6, 11, 16, 22, 27, 32, 37]
FEATURE_LAYER_COUNT = len(TARGET_LAYERS)
HIDDEN_SIZE = 2048
FEATURE_WIDTH = FEATURE_LAYER_COUNT * HIDDEN_SIZE
FEATURE_BYTES_PER_TOKEN = FEATURE_WIDTH * 2
VOCAB_SIZE = 248320
MASK_TOKEN_ID = 248077
SELECTOR_TOP_K = 16
METRICS = ("loss", "ce", "selector_ce", "accuracy", "prefix_tokens", "top16_recall")
TRAINING_SEED = 731
UNIFORM_OBJECTIVE = "uniform"
DPACE_OBJECTIVE = "dpace"
TRAINING_OBJECTIVES = (UNIFORM_OBJECTIVE, DPACE_OBJECTIVE)
SELECTOR_LOSS_WEIGHT = 0.5
SELECTOR_IGNORE_INDEX = -100
DPACE_ALPHA = 0.5

DRAFT_MODEL_CONFIG = {
    "hidden_size": HIDDEN_SIZE,
    "intermediate_size": 6144,
    "num_hidden_layers": 6,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": VOCAB_SIZE,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 262144,
    "sliding_window": 4096,
    "use_sliding_window": True,
    "layer_types": ["sliding_attention"] * 6,
    "is_causal": False,
    "num_target_layers": 40,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1e7},
    "dflash_config": {
        "block_size": 16,
        "target_layer_ids": [x - 1 for x in TARGET_LAYERS],
        "mask_token_id": MASK_TOKEN_ID,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": SELECTOR_TOP_K,
    },
}
