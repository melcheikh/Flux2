import torch

# Function to normalize NVFP4 keys
def normalize_keys(pipeline):
    # Update weight_scale_2 to weight_scale
    if 'weight_scale_2' in pipeline:
        pipeline['weight_scale'] = pipeline.pop('weight_scale_2')
    # Update weight_2 to weight
    if 'weight_2' in pipeline:
        pipeline['weight'] = pipeline.pop('weight_2')
    return pipeline

# Load the pipeline on CPU
pipeline = load_pipeline()
pipeline = normalize_keys(pipeline)
device_map = "cpu"
pipeline.to(device_map)

# Empty CUDA memory
torch.cuda.empty_cache()

# Meta/to_empty flow logic
# Retaining existing logic if needed
# Continue with the rest of the program logic as it is
# ...
# more code ...
