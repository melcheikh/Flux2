import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

repo_id = "black-forest-labs/FLUX.2-dev-NVFP4"
filename = "flux2-dev-nvfp4-mixed.safetensors"

try:
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    sd = load_file(path)
    q_keys = [k for k, v in sd.items() if v.dtype == torch.uint8]
    logger.info(f"Total uint8 keys: {len(q_keys)}")

    prefixes = {}
    for k in q_keys:
        p = ".".join(k.split(".")[:2])
        prefixes[p] = prefixes.get(p, 0) + 1

    logger.info("Keys by prefix:")
    for p, count in sorted(prefixes.items()):
        logger.info(f"  {p}: {count}")

    logger.info("\nTop-level keys (All):")
    for k in sorted(sd.keys()):
        if "." not in k or k.split(".")[0] not in ["double_blocks", "single_blocks"]:
            logger.info(f"  {k:50} | {sd[k].dtype} | {list(sd[k].shape)}")

except Exception as e:
    logger.error(f"Error: {e}")
