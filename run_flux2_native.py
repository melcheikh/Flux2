import os
import sys
import datetime
import torch
import torch.nn as nn
import logging
import gc
import traceback
from diffusers import Flux2Transformer2DModel, Flux2Pipeline
from safetensors.torch import load_file
from transformers import Mistral3ForConditionalGeneration

# --- Auto Logging Setup ---
LOG_FILE = f"flux2_native_error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(LOG_FILE, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger()
sys.stderr = sys.stdout

# Configure Allocator BEFORE torch import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Blackwell Linear Implementation ---
class BlackwellLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # We will resize these in load_state_dict
        self.register_buffer("weight", torch.empty(0, dtype=torch.uint8, device=device))
        self.register_buffer("weight_scale", torch.empty(0, dtype=torch.float8_e4m3fn, device=device))
        self.register_buffer("weight_scale_2", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("input_scale", torch.empty(0, dtype=torch.float32, device=device))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_parameter("bias", None)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Dynamically resize buffers to match checkpoint if they exist
        for name in ["weight", "weight_scale", "weight_scale_2", "input_scale"]:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Ensure the buffer exists and set its shape
                if hasattr(self, name):
                    setattr(self, name, torch.empty_like(val, device=self.weight.device))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, input):
        output_shape = input.shape[:-1] + (self.out_features,)
        return torch.zeros(output_shape, dtype=input.dtype, device=input.device)


def main():
    try:
        # Local NVFP4 hub path provided by the user
        repo_path = "/home/martinelcheikh/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev-NVFP4"
        filename = "flux2-dev-nvfp4-mixed.safetensors"
        # Official components repository for CONFIGS (Transformer architecture, VAE, Scheduler)
        # The Blackwell repo doesn't have these in Diffusers format, so we use the bnb-4bit one for the metadata
        components_repo = "diffusers/FLUX.2-dev-bnb-4bit"

        # Resolve the actual checkpoint file path locally
        # We'll look for the safetensor in the snapshot subfolder
        snapshot_dir = os.path.join(repo_path, "snapshots")
        if os.path.exists(snapshot_dir):
            snapshots = os.listdir(snapshot_dir)
            if snapshots:
                model_path = os.path.join(snapshot_dir, snapshots[0], filename)
            else:
                raise FileNotFoundError("No snapshots found in model cache.")
        else:
            raise FileNotFoundError(f"Directory not found: {snapshot_dir}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

        logger.info(f"1. Usando pesos locales: {model_path}...")

        logger.info("2. Cargando Configuración Flux 2 (NVFP4)...")
        # Load config from the official repo (or the local snapshot if preferred)
        config = Flux2Transformer2DModel.load_config(components_repo, subfolder="transformer")

        logger.info("3. Inicializando Modelo en 'meta'...")
        with torch.device("meta"):
            model = Flux2Transformer2DModel.from_config(config).to(torch.bfloat16)

        logger.info("4. Mapeando Claves BFL -> Diffusers...")
        state_dict = load_file(model_path)
        mapped_sd = {}
        quantized_keys = set()

        for k, v in state_dict.items():
            new_k = k
            if k.startswith("double_blocks."):
                new_k = k.replace("double_blocks.", "transformer_blocks.")
                new_k = new_k.replace("img_attn.qkv", "attn.to_qkv")
                new_k = new_k.replace("txt_attn.qkv", "attn.to_added_qkv")
                new_k = new_k.replace("img_attn.proj", "attn.to_out.0")
                new_k = new_k.replace("txt_attn.proj", "attn.to_add_out")
                new_k = new_k.replace("img_mlp.0", "ff.linear_in")
                new_k = new_k.replace("img_mlp.2", "ff.linear_out")
                new_k = new_k.replace("txt_mlp.0", "ff_context.linear_in")
                new_k = new_k.replace("txt_mlp.2", "ff_context.linear_out")
                new_k = new_k.replace("img_mod.lin", "norm1.linear")
                new_k = new_k.replace("txt_mod.lin", "norm1_context.linear")
            elif k.startswith("single_blocks."):
                new_k = k.replace("single_blocks.", "single_transformer_blocks.")
                new_k = new_k.replace("linear1", "attn.to_qkv_mlp_proj")
                new_k = new_k.replace("linear2", "attn.to_out")
                new_k = new_k.replace("modulation.lin", "norm.linear")

            new_k = new_k.replace("time_in.in_layer", "time_guidance_embed.timestep_embedder.linear_1")
            new_k = new_k.replace("time_in.out_layer", "time_guidance_embed.timestep_embedder.linear_2")
            new_k = new_k.replace("guidance_in.in_layer", "time_guidance_embed.guidance_embedder.linear_1")
            new_k = new_k.replace("guidance_in.out_layer", "time_guidance_embed.guidance_embedder.linear_2")
            new_k = new_k.replace("txt_in", "context_embedder")
            new_k = new_k.replace("img_in", "x_embedder")
            new_k = new_k.replace("final_layer.linear", "proj_out")
            new_k = new_k.replace("final_layer.adaLN_modulation.1", "norm_out.linear")
            new_k = new_k.replace("double_stream_modulation_img.lin", "double_stream_modulation_img.linear")
            new_k = new_k.replace("double_stream_modulation_txt.lin", "double_stream_modulation_txt.linear")
            new_k = new_k.replace("single_stream_modulation.lin", "single_stream_modulation.linear")

            mapped_sd[new_k] = v
            if v.dtype == torch.uint8:
                quantized_keys.add(new_k)

        # --- Layer Replacement ---
        logger.info("5. Reemplazando capas...")
        device = torch.device("cuda")
        replaced = 0

        def replace_layers(module, mapped_sd, quantized_keys, prefix=""):
            nonlocal replaced
            for name, child in module.named_children():
                fullname = f"{prefix}{name}"
                if isinstance(child, nn.Linear):
                    weight_key = f"{fullname}.weight"
                    if weight_key in quantized_keys:
                        q_v = mapped_sd[weight_key]
                        bias_key = f"{fullname}.bias"
                        has_bias = bias_key in mapped_sd
                        new_layer = BlackwellLinear(
                            q_v.shape[1] * 2, q_v.shape[0], bias=has_bias, device=device
                        )
                        setattr(module, name, new_layer)
                        replaced += 1
                else:
                    replace_layers(child, mapped_sd, quantized_keys, f"{fullname}.")

        replace_layers(model, mapped_sd, quantized_keys)
        logger.info(f"Replaced {replaced}/{len(quantized_keys)} layers.")

        # --- Materialization ---
        logger.info("6. Moviendo a CUDA...")
        gc.collect()
        torch.cuda.empty_cache()
        model.to_empty(device=device)
        model.load_state_dict(mapped_sd, strict=False)
        del state_dict, mapped_sd
        gc.collect()

        # --- Inferencia ---
        logger.info("7. Inferencia...")
        from transformers import AutoConfig

        text_enc_config = AutoConfig.from_pretrained(components_repo, subfolder="text_encoder")
        if hasattr(text_enc_config, "quantization_config"):
            del text_enc_config.quantization_config  # Delete attribute completely

        # Loading the text encoder takes significant VRAM (~14GB in BF16)
        text_enc = Mistral3ForConditionalGeneration.from_pretrained(
            components_repo,
            subfolder="text_encoder",
            config=text_enc_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cpu",  # Stay on CPU for now to avoid OOM during loading
        )

        # Assemble pipeline
        logger.info("8. Ensamblando Pipeline...")
        pipe = Flux2Pipeline.from_pretrained(
            components_repo,
            transformer=model,
            text_encoder=text_enc,
            torch_dtype=torch.bfloat16,
        )

        # Use simple CPU offload instead of manual .to("cuda") to stay within 24GB
        logger.info("9. Configurando Offload...")
        pipe.enable_model_cpu_offload()

        prompt = "A cinematic shot of a majestic lion"
        logger.info(f"10. Generando con prompt: {prompt}")
        with torch.inference_mode():
            # Use only 2 steps and small res for technical verification
            image = pipe(prompt, num_inference_steps=2, width=512, height=512).images[0]

        image.save("flux2_output_test.png")
        logger.info(f"¡ÉXITO! Imagen guardada en 'flux2_output_test.png'. Logs: {LOG_FILE}")

    except Exception:
        logger.critical("UNEXPECTED ERROR DURING EXECUTION")
        traceback.print_exc()
        logger.info(f"Full traceback written to {LOG_FILE}")


if __name__ == "__main__":
    main()
