def normalize_checkpoint_keys(state_dict):
    new_state_dict = {}  # Initialize a new dictionary to hold the updated keys
    for key, value in state_dict.items():
        # Strip the prefixes "model.transformer.", "transformer.", and leading "model."
        new_key = key.replace("model.transformer.", "").replace("transformer.", "").lstrip("model.")
        new_state_dict[new_key] = value
        # Maintain existing handling of .weight_2
        if ".weight_2" in new_key:
            new_state_dict[new_key] = value

    return new_state_dict  # Return the updated state_dict