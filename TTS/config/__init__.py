import json
import os
import re
from typing import Dict

import fsspec
import yaml  # Ensure pyyaml is installed: pip install pyyaml
from coqpit import Coqpit  # Ensure coqpit is installed: pip install coqpit

# Import necessary shared configs and utility functions from TTS
from TTS.config.shared_configs import *
from TTS.utils.generic_utils import find_module

def read_json_with_comments(json_path):
    """For backward compatibility with JSON that may have comments."""
    # Read the content of the JSON file
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    
    # Handle comments (remove // and /* */) but retain the rest of the JSON format
    input_str = re.sub(r"(\"(?:[^\"\\]|\\.)*\")|(/\*(?:.|[\\n\\r])*?\*/)|(//.*)", lambda m: m.group(1) or m.group(2) or "", input_str)
    
    # Return the JSON data after removing comments
    return json.loads(input_str)

def register_config(model_name: str) -> Coqpit:
    """Find and return the correct configuration class for the given model name.

    Args:
        model_name (str): The name of the model to find the configuration for.

    Raises:
        ModuleNotFoundError: If no matching config is found.

    Returns:
        Coqpit: The configuration class for the model.
    """
    config_class = None
    config_name = model_name + "_config"

    # For the 'xtts' model, we have a specific config
    if model_name == "xtts":
        from TTS.tts.configs.xtts_config import XttsConfig
        config_class = XttsConfig
    
    # Search for the config in various paths
    paths = ["TTS.tts.configs", "TTS.vocoder.configs", "TTS.encoder.configs", "TTS.vc.configs"]
    for path in paths:
        try:
            config_class = find_module(path, config_name)
        except ModuleNotFoundError:
            pass  # Continue searching in other paths
    
    if config_class is None:
        raise ModuleNotFoundError(f" [!] Config for {model_name} cannot be found.")
    
    return config_class

def _process_model_name(config_dict: Dict) -> str:
    """Process and format the model name, removing unwanted suffixes.

    Args:
        config_dict (Dict): The configuration dictionary.

    Returns:
        str: The formatted model name.
    """
    model_name = config_dict["model"] if "model" in config_dict else config_dict["generator_model"]
    # Remove unwanted suffixes like _generator or _discriminator
    model_name = model_name.replace("_generator", "").replace("_discriminator", "")
    return model_name

def load_config(config_path: str) -> Coqpit:
    """Load a TTS config from a JSON or YAML file, and return the corresponding config object.

    Args:
        config_path (str): Path to the config file.

    Raises:
        TypeError: If the config file type is unknown.

    Returns:
        Coqpit: The TTS config object.
    """
    config_dict = {}
    ext = os.path.splitext(config_path)[1]
    
    # If it's a YAML file, load it as YAML
    if ext in (".yml", ".yaml"):
        with fsspec.open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    
    # If it's a JSON file, load it as JSON (handling comments if necessary)
    elif ext == ".json":
        try:
            with fsspec.open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError:
            # If JSON contains comments, use the read_json_with_comments function
            data = read_json_with_comments(config_path)
    else:
        raise TypeError(f" [!] Unknown config file type {ext}")
    
    config_dict.update(data)
    model_name = _process_model_name(config_dict)
    config_class = register_config(model_name.lower())
    
    # Instantiate and load the config class
    config = config_class()
    config.from_dict(config_dict)
    
    return config

def check_config_and_model_args(config, arg_name, value):
    """Check if the argument exists in either `config.model_args` or `config`."""
    if hasattr(config, "model_args"):
        if arg_name in config.model_args:
            return config.model_args[arg_name] == value
    if hasattr(config, arg_name):
        return config[arg_name] == value
    return False

def get_from_config_or_model_args(config, arg_name):
    """Get the given argument from `config.model_args` if it exists, or from `config`."""
    if hasattr(config, "model_args"):
        if arg_name in config.model_args:
            return config.model_args[arg_name]
    return config[arg_name]

def get_from_config_or_model_args_with_default(config, arg_name, def_val):
    """Get the given argument from `config.model_args` or `config`, returning a default value if it doesn't exist."""
    if hasattr(config, "model_args"):
        if arg_name in config.model_args:
            return config.model_args[arg_name]
    if hasattr(config, arg_name):
        return config[arg_name]
    return def_val
