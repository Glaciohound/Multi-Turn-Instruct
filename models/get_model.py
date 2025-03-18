from .openai_model import OpenAIModel
from .bedrock_model import AnthropicModel, LlamaModel, MistralModel


def get_model(model_name, openai_key, region):
    if model_name in OpenAIModel.model_name_list:
        return OpenAIModel(model_name, openai_key)
    elif model_name.startswith("anthropic."):
        return AnthropicModel(model_name, region)
    # elif model_name in LlamaModel.model_name_list:
    elif model_name.startswith("meta."):
        return LlamaModel(model_name, region)
    # elif model_name in MistralModel.model_name_list:
    elif model_name.startswith("mistral."):
        return MistralModel(model_name, region)
    else:
        raise ValueError(f"Unknown model: {model_name}")
