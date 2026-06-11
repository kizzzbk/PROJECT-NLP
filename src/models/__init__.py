from src.models.base_model import BaseSentimentModel
from src.models.bilstm import BiLSTMModel
from src.models.bilstm_attention import BiLSTMAttentionModel
from src.models.phobert import PhoBERTModel
from src.models.attention import AdditiveAttention


MODEL_REGISTRY = {
    "bilstm": BiLSTMModel,
    "bilstm_attention": BiLSTMAttentionModel,
    "phobert": PhoBERTModel,
}


def get_model_class(name: str):
    """
    Get model class by name.
    
    Args:
        name: One of 'bilstm', 'bilstm_attention', 'phobert'.
        
    Returns:
        Model class.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]
