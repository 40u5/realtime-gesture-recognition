
import torch
from gesture_transformer import GestureTransformer, create_padding_mask

# Test function
def test_model():
    """Test the GestureTransformer model."""
    # Model parameters
    batch_size = 4
    seq_len = 32
    landmark_dim = 63
    num_classes = 5
    
    # Create model
    model = GestureTransformer(
        num_classes=num_classes,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        max_seq_length=seq_len
    )
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, landmark_dim)
    
    # Create padding mask (simulate variable length sequences)
    lengths = torch.tensor([seq_len, seq_len-5, seq_len-10, seq_len-2])
    padding_mask = create_padding_mask(dummy_input, lengths)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input, padding_mask)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    test_model()
