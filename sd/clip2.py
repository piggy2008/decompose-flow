import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel


class CLIPTextEncoder(nn.Module):
    """
    A PyTorch module that extracts CLIP text features.
    Outputs token-level embeddings of shape (batch, 77, 768).
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes the CLIP text encoder.

        Args:
            model_name (str): The pre-trained CLIP model name.
                              Examples: "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"
        """
        super(CLIPTextEncoder, self).__init__()

        # Load CLIP tokenizer and text model
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)


        # Set model to evaluation mode and disable gradients (feature extraction only)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze the model

    def forward(self, text_list):
        """
        Forward pass to extract text features from CLIP.

        Args:
            text_list (list of str): A batch of text prompts.

        Returns:
            torch.Tensor: Feature tensor of shape (batch, 77, 768)
        """
        # Tokenize input text
        inputs = self.tokenizer(text_list, return_tensors="pt", padding="max_length", truncation=True, max_length=77)

        # Move input tensors to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract CLIP features
        with torch.no_grad():
            text_features = self.model(**inputs).last_hidden_state  # Shape: (batch, 77, 768)

        return text_features  # Return full sequence features


# Example usage
if __name__ == "__main__":
    # Initialize the CLIP text encoder
    clip_encoder = CLIPTextEncoder("openai/clip-vit-large-patch14")

    # Example input: A batch of short text prompts
    text_inputs = [
        "please remove the haze noise in the underwater image",
        "enhance the contrast of the blurry photo",
    ]

    # Extract features
    features = clip_encoder(text_inputs)

    # Output feature tensor shape
    print("Feature tensor shape:", features.shape)  # Expected: (batch_size, 77, 768)
