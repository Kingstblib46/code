import torch
import common

char_to_idx = {char: idx for idx, char in enumerate(common.captcha_array)}
idx_to_char = {idx: char for idx, char in enumerate(common.captcha_array)}

def text_to_vec(text):
    """Convert captcha text to one-hot encoded vector"""
    if len(text) != common.captcha_size:
        raise ValueError(f"Input text length {len(text)} does not match captcha_size {common.captcha_size}")

    vector = torch.zeros(common.captcha_size, common.num_classes)
    for i, char in enumerate(text):
        try:
            idx = char_to_idx[char]
            vector[i, idx] = 1.0
        except KeyError:
            raise ValueError(f"Character '{char}' not found in captcha_array.")
    return vector

def vec_to_text(vec):
    """Convert model output vector (Batch, captcha_size, num_classes) or (captcha_size, num_classes) to text"""
    if vec.dim() == 2:
        vec = vec.unsqueeze(0)
    elif vec.dim() != 3:
        raise ValueError(f"Input vector must have 2 or 3 dimensions, got {vec.dim()}")

    char_indices = torch.argmax(vec, dim=2)

    texts = []
    for indices in char_indices:
        text = "".join([idx_to_char[idx.item()] for idx in indices])
        texts.append(text)

    return texts[0] if len(texts) == 1 and vec.shape[0] == 1 and vec.dim()==3 else texts


if __name__ == '__main__':
    test_text = "a1b2"
    print(f"Original text: {test_text}")

    vec = text_to_vec(test_text)
    print(f"Vector shape: {vec.shape}")

    decoded_text = vec_to_text(vec)
    print(f"Decoded text (single): {decoded_text}")
    assert decoded_text == test_text, "Single vector decode failed!"

    batch_vec = torch.stack([vec, text_to_vec("c3d4"), text_to_vec("e5f6")], dim=0)
    print(f"Batch vector shape: {batch_vec.shape}")
    decoded_texts = vec_to_text(batch_vec)
    print(f"Decoded texts (batch): {decoded_texts}")
    assert decoded_texts == [test_text, "c3d4", "e5f6"], "Batch vector decode failed!"

    print("One-hot encoding tests passed.")