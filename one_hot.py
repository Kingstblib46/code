# one_hot.py
import torch
import common # 导入共享变量

# 预先计算字符到索引的映射，提高效率
char_to_idx = {char: idx for idx, char in enumerate(common.captcha_array)}
idx_to_char = {idx: char for idx, char in enumerate(common.captcha_array)}

def text_to_vec(text):
    """将验证码文本转换为独热编码向量"""
    if len(text) != common.captcha_size:
        raise ValueError(f"Input text length {len(text)} does not match captcha_size {common.captcha_size}")

    # 创建一个 (captcha_size, num_classes) 的零向量
    vector = torch.zeros(common.captcha_size, common.num_classes)
    for i, char in enumerate(text):
        try:
            idx = char_to_idx[char]
            vector[i, idx] = 1.0
        except KeyError:
            raise ValueError(f"Character '{char}' not found in captcha_array.")
    return vector

def vec_to_text(vec):
    """将模型的输出向量 (Batch, captcha_size, num_classes) 或 (captcha_size, num_classes) 转换为文本"""
    # 判断输入是单个向量还是一个批次
    if vec.dim() == 2: # 单个向量 (captcha_size, num_classes)
        vec = vec.unsqueeze(0) # 增加 Batch 维度 -> (1, captcha_size, num_classes)
    elif vec.dim() != 3:
        raise ValueError(f"Input vector must have 2 or 3 dimensions, got {vec.dim()}")

    # 找到每个位置概率最大的字符索引
    # vec shape: (Batch, captcha_size, num_classes)
    char_indices = torch.argmax(vec, dim=2) # shape: (Batch, captcha_size)

    # 将索引转换为字符并组合成字符串
    texts = []
    for indices in char_indices: # indices shape: (captcha_size,)
        text = "".join([idx_to_char[idx.item()] for idx in indices])
        texts.append(text)

    # 如果输入是单个向量，返回单个字符串，否则返回列表
    return texts[0] if len(texts) == 1 and vec.shape[0] == 1 and vec.dim()==3 else texts


# 测试函数 (可选)
if __name__ == '__main__':
    test_text = "a1b2"
    print(f"Original text: {test_text}")

    # 测试 text_to_vec
    vec = text_to_vec(test_text)
    print(f"Vector shape: {vec.shape}")
    # print(f"Vector:\n{vec}")

    # 测试 vec_to_text (单个向量)
    decoded_text = vec_to_text(vec)
    print(f"Decoded text (single): {decoded_text}")
    assert decoded_text == test_text, "Single vector decode failed!"

    # 测试 vec_to_text (批处理向量)
    batch_vec = torch.stack([vec, text_to_vec("c3d4"), text_to_vec("e5f6")], dim=0)
    print(f"Batch vector shape: {batch_vec.shape}")
    decoded_texts = vec_to_text(batch_vec)
    print(f"Decoded texts (batch): {decoded_texts}")
    assert decoded_texts == [test_text, "c3d4", "e5f6"], "Batch vector decode failed!"

    print("One-hot encoding tests passed.")