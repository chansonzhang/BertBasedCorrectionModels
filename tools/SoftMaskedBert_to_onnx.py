"""
@Time   :   2025-03-24 20:24:00
@File   :   SoftMaskedBert_to_onnx.py
@Author :   Zhang Chen
@Email  :   zhangchen.shaanxi{at}gmail.com
"""

from transformers import BertModel, BertTokenizer
from tools.inference import load_model_directly_onnx
import torch
from pathlib import Path
import onnxruntime as ort
import numpy as np

if __name__ == '__main__':
    ckpt_file = './checkpoints/SoftMaskedBert/epoch=09-val_loss=0.03032.ckpt'
    config_file = 'csc/train_SoftMaskedBert.yml'

    tokenizer: BertTokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    onnx_model = load_model_directly_onnx(ckpt_file, config_file)

    print(f"onnx_model.device: {onnx_model.device}")

    inputs = tokenizer(
        ["你好帮我生成一个荷花店的课件好吗行吧巴拉巴拉小魔哈哈哈哈红红火火恍恍惚惚"],
        padding=True,
                       return_tensors="pt",
                       truncation = True,
    )
    device = onnx_model.device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)

    # 定义动态轴（处理可变 batch size 和序列长度）
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "prob": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
        "sequence_output": {0: "batch_size", 1: "sequence_length"}
    }

    result_file = f"{Path(ckpt_file).parent}/{Path(ckpt_file).name[:-5]}.onnx"
    # 导出为 ONNX
    torch.onnx.export(
        onnx_model,
        (input_ids, attention_mask, token_type_ids),
        result_file,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["prob","output","sequence_output"],
        dynamic_axes=dynamic_axes,
        opset_version=12,  # 建议使用 11 或更高版本
    )
    print(f"result_file: {result_file}")

    # 加载 ONNX 模型
    ort_session = ort.InferenceSession(result_file)

    # 准备输入数据
    inputs_onnx = {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
        "token_type_ids": token_type_ids.cpu().numpy()
    }

    # 运行推理
    outputs = ort_session.run(None, inputs_onnx)

    # 比较原始模型和 ONNX 模型的输出
    with torch.no_grad():
        original_outputs = onnx_model(input_ids, attention_mask, token_type_ids)

    print(f"original_outputs[1].shape: {original_outputs[1].shape}")
    print(f"len(outputs): {len(outputs)}")
    print(f"outputs[1].shape: {outputs[1].shape}")
    # 检查输出是否一致
    atol = 1e-5
    ref_value =original_outputs[1].cpu().numpy()
    ort_value=outputs[1]
    all_close = np.allclose(ref_value,
                       ort_value,
                       atol=atol)
    if not all_close:
        max_diff = np.amax(np.abs(ref_value - ort_value))
        # print(ref_value)
        # print(ort_value)
        print(f"\t\t-[x] values not close enough, max diff: {max_diff} (atol: {atol})")
    else:
        print(f"\t\t-[✓] all values close (atol: {atol})")