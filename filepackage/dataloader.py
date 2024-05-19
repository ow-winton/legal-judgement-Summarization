# 导入包
import csv
# 导入包
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
import csv
import os
import re
import pandas as pd
from datasets import Dataset
import numpy as np
import random
import torch
import fire
from transformers import BartForConditionalGeneration
from statistics import mean
from datasets import load_dataset, load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BertTokenizer
)
from filepackage import dataloader
import evaluate
from transformers import BertTokenizer

# 使用相对路径加载模型
print("Current working directory:", os.getcwd())

# 设置模型的绝对路径
model_path = "/root/autodl-tmp/legal/models/fnlp-bart-base-chinese"
print("Loading tokenizer and model from:", model_path)

# 打印目录内容
print("Contents of model directory:", os.listdir(model_path))

# 使用绝对路径加载模型
tokenizer = BertTokenizer.from_pretrained(model_path)
encoder_max_length = 512
decoder_max_length = 512
rouge = load_metric("rouge")
# compute Rouge score during validation

def compute_metrics(pred):
    label_ids = pred.label_ids
    pred_ids = pred.predictions
    if label_ids is None or pred_ids is None:
        logger.warning("预测或参考文本为空，跳过此批次的ROUGE计算。")
        return {}

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    try:
        rouge2_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
        return {
            "rouge2_precision": round(rouge2_output.precision, 4),
            "rouge2_recall": round(rouge2_output.recall, 4),
            "rouge2_fmeasure": round(rouge2_output.fmeasure, 4)
        }
    except Exception as e:
        logger.error(f"在计算ROUGE分数时发生错误：{e}")
        return {}

def transfer_Data_to_inputs(batch, tokenizer, encoder_max_length=512, decoder_max_length=512):
    inputs = tokenizer(
        batch["source"],
        padding ="max_length",
        truncation = True,
        max_length = encoder_max_length
    )
    outputs = tokenizer(
        batch["target"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length

    )
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["labels"] = outputs.input_ids
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]
    return batch
# 设置seed 确保可以重复实验
def set_seed(seed: int=3407):
    # 设置 NumPy 随机数生成器的种子
    np.random.seed(seed)
    # 设置 Python random 模块的随机数生成器的种子
    random.seed(seed)
    # 设置 PyTorch 的随机数生成器的种子
    torch.manual_seed(seed)
    # 如果使用 CUDA，也要设置 CUDA 随机数生成器的种子
    torch.cuda.manual_seed(seed)
    # 确保 PyTorch 使用确定性算法进行计算（在某些情况下会影响性能）
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark =False
    # 设置 Python 内置的哈希函数种子（防止哈希随机性）
    os.environ["PYTHONHASHSEED"]= str(seed)
    # 输出设置的随机种子
    # print(f"random seed set as {seed}")
    # print("NumPy random number:", np.random.rand())
    # print("Python random number:", random.random())
    #
    # # 测试 PyTorch 随机数生成
    # print("PyTorch random number:", torch.rand(1))

# 真正load_Data的部分
def load_data_from_file(filepath):
    all_files = os.listdir(filepath)  # file_path 目录中的所有文件和文件夹，并返回一个包含这些名称的列表，保存在all_file 变量里面
    all_df = pd.DataFrame()  # 创建一个空的 Pandas DataFrame

    for file in all_files:
        full_file_path = os.path.join(filepath, file)  # 构建了遍历文件的路径
        input_file = open(full_file_path, encoding="utf-8").readlines()  # 遍历所有文件
        data = "".join(input_file)  # 把遍历到的每一个文件内容循环存储到data里面

        parts = re.split("本院认为，|本院意见，", data, maxsplit=1)

        data_dict = {}

        if len(parts)==2:  # 如果成果分割就将他存储在dict里面
            data_dict["source"] = parts[0]
            data_dict["target"]=parts[1]
        else:
            data_dict["source"] = data
            data_dict["target"] = ""
        data_df = pd.DataFrame(data_dict,index=[0])
        all_df = pd.concat([all_df,data_df],ignore_index=True)
    #这部分对遍历完成的数据进行了处理
    all_df = all_df[all_df["target"].str.strip() !=""] # 去除空行

    garbled_rows = [] # 循环遍历df，然后把有问题的行的索引加到garbled_rows中
    for index,row in all_df.iterrows():
        if row_contains_garbled_text(row):
            garbled_rows.append(index)

    all_df = all_df.drop(garbled_rows)

    # 接下来把数据转换成dataset对象  l46
    new_dataset = Dataset.from_pandas(all_df)
    train_index = int(0.8 * len(new_dataset))
    val_index = int(0.9 * len(new_dataset))

    train_dataset = new_dataset[:train_index]
    val_dataset = new_dataset[train_index:val_index]
    test_dataset = new_dataset[val_index:]

    return train_dataset,val_dataset,test_dataset
'''
load_data_to_csv 将文档内容分成两个部分存储到csv文件中
'''
# csv部分
def load_data_to_csv(filepath):
    all_files = os.listdir(filepath)  # file_path 目录中的所有文件和文件夹，并返回一个包含这些名称的列表，保存在all_file 变量里面

    # 然后开始进行文档分割，分割的标志是 “本院认为|本院意见”这两个句子，裁判文书数据大部分严格遵循这个标准

    # 负责生成示范的csv文件
    csv_file_path = "D:\legal-judgement-Summarization\example_save/分割数据.csv"
    with open(csv_file_path, mode='w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['input', 'target'])  # 写入表头

        for file in all_files:
            # 打开遍历到的文件并且读取其中所有内容 这是为了后续我们能够把他们存放到dataframe里面
            full_file_path = os.path.join(filepath, file)
            # inputFile = open(full_file_path,encoding="utf-8").readlines() #
            with open(full_file_path, encoding='utf-8') as inputFile:
                lines = inputFile.readlines()
                lines = [line.lstrip('\ufeff') for line in lines]  # 去除BOM

            # 在大多数操作系统（包括 Unix、Linux 和 macOS）中，路径分隔符是 /。在 Windows 中，虽然 \ 是默认路径分隔符，但 Python 也接受 / 作为路径分隔符。

            cleaned_lines = [line.encode('utf-8', errors='ignore').decode('utf-8') for line in lines]
            data = "".join(cleaned_lines)

            parts = re.split("本院认为，|本院意见，", data, maxsplit=1)

            # 确保parts有足够的分割结果，否则填充空字符串
            if len(parts) == 1:
                parts.append('')
            # 写入分割后的内容
            writer.writerow(parts)

'''
接下来的方法是为了检查csv数据的完整性，对数据进行预处理
'''


def contains_garbled_text(text): # 检查是否包含乱码
    try:
        # 尝试解码文本，如果文本不能被解码为UTF-8，则视为包含乱码
        text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return True

    # 允许更多的常见字符和符号，包括中文标点符号、换行符等
    garbled_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
    return bool(garbled_pattern.search(text))
def row_contains_garbled_text(row):  # 检查行中的乱码情况
    for value in row:
        if isinstance(value, str) and contains_garbled_text(value):
            return True
    return False

def preprocess_csv(input_csv_path, output_csv_path):
    # 加载 CSV 文件
    df = pd.read_csv(input_csv_path, encoding='utf-8')

    # 去除目标列 'target' 为空的数据
    df = df.dropna(subset=['target'])

    # 过滤掉包含乱码的数据，并打印调试信息
    garbled_rows = []
    for index, row in df.iterrows():
        if row_contains_garbled_text(row):
            garbled_rows.append(index)


    df = df.drop(garbled_rows)

    # 保存处理后的数据
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')