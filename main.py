# 导入包

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

filepath = 'D:\legal-judgement-Summarization\Judicial Documents'
def main(filepath:str= 'D:\legal-judgement-Summarization\Judicial Documents'
         ):
    #加载数据
    def load_data():
        all_file = os.listdir(filepath) # file_path 目录中的所有文件和文件夹，并返回一个包含这些名称的列表，保存在all_file 变量里面
        all_dataFrame = pd.DataFrame() # 创建一个空的 Pandas DataFrame
        for file in all_file:
            # 打开遍历到的文件并且读取其中所有内容 这是为了后续我们能够把他们存放到dataframe里面
            full_file_path = os.path.join(filepath, file)
            inputFile = open(full_file_path,encoding="utf-8").readlines() #
            '''
            在大多数操作系统（包括 Unix、Linux 和 macOS）中，路径分隔符是 /。在 Windows 中，虽然 \ 是默认路径分隔符，但 Python 也接受 / 作为路径分隔符。
            '''
            print(inputFile)
            #
            # data = "".join(inputFile)
            #
            # parts = re.split("本院认为，|本院意见，",data, maxsplit=1)
            #
            # csv_file_path ="D:\legal-judgement-Summarization\example_save分割数据.csv"
            # with open(csv_file_path,mode="w")
    load_data()
if __name__ == '__main__':
    fire.Fire(main)


