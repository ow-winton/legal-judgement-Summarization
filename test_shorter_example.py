# 这个文档用来检查我的分割标准是否可行
# 导入包
import csv
import os
import pandas as pd
import fire
import re
from filepackage import dataloader

def main(filepath: str = 'D:\legal-judgement-Summarization\Judicial Documents shorter'  # 目前的开发使用前一百多个数据
):
    dataloader.load_data_to_csv(filepath)

# 加载数据





if __name__ == '__main__':
    fire.Fire(main)
