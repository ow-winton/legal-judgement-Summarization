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
   #  dataloader.load_data_to_csv(filepath) #将1000个示例生成
   ''' 数据清洗
   input_csv = 'D:\legal-judgement-Summarization\example_save\分割数据.csv'  # 输入的 CSV 文件路径
   output_csv = 'D:\legal-judgement-Summarization\example_save\预处理过的分割数据.csv'  # 处理后的 CSV 文件路径

   dataloader.preprocess_csv(input_csv, output_csv)
    '''


# 打印部分数据样本
# print("Training Dataset Sample:")
# print(train_df_original[0])
# print(train_df_original[1])
# print(train_df_original[2])
#
# print("\nValidation Dataset Sample:")
# print(val_df_original[0])
# print(val_df_original[1])
# print(val_df_original[2])
#
# print("\nTest Dataset Sample:")
# print(test_df_original[0])
# print(test_df_original[1])
# print(test_df_original[2])


# 加载数据





if __name__ == '__main__':
    fire.Fire(main)
