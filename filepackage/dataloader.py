import csv
import os
import pandas as pd
import fire
import re
def load_data_to_csv(filepath):
    all_files = os.listdir(filepath)  # file_path 目录中的所有文件和文件夹，并返回一个包含这些名称的列表，保存在all_file 变量里面

    # 然后开始进行文档分割，分割的标志是 “本院认为|本院意见”这两个句子，裁判文书数据大部分严格遵循这个标准

    # 负责生成示范的csv文件
    csv_file_path = "D:\legal-judgement-Summarization\example_save/分割数据.csv"
    with open(csv_file_path, mode='w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Part 1', 'Part 2'])  # 写入表头

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