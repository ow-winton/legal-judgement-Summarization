import pandas as pd
df = pd.read_csv('./example_save/result 1.csv')
# 假设df是你的DataFrame
df.to_csv('D:\legal-judgement-Summarization\example_save/result222.csv', encoding='utf-8', errors='ignore')
