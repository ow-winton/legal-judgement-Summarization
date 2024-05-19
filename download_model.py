from transformers import BertTokenizer, BertModel

# 指定模型名称
model_name = "fnlp/bart-base-chinese"

# 下载并加载模型
print(f"Downloading and loading tokenizer and model for: {model_name}")
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 打印tokenizer和model内容确认下载成功
print(tokenizer)
print(model)

# 保存tokenizer和model到本地目录
save_directory = "./models/fnlp-bart-base-chinese"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
print(f"Tokenizer and model saved to: {save_directory}")
