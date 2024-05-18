# 导入包
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

def main(filepath:str= 'D:\legal-judgement-Summarization\Judicial Documents shorter' # 目前的开发使用前一百多个数据
         ):
    # 加载数据集
    train_df_original, val_df_original, test_df_original = dataloader.load_data_from_file(filepath)

    #将数据转换为  Dataset 对象 这个包是Hugging Face提供的服务
    train_df_original = Dataset.from_dict(train_df_original)
    val_df_original = Dataset.from_dict(val_df_original)
    test_df_original = Dataset.from_dict(test_df_original)


    # 加载 ROUGE 评估指标 load_metric 是 Hugging Face 的一个方法，用于加载评估指标
    # rouge =load_metric('rouge')
    rouge = evaluate.load('rouge') # 由于metric 在下一个主要版本移除，所以改用evaluate方法
    #加载 BERT 分词器  加载预训练的 BERT 分词器，用于将文本转换为token格式，也就是数字格式（）
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    #bert-base-chinese是以一个字作为一个词，开头是特殊符号 [CLS]，两个句子中间用 [SEP] 分隔，句子末尾也是 [SEP]，最后用 [PAD] 将句子填充到 max_length 长度


    #设置编码器和解码器的最大长度
    encoder_max_length = 512
    decoder_max_length = 512

    #设置训练参数
    batch_size = 1
    learning_rate = 3e-5
    weight_decay = 0.01
    num_train_epochs = 100
    random_seed = 3407

    dataloader.set_seed(random_seed)
    train_df = train_df_original.map( #map 方法用于将指定的函数应用到数据集的每一个元素或批次上
        lambda batch:dataloader.transfer_Data_to_inputs(batch, tokenizer),
        batched=True,
        batch_size = batch_size,remove_columns=["source","target"]

    )
    val_df = val_df_original.map( #map 方法用于将指定的函数应用到数据集的每一个元素或批次上
        lambda batch:dataloader.transfer_Data_to_inputs(batch, tokenizer),
        batched=True,
        batch_size = batch_size,remove_columns=["source","target"]

    )
    test_df = test_df_original.map( #map 方法用于将指定的函数应用到数据集的每一个元素或批次上
        lambda batch:dataloader.transfer_Data_to_inputs(batch, tokenizer),
        batched=True,
        batch_size = batch_size,remove_columns=["source","target"]

    )
    # set Python list to PyTorch tensor
    train_df.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    val_df.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    test_df.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    #输入训练参数
    training_args = Seq2SeqTrainingArguments(
        #使用生成模式进行预测
        predict_with_generate=True,
        #评估策略设置为每个 epoch 结束时进行评估
        evaluation_strategy="epoch",
        #模型保存策略设置为每个 epoch 结束时保存模型
        save_strategy="epoch",
        #每个设备（例如 GPU）上的训练 batch 大小
        per_device_train_batch_size=batch_size,
        #每个设备上的评估 batch 大小
        per_device_eval_batch_size=batch_size,
        #学习率
        learning_rate=learning_rate,
        #权重衰减（L2正则化）系数
        weight_decay=weight_decay,
        #训练的总 epoch 数
        num_train_epochs=num_train_epochs,
        #是否使用 16 位浮点数（混合精度）训练
        fp16=False,
        #模型输出目录，用于保存训练过程中生成的模型和检查点
        output_dir="./module",
        #学习率调度器类型
        lr_scheduler_type="cosine",
        #保存的模型数量上限
        save_total_limit=2,
        #梯度累积步数，意味着在进行反向传播前累积多少步的梯度
        gradient_accumulation_steps=1,
        #使用 adafactor 优化器。这是一个特别适用于内存受限环境的优化器。
        optim="adafactor",
        #在训练结束时加载最好的模型（基于验证集的性能）
        load_best_model_at_end=True,
        #按照样本长度分组以进行更高效的批处理
        group_by_length=True,
        #启用梯度检查点，以节省内存
        gradient_checkpointing=True,
        #设置随机种子，确保实验的可重复性
        seed=3407
    )


    # load model + enable gradient checkpointing & disable cache for checkpointing
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese",use_cache=False)

    model.config.num_beams = 4
    model.config.max_length = 512
    model.config.min_length = 256
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=dataloader.compute_metrics,
        train_dataset=train_df,
        eval_dataset=val_df,
    )


    # start training
    # torch.autograd.set_detect_anomaly(True)
    trainer.train()

    predictions = trainer.predict(test_df)
    metrics= dataloader.computer_metrics(predictions)
    print("ROUGE-2 Precision:", metrics['rouge2_precision'])
    print("ROUGE-2 Recall:", metrics['rouge2_recall'])
    print("ROUGE-2 Fmeasure:",metrics['rouge2_fmeasure'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def generate_summarization(batch):
        inputs_dict = tokenizer(batch["source"], padding="max_length", max_length=512, return_tensors="pt",
                                truncation=True)
        # Move tensors to the device
        input_ids = inputs_dict.input_ids.to(device)
        predicted_abstract_ids = model.generate(input_ids)
        batch["predicted_target"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
        return batch

    result = test_df_original.map(generate_summarization, batch_size=1, batched=True)
    result_df = pd.DataFrame(result)
    result_df.to_csv("result.csv")

if __name__ == '__main__':
    fire.Fire(main)


