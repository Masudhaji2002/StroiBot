
import sys

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

# Пути к данным
train_file = "data/train.txt"
dev_file = "data/dev.txt"
output_dir = "models/dl_extractor"

# Шаг 1: Загрузка данных
def load_data(file_path):
    sentences, tags = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        sentence, tag_seq = [], []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag_seq)
                    sentence, tag_seq = [], []
                continue
            token, tag = line.split()
            sentence.append(token)
            tag_seq.append(tag)
        if sentence:  # Добавляем последний пример
            sentences.append(sentence)
            tags.append(tag_seq)
    return sentences, tags

# Загружаем данные
train_sentences, train_tags = load_data(train_file)
dev_sentences, dev_tags = load_data(dev_file)

# Конвертируем данные в Hugging Face Dataset
def convert_to_dataset(sentences, tags):
    return Dataset.from_dict({
        "tokens": sentences,
        "ner_tags": tags,
    })

# Создаём DatasetDict
dataset = DatasetDict({
    "train": convert_to_dataset(train_sentences, train_tags),
    "validation": convert_to_dataset(dev_sentences, dev_tags),
})

# Шаг 2: Маппинг тегов BIO
unique_tags = set(tag for seq in train_tags for tag in seq)
tag2id = {tag: i for i, tag in enumerate(sorted(unique_tags))}
id2tag = {i: tag for tag, i in tag2id.items()}

# Шаг 3: Загрузка модели и токенизатора
model_name = "bert-base-multilingual-cased"
pretrained_model_path = "models/dl_extractor"

# Если модель уже существует, загружаем её
try:
    model = AutoModelForTokenClassification.from_pretrained(pretrained_model_path, num_labels=len(tag2id))
    print("Загружена существующая модель для дообучения.")
except Exception as e:
    print("Не удалось загрузить существующую модель. Загружаем новую.", str(e))
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(tag2id))

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Функция для вычисления метрик
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Преобразуем метки в список
    true_labels = [
        [id2tag[label] for label in label_seq if label != -100] for label_seq in labels
    ]
    pred_labels = [
        [id2tag[pred] for pred, label in zip(pred_seq, label_seq) if label != -100] 
        for pred_seq, label_seq in zip(predictions, labels)
    ]
    
    # Вычисляем метрики
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

# Токенизация и выравнивание меток
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,  # Обрезаем длинные последовательности
        padding=True,     # Добавляем паддинг
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Получаем соответствие токенов словам
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:  # Игнорируем [CLS] и [SEP]
                label_ids.append(-100)
            elif word_id != previous_word_id:  # Новый токен, добавляем метку
                label_ids.append(tag2id[label[word_id]])
            else:  # Для сабтокенов используем ту же метку, что и для основного слова
                label_ids.append(-100)
            previous_word_id = word_id
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Токенизация датасетов
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Шаг 4: Настройки обучения
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-6,  # Пониженная скорость обучения для дообучения
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=3,
    logging_dir="logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

# Шаг 5: Инициализация Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Передаем функцию для вычисления метрик
)

# Шаг 6: Дообучение модели
trainer.train()

# Сохранение модели и токенизатора
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Модель дообучена и сохранена в {output_dir}")

# Логируем результаты обучения
print("Результаты обучения:")
train_results = trainer.train()
print(train_results)

# Логируем валидационные метрики после обучения
print("Оценка модели на валидации:")
eval_results = trainer.evaluate()
print(eval_results)
