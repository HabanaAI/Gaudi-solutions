import pandas as pd 
import numpy as np

from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset

from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()


def load_data():
    df = pd.read_csv(
        'FinancialPhraseBank-v1.0/Sentences_50Agree.txt',
        sep='@',
        names=['sentence', 'label'],
        encoding = "ISO-8859-1")
    df = df.dropna()
    df['label'] = df['label'].map({"neutral": 0, "positive": 1, "negative": 2})
    df.head()

    df_train, df_test, = train_test_split(df, stratify=df['label'], test_size=0.1, random_state=42)
    df_train, df_val = train_test_split(df_train, stratify=df_train['label'],test_size=0.1, random_state=42)

    dataset_train = Dataset.from_pandas(df_train, preserve_index=False)
    dataset_val = Dataset.from_pandas(df_val, preserve_index=False)
    dataset_test = Dataset.from_pandas(df_test, preserve_index=False)

    return dataset_train, dataset_val, dataset_test


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(predictions, labels)}


def main():
    dataset_train, dataset_val, dataset_test = load_data()

    bert_model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=3)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    dataset_train = dataset_train.map(lambda e: bert_tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=128), batched=True)
    dataset_val = dataset_val.map(lambda e: bert_tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=128), batched=True)
    dataset_test = dataset_test.map(lambda e: bert_tokenizer(e['sentence'], truncation=True, padding='max_length' , max_length=128), batched=True)

    dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    dataset_val.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    args = TrainingArguments(
        output_dir='temp/',
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        save_strategy='no',
        logging_strategy='epoch',
        logging_dir='logs/',
        report_to='tensorboard',

        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        metric_for_best_model='accuracy',

        use_habana=True,                   # use Habana device
        use_lazy_mode=True,                # use Gaudi lazy mode
        use_fused_adam=True,               # used optimised version of Adam for Gaudi
        use_fused_clip_norm=True,          # use Habana's fused gradient norm clipping operator
    )

    trainer = Trainer(
        model=bert_model,                   # the instantiated 🤗 Transformers model to be trained
        args=args,                          # training arguments, defined above
        train_dataset=dataset_train,        # training dataset
        eval_dataset=dataset_val,           # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()   


if __name__ == '__main__':
    main()

