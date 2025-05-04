import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data.load_dataset import load_dataset_from_csv
from models.bert import train_bert, test_bert

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--test', action='store_true')
parser.add_argument('--shap', action='store_true')
parser.add_argument('--lime', action='store_true')
parser.add_argument('--additional_metrics', action='store_true')
parser.add_argument('--checkpoint_name', type=str, default=None, help='Name of the checkpoint to load for bert (None if training)')
parser.add_argument('--samples_to_explain', type=int, default=100, )
parser.add_argument('--steps', type=int, default=5)
parser.add_argument('--percent_dataset', type=int, default=100)
args = parser.parse_args()


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU")


if __name__ == "__main__":

    train_dataset, test_dataset = load_dataset_from_csv(black_box=True, percent_dataset=args.percent_dataset)
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    model.config.problem_type = "single_label_classification"

    def tokenize_function(examples):
        return tokenizer(examples["cleaned_text"], truncation=True, padding=True)

    encoded_dataset_train = train_dataset.map(tokenize_function, batched=True)
    encoded_dataset_test = test_dataset.map(tokenize_function, batched=True)

    if args.train:
        train_bert(model, encoded_dataset_train, encoded_dataset_test, epochs=args.epochs)
    if args.test:
        test_bert(encoded_dataset_test, tokenizer, args.shap, args.lime, args.samples_to_explain, args.steps, args.checkpoint_name, additional_metrics=args.additional_metrics)


        

        