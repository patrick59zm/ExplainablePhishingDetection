from datasets import load_dataset


def load_dataset_from_csv(black_box=True, percent_dataset=100):
    if black_box:
        ### Load the datasets ####
        train_dataset = load_dataset('csv', data_files={
            'train': 'data/train/train_dataset.csv'
        }, split=f'train[:{percent_dataset}%]')
        test_dataset = load_dataset('csv', data_files={
            'test': 'data/test/test_dataset.csv'
        }, split=f'test[:{percent_dataset}%]')

        columns_to_keep: list = ["p_label", "cleaned_text"]
        train_dataset = train_dataset.select_columns(columns_to_keep)
        test_dataset = test_dataset.select_columns(columns_to_keep)

        train_dataset = train_dataset.filter(lambda x: x["cleaned_text"] is not None)
        test_dataset = test_dataset.filter(lambda x: x["cleaned_text"] is not None)
        return train_dataset, test_dataset
    else:
        #TODO: Load dataset for white box model
        return
  
