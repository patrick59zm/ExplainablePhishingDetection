from datasets import load_dataset, Value


def load_dataset_from_csv(seed, percent_dataset=100, machine_generated=False):
    dataset = "_machine_data" if machine_generated else "_dataset"
    label = "g_label" if machine_generated else "p_label"
    train_dataset = load_dataset('csv', data_files={
        'train': f'data/train/train{dataset}.csv'
    }, split=f'train')
    test_dataset = load_dataset('csv', data_files={
        'test': f'data/test/test{dataset}.csv'
    }, split=f'test')
    
    if percent_dataset < 100:
        assert percent_dataset > 0, "percent_dataset must be greater than 0"
        assert percent_dataset <= 100, "percent_dataset must be less than or equal to 100"
        # Shuffle the dataset and select the first 10% of the indices
        train_dataset = train_dataset.shuffle(seed=seed).select(range(int((percent_dataset/100) * len(train_dataset))))
        test_dataset = test_dataset.shuffle(seed=seed).select(range(int((percent_dataset/100) * len(test_dataset))))

    columns_to_keep: list = [label, "cleaned_text"]
    train_dataset = train_dataset.select_columns(columns_to_keep)
    test_dataset = test_dataset.select_columns(columns_to_keep)

    train_dataset = train_dataset.filter(lambda x: x["cleaned_text"] is not None)
    test_dataset = test_dataset.filter(lambda x: x["cleaned_text"] is not None)
    
    train_dataset = train_dataset.cast_column(label, Value(dtype="int32"))
    test_dataset = test_dataset.cast_column(label, Value(dtype="int32"))
    
    return train_dataset, test_dataset

 