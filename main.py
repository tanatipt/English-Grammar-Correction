from grammar_correction_trainer import GrammarCorrectionTrainer
from utils import GCEDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
from config import settings
import torch


if __name__ == "__main__":
    train_path = 'preprocessed/train.csv'
    es_path = "preprocessed/es.csv"
    valid_path = "preprocessed/valid.csv"
    test_path = "preprocessed/test.csv"

    train_data = GCEDataset(train_path)
    es_data = GCEDataset(es_path)
    valid_data = GCEDataset(valid_path)
    test_data = GCEDataset(test_path)
    full_train_data = ConcatDataset([train_data, es_data, valid_data])

    batch_size = settings.batch_size
    torch.manual_seed(2543673)

    # Create DataLoaders for training, early stopping, validation, and test datasets
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    es_loader = DataLoader(es_data, batch_size, shuffle = False)
    valid_loader = DataLoader(valid_data, batch_size, shuffle = False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)
    full_train_loader = DataLoader(full_train_data, batch_size, shuffle=True)

    gct = GrammarCorrectionTrainer(
        train_loader=train_loader,
        es_loader=es_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        full_train_loader=full_train_loader
    )

    gct.hyperparameter_tuning()

    gct.refit_model()