import pickle
import random

import secrets
import string
import torch.nn.functional as F

import nni
import pandas as pd
import torch
from Bio import SeqIO
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score

from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm
from tqdm.contrib import tenumerate


class DataProcess():
    def __init__(self, path):
        self.train_df = None
        self.val_df = None
        self.test_df = None
        if path == 'data':
            if 'c' not in params.keys():
                if os.path.exists('data/DNA_data.csv') and os.path.exists('data/RNA_data.csv'):
                    df = [
                        pd.read_csv('data/DNA_data.csv'),
                        pd.read_csv('data/RNA_data.csv'),
                    ]
                    df = pd.concat(df, ignore_index=True)
                    df = df.drop_duplicates(subset=['Sequence', 'Ligand_Sequence'], keep='first').reset_index(drop=True)
                    self.df_error_Seq = df[~df['Ligand_Sequence'].str.contains('^[atucg]*$', regex=True)].reset_index(drop=True)
                    self.df = df[df['Ligand_Sequence'].str.contains('^[atucg]*$', regex=True)].reset_index(drop=True)
                    self.max_ligand_length = max(len(seq) for seq in self.df['Ligand_Sequence'])
                    self.max_peptide_length = max(len(seq) for seq in self.df['Sequence'])
                else:
                    self.RawData('dna').to_csv('data/DNA_data.csv', sep=',', index=None)
                    self.RawData('rna').to_csv('data/RNA_data.csv', sep=',', index=None)
            else:
                df = pd.read_csv(f'data/CD-HIT/DNA_RNA_data_{params["c"]}_{params["n"]}.csv')
                self.df_error_Seq = df[~df['Ligand_Sequence'].str.contains('^[atucg]*$', regex=True)].reset_index(drop=True)
                self.df = df[df['Ligand_Sequence'].str.contains('^[atucg]*$', regex=True)].reset_index(drop=True)
                self.max_ligand_length = max(len(seq) for seq in self.df['Ligand_Sequence'])
                self.max_peptide_length = max(len(seq) for seq in self.df['Sequence'])
        else:
            self.df = pd.read_csv(path)
            self.max_ligand_length = max(len(seq) for seq in self.df['Ligand_Sequence'])
            self.max_peptide_length = max(len(seq) for seq in self.df['Sequence'])

    def shuffle(self):
        self.df['label'] = 1
        neg_df = self.df.copy().reset_index(drop=True)
        neg_df['Ligand_ID'] = None
        neg_df['residue(reindexed)'] = None
        neg_df['affinity'] = None
        neg_df['label'] = 0
        while True:
            shuffled_ligand = neg_df['Ligand_Sequence'].sample(frac=1).reset_index(drop=True)
            temp_sequence = neg_df['Sequence'].reset_index(drop=True)
            if sum(temp_sequence.values == shuffled_ligand.values) == 0:
                neg_df['Ligand_Sequence'] = shuffled_ligand
                break
        self.df = pd.concat([self.df, neg_df], ignore_index=True)

    @staticmethod
    def fasta_to_dataframe(fasta_file):
        ids = []
        sequences = []
        with open(f'data/{fasta_file}.fasta', "r") as file:
            for record in SeqIO.parse(file, "fasta"):
                ids.append(record.id)
                sequences.append(str(record.seq))
        df = pd.DataFrame({
            "Ligand_ID": ids,
            "Ligand_Sequence": sequences
        })
        return df

    @staticmethod
    def RawData(ligand):
        BioLiP = pd.read_csv('data/BioLiP.txt', sep='\t', header=None)
        BioLiP.columns = ['PDB_ID', 'chain_ID', 'Resolution', 'Site', 'Ligand', 'Ligand_ID', '0', 'residue(original)',
                          'residue(reindexed)', 'Catalytic(original)', 'Catalytic(reindexed)', 'EC_Number', 'GO',
                          'nan1',
                          'nan2', 'affinity', 'nan3', 'UniProt_ID', 'PubMed_ID', '?2', 'Sequence']
        BioLiP_ligand = BioLiP[BioLiP['Ligand'] == ligand]
        BioLiP_ligand['Ligand_ID'] = BioLiP_ligand['PDB_ID'] + '_' + BioLiP_ligand['Ligand'] + '_' + BioLiP_ligand[
            'Ligand_ID']
        ligand_fasta = DataProcess.fasta_to_dataframe(ligand)
        BioLiP_ligand = pd.merge(BioLiP_ligand, ligand_fasta, on="Ligand_ID", how="left")
        return BioLiP_ligand[['Ligand_ID', 'residue(reindexed)', 'Sequence', 'Ligand_Sequence', 'affinity']]


class PNIDataset(Dataset):
    def __init__(self, data, max_ligand_length, max_peptide_length, mode):
        self.pos_weight = None
        self.mode = None
        self.peptide_tensor = None
        self.ligand_tensor = None
        self.mode = mode
        self.peptide = data.df.Sequence
        self.ligand = data.df.Ligand_Sequence
        self.affinity = data.df.affinity.values
        self.label = data.df.label.values
        self.site = data.df['residue(reindexed)'].values
        self.max_ligand_length = max_ligand_length
        self.max_peptide_length = max_peptide_length

    def encode_ligand(self):
        nucleotide = 'aucgt'
        nucleotide_dict = {amino_acid: i for i, amino_acid in enumerate(nucleotide)}
        print('encode ligand sequence...')
        self.ligand_tensor = torch.zeros(len(self.ligand), self.max_ligand_length, len(nucleotide), dtype=torch.float32)
        for i, sequence in tenumerate(self.ligand):
            for j, amino_acid in enumerate(sequence):
                if amino_acid in nucleotide_dict and j < self.max_ligand_length:
                    self.ligand_tensor[i, j, nucleotide_dict[amino_acid]] = 1

    def encode_peptide(self):
        print('encode peptide sequence...')
        if self.mode == 'OH':
            amino_acids = 'ARNDCQEGHILKOMFPSTWYVX'
            aa_dict = {amino_acid: i for i, amino_acid in enumerate(amino_acids)}
            self.peptide_tensor = torch.zeros(len(self.peptide), self.max_peptide_length, len(amino_acids), dtype=torch.float32)
            print('encode peptide sequence...')
            for i, sequence in tenumerate(self.peptide):
                for j, amino_acid in enumerate(sequence):
                    if amino_acid in aa_dict and j < self.max_peptide_length:
                        self.peptide_tensor[i, j, aa_dict[amino_acid]] = 1
        elif self.mode == 'BS':
            letter_num_dict = {'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
                               'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
                               'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
                               'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
                               'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
                               'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
                               'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
                               'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
                               'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
                               'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
                               'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
                               'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
                               'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
                               'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
                               'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
                               'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
                               'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
                               'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
                               'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
                               'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
                               'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               }
            self.peptide.str = self.peptide.str.replace('U', 'C')
            self.peptide.str = self.peptide.str.replace('O', 'K')
            self.peptide.str = self.peptide.str.replace('Z', 'E')
            embedding = []
            for sequence in tqdm(self.peptide.str):
                encoded_sequence = []
                if len(sequence) < self.max_peptide_length:
                    padding = ''.join('X' * (self.max_peptide_length - len(sequence)))
                    sequence += padding
                elif len(sequence) > self.max_peptide_length:
                    sequence = sequence[:self.max_peptide_length]
                for amino_acid in sequence:
                    try:
                        encoding = letter_num_dict[amino_acid]
                    except:
                        pass
                    encoded_sequence.append(encoding)
                embedding.append(encoded_sequence)
            self.peptide_tensor = torch.tensor(embedding)

    def encode_binding_site(self):
        print('encode binding site...')
        site = list(self.site)
        num_sequences = len(site)
        self.site = torch.zeros(num_sequences, self.max_peptide_length)
        for i, item in tenumerate(site):
            if item is None:
                continue
            if type(item) != str:
                continue
            parts = item.split()
            for part in parts:
                number = int(part[1:])
                if number <= self.max_peptide_length:
                    self.site[i, number - 1] = 1
        site_sum = torch.sum(self.site, dim=1)
        site_sum = site_sum[site_sum != 0]
        mean_binding_site = torch.mean(site_sum)
        self.pos_weight = (self.max_peptide_length - mean_binding_site) / mean_binding_site

    def get_dataloader(self, dataset, batch_size, shuffle):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.peptide_tensor[index], self.ligand_tensor[index], self.label[index], self.site[index]


class ModelEvaluator:
    @staticmethod
    def getModelName(length=10):
        all_chars = string.ascii_letters + string.digits
        return ''.join(secrets.choice(all_chars) for _ in range(length))

    @staticmethod
    def save_performance(acc, auc, params):
        df = pd.DataFrame(params, index=[0])
        df['acc'] = acc
        df['auc'] = auc
        if not os.path.isfile(f'performace.csv'):
            df.to_csv(f'performace.csv', mode='w', header=True, index=False)
        else:
            df.to_csv(f'performace.csv', mode='a', header=False, index=False)

    @staticmethod
    def seed_everything(seed=3407):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def cal_perf(y_test, y_pred):
        y_test = np.hstack(y_test)
        y_pred = np.vstack(y_pred)
        acc = accuracy_score(y_test, np.argmax(y_pred, axis=1))
        auc = roc_auc_score(y_test, y_pred[:, 1])
        precision = metrics.precision_score(y_test, np.argmax(y_pred, axis=1))
        recall = metrics.recall_score(y_test, np.argmax(y_pred, axis=1))
        f1 = metrics.f1_score(y_test, np.argmax(y_pred, axis=1))
        print(
            f"  Accuracy: {round(acc, 3)}; AUC: {round(auc, 3)}; Precision: {round(precision, 3)}; Recall: {round(recall, 3)}; F1 score: {round(f1, 3)};")
        return {
            'acc': acc,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    @staticmethod
    def print_results(auc_list, acc_list):
        print(f"Mean Accuracy: {np.mean(acc_list):.3f}; Mean AUC: {np.mean(auc_list):.3f}; ")
        print(f"Mean Accuracy STD: {np.std(acc_list):.3f}; Mean AUC STD: {np.std(auc_list):.3f}; ")
        nni.report_final_result(np.mean(auc_list))


class DeepLearningToolkit:
    @staticmethod
    def test(model, dataloader, params):
        device = torch.device("cuda" if torch.cuda.is_available() and params['use_cuda'] else "cpu")
        model.to(device)
        test_predictions = []
        test_labels = []
        test_pred_site_list = []
        test_site_list = []
        with torch.no_grad():
            model.eval()
            for peptide, ligand, label, site in dataloader:
                peptide, ligand, label, site = peptide.to(device), ligand.to(device), label.to(device), site.to(device)
                pred, pred_site = model(peptide, ligand)
                pred = F.softmax(pred, dim=1)
                test_predictions.append(pred.detach().cpu().numpy())
                test_labels.append(label.detach().cpu().numpy())
                if params['MTL']:
                    test_pred_site_list.append(pred_site.detach().cpu().numpy())
                    test_site_list.append(site.detach().cpu().numpy())
        if params['MTL']:
            result = {'label': np.hstack(test_labels), 'y_pred': np.vstack(test_predictions),
                      'site_list': np.vstack(test_site_list), 'pred_site_list': np.vstack(test_pred_site_list),
                      'best_model': model}
        else:
            result = {'label': np.hstack(test_labels), 'y_pred': np.vstack(test_predictions), 'site_list': None,
                      'pred_site_list': None, 'best_model': model}
        return model, result


def HTS():
    ModelEvaluator.seed_everything()
    test_data = DataProcess('data/BioLip_20240530/BioLip_20240530_candidate.csv')
    test_dataset = PNIDataset(test_data, max_ligand_length=params['ligand_length'],
                              max_peptide_length=params['peptide_length'],
                              mode=params['mode'])
    test_dataset.encode_ligand()
    test_dataset.encode_peptide()
    test_dataset.encode_binding_site()
    test_dl = test_dataset.get_dataloader(test_dataset, params['batch_size'], False)
    with open(f"result/{params['base']}_BioLip_20240530_1:1.pkl", 'rb') as f:
        model = pickle.load(f)['best_model']
    model, result = DeepLearningToolkit.test(model, test_dl, params)
    result['df'] = test_data.df
    os.makedirs('result', exist_ok=True)
    with open(f"result/{params['base']}_BioLip_20240530_candidate.pkl", 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    # # PNI_FCN
    # params = dict(
    # base = 'FCN',
    # data_root="data",
    # save_dir="save",
    # ES=20,
    # EPOCH=1000,
    # lr=0.001,
    # l2_lambda=0,
    # batch_size=128,
    # peptide_length=1000,
    # ligand_length=100,
    # mode='OH',
    # hidden_dim=256,
    # use_cuda=True,
    # use_radam=False,
    # show_loss=True,
    # c=0.9,
    # n=5,
    # max_auc=0,
    # dp=0.1,
    # MTL=False,
    # )

    # # PNI_Transformer
    # params = dict(
    # base = 'Transformer',
    # data_root="data",
    # save_dir="save",
    # ES=20,
    # EPOCH=1000,
    # lr=0.001,
    # l2_lambda=0,
    # batch_size=128,
    # peptide_length=1000,
    # ligand_length=100,
    # mode='OH',
    # hidden_dim=256,
    # use_cuda=True,
    # use_radam=False,
    # show_loss=True,
    # c=0.9,
    # n=5,
    # d_model=32,
    # max_auc=0,
    # MTL=True,
    # a=0.1,
    # nhead=2,
    # dim_feedforward=2,
    # dropout=0.1,
    # num_encoder_layers=1
    # )

    # PNI_mamba
    params = dict(
    base='MAMBA',
    data_root="data",
    save_dir="save",
    ES=20,
    EPOCH=1000,
    lr=0.001,
    l2_lambda=0,
    batch_size=128,
    peptide_length=1000,
    ligand_length=100,
    mode='OH',
    use_cuda=True,
    use_radam=False,
    show_loss=True,
    c=0.9,
    n=5,
    d_model=32,
    d_state=64,
    d_conv=4,
    expand=16,
    max_auc=0,
    dp=0.1,
    MTL=True,
    a=0.1,
)

    # # PNI_mamba2
    # params = dict(
    #     base='MAMBA2',
    #     data_root="data",
    #     save_dir="save",
    #     ES=10,
    #     EPOCH=1000,
    #     lr=0.001,
    #     l2_lambda=0,
    #     batch_size=128,
    #     peptide_length=1000,
    #     ligand_length=100,
    #     mode='OH',
    #     use_cuda=True,
    #     use_radam=False,
    #     show_loss=True,
    #     c=0.9,
    #     n=5,
    #     d_model=32,
    #     d_state=64,
    #     d_conv=4,
    #     expand=16,
    #     max_auc=0,
    #     dp=0.1,
    #     MTL=True,
    #     a=0.1,
    # )

    params['modelName'] = ModelEvaluator.getModelName()
    print(params['modelName'])
    HTS()