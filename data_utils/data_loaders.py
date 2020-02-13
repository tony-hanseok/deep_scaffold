"""Data loaders"""
import typing as t

import numpy as np
from rdkit import Chem
from torch.utils import data

from data_utils import utils
from data_utils.datasets import ScaffoldMolDataset
from data_utils.samplers import ScaffoldMolSampler
from mol_spec import MoleculeSpec

__all__ = ['get_data_loader', 'get_data_loader_full']


class DataLoader(data.DataLoader):
    """Loading scaffold-molecule pair from ScaffoldDataset"""

    def __init__(self, dataset, sampler, k=5, p=0.5, num_workers=0, ms=MoleculeSpec.get_default()):
        """Construct ScaffoldLoader from the given dataset

        Args:
            dataset (ScaffoldMolDataset): The Dataset to be loaded
            sampler (ScaffoldMolSampler): The batch sampler
            k (int): Number of importance samples, default to 5
            p (float): Degree of uncertainty during route sampling, should be in (0, 1), default to 0.5
        """
        self.k = k
        self.p = p
        self.ms = ms

        super(DataLoader, self).__init__(dataset,
                                         collate_fn=self._collate_fn,
                                         batch_sampler=sampler,
                                         num_workers=num_workers)

    def _collate_fn(self, batch):
        """Collate batch

        Args:
            batch (list): A batch of scaffold-molecule pairs

        Returns: t.Tuple[np.ndarray, np.ndarray]
        """
        mol_array, logp = [], []

        for mol_smiles_i, scaffold_ids_i, nh_ids_i, np_ids_i in batch:
            # size:
            # mol_array_i : k x num_steps_i x 4
            # logp_i: k
            mol_array_i, logp_i = utils.get_array_from_mol(mol=Chem.MolFromSmiles(mol_smiles_i),
                                                           scaffold_nodes=scaffold_ids_i,
                                                           nh_nodes=nh_ids_i,
                                                           np_nodes=np_ids_i,
                                                           k=self.k,
                                                           p=self.p,
                                                           ms=self.ms)
            mol_array.append(mol_array_i)
            logp.append(logp_i)

        # Pad mol_array
        max_num_steps = max([mol_array_i.shape[1] for mol_array_i in mol_array])
        mol_array_padded = []
        for mol_array_i in mol_array:
            # pad to the same length
            num_steps_i = mol_array_i.shape[1]
            mol_array_i = np.pad(mol_array_i, pad_width=[[0, 0], [0, max_num_steps - num_steps_i], [0, 0]],
                                 mode='constant', constant_values=-1)
            mol_array_padded.append(mol_array_i)

        mol_array = np.stack(mol_array_padded, axis=0)
        logp = np.stack(logp, axis=0)

        # Output size:
        # mol_array: batch_size x k x max_num_steps x 5
        # logp: batch_size x k

        return mol_array, logp


def get_data_loader(scaffold_network_loc, molecule_smiles_loc, exclude_ids_loc, split_type, batch_size, batch_size_test,
                    num_iterations, num_workers, k, p, ms=MoleculeSpec.get_default()):
    """Helper function for getting the dataloader

    Args:
        scaffold_network_loc (str): The location to the network file
        molecule_smiles_loc (str): The location to the file containing molecular SMILES
        exclude_ids_loc (str): File storing the indices of which molecule/scaffold should be excluded
        split_type (str): The type of split, should be 'scaffold' or 'molecule'
        batch_size (int): Batch size for training
        batch_size_test (int): Batch size for test
        num_iterations (int): The number of iterations to train
        num_workers (int): The number of workers used for data loading
        k (int)
        p (float)
        ms (MoleculeSpec, optional)

    Returns:
        t.Tuple[DataLoader, DataLoader]: The loader for training data and test data
    """
    # Get dataset
    db = ScaffoldMolDataset(scaffold_network_loc, molecule_smiles_loc)

    assert batch_size % 2 == 0 and batch_size_test % 2 == 0

    sampler_train = ScaffoldMolSampler(dataset=db,
                                       batch_size=(batch_size // 2, batch_size // 2),
                                       num_iterations=num_iterations,
                                       exclude_ids_loc=exclude_ids_loc,
                                       training=True,
                                       split_type=split_type)
    sampler_test = ScaffoldMolSampler(dataset=db,
                                      batch_size=(batch_size_test // 2, batch_size_test // 2),
                                      num_iterations=num_iterations,
                                      exclude_ids_loc=exclude_ids_loc,
                                      training=False,
                                      split_type=split_type)

    # Get DataLoaders
    loader_train = DataLoader(dataset=db,
                              sampler=sampler_train,
                              k=k,
                              p=p,
                              num_workers=num_workers,
                              ms=ms)
    loader_test = DataLoader(dataset=db,
                             sampler=sampler_test,
                             k=k,
                             p=p,
                             num_workers=0,
                             ms=ms)

    return loader_train, loader_test


def get_data_loader_full(scaffold_network_loc, molecule_smiles_loc, batch_size, num_iterations, num_workers, k, p,
                         ms=MoleculeSpec.get_default()):
    """Helper function for getting the dataloader

    Args:
        scaffold_network_loc (str): The location to the network file
        molecule_smiles_loc (str): The location to the file containing molecular SMILES
        batch_size (int): Batch size for training
        num_iterations (int): The number of iterations to train
        num_workers (int): The number of workers used for data loading
        k (int)
        p (float)
        ms (MoleculeSpec, optional)

    Returns:
        t.Tuple[DataLoader, DataLoader]: The loader for training data and test data
    """
    # Get dataset
    db = ScaffoldMolDataset(scaffold_network_loc=scaffold_network_loc,
                            molecule_smiles_loc=molecule_smiles_loc)

    # batch_size
    assert batch_size % 2 == 0

    # Get sampler
    sampler = ScaffoldMolSampler(dataset=db, batch_size=(batch_size // 2, batch_size // 2),
                                 num_iterations=num_iterations, exclude_ids_loc=None, training=None, split_type=None)

    # Get DataLoaders
    loader = DataLoader(db, sampler, k, p, num_workers, ms)

    return loader
