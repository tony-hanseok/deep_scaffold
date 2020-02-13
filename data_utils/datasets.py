"""Dataset for loading scaffold-molecule pairs"""
import gc
import gzip
import io
import pickle
import typing as t
from os import path

from torch.utils import data

__all__ = ['ScaffoldMolDataset']


class ScaffoldMolDataset(data.Dataset):
    """A bipartite network between scaffolds and molecules."""

    def __init__(self, scaffold_network_loc=path.join(path.dirname(__file__), 'scaffolds_molecules.pkl.gz'),
                 molecule_smiles_loc=path.join(path.dirname(__file__), 'molecules.smi')):
        """
        Construct the dataset from files

        Args:
            scaffold_network_loc (str): Location of the .bin file for the bipartite network
            molecule_smiles_loc (str): Location of the molecule smiles file
        """
        # Load SMILES strings for molecules
        with open(molecule_smiles_loc) as f:
            self._molecules = f.read().splitlines()

        # Compile the mapping between scaffold and molecule
        gc.disable()
        with gzip.open(scaffold_network_loc, 'rb') as f:
            self._scaffold2molecule, self._molecule2scaffold, self._atom_labels = pickle.load(io.BufferedReader(f))
        gc.enable()

        # All scaffold id
        self.scaffold_ids: t.List[int] = list(self._scaffold2molecule.keys())

        # All molecule id
        self.molecule_ids: t.List[int] = list(self._molecule2scaffold.keys())

    def get_mol_from_scaffold(self, scaffold_id):
        """Get the set of all molecule(ids) containing a given scaffold

        Args:
            scaffold_id: The id of the given scaffold

        Returns:
            t.Set[int]: The set of all molecule(ids) containing a given scaffold
        """
        return self._scaffold2molecule[scaffold_id]

    def get_scaffold_from_mol(self, mol_id):
        """Get the set of all scaffolds(id) inside a give molecule

        Args:
            mol_id (int): The id of the given molecule

        Returns:
            t.Set[int]: The set of all scaffolds(id) inside a give molecule
        """
        return self._molecule2scaffold[mol_id]

    def get_atom_mapping(self, scaffold_id, mol_id):
        """Get the atom mapping between a given molecule(id) and a given scaffold (id)

        Args:
            scaffold_id (int): The id of the given scaffold
            mol_id (int): The id of the given molecule

        Returns:
            tuple: The atom mapping between a given molecule(id) and a given scaffold (id)
        """
        return self._atom_labels[scaffold_id, mol_id]

    def get_item(self, scaffold_id, mol_id, record_id=None):
        """A explicit version of __getitem__

        Args:
            scaffold_id (t.Optional[int]): The id of the scaffold query, could be None
            mol_id (t.Optional[int]): The id of the molecule query, could be None
            record_id (t.Optional[int]): The id of the record query, could be None
        """
        if scaffold_id is None:
            assert mol_id is not None
            assert record_id is None
            return self.get_scaffold_from_mol(mol_id)

        if mol_id is None:
            assert record_id is None
            return self.get_mol_from_scaffold(scaffold_id)

        if record_id is None:
            return self.get_atom_mapping(scaffold_id, mol_id)

        atom_ids, nh_ids, np_ids = self._atom_labels[(scaffold_id, mol_id)][record_id]
        mol_smiles = self._molecules[mol_id]
        return mol_smiles, atom_ids, nh_ids, np_ids

    def __getitem__(self, item):
        """
        A wrapped version of self.get_item
        """
        return self.get_item(*item)

    def __len__(self):
        """
        Just do nothing.
        Required by pytorch data api
        """
        raise NotImplementedError
