"""The evaluation system

A system for model evaluation based on zerorpc. The system is composed of three
basic components:
1. Worker: Processes that performs actuall evaluatioin tasks.
2. Dispatcher: The server responsible for dispatching jobs/tasks to workers.
3. Collector: The process that collects results generated by workers.
Those components collaborates in the following way:
1. Dispatcher <-> Worker:
After intialization (or after each task), worker would voluntarily request a
new task/job from the dispatcher. The dispatcher manages a list of unassigned
tasks.
2. Worker <-> Collector:
When a worker finish the execution of a task, the result is forwarded to the
collector, and it is the collector's responsibility to store the result of the
task.
"""
import gc
import gzip
import io
import json
import os
import pickle
import random
import sys
import typing as t

import multiprocess as mp
import numpy as np
import torch
import zerorpc
from cupyx.scipy.sparse import csr_matrix as csr_matrix_cu
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from scipy.sparse import csr_matrix
from torch import nn

from data_utils import get_array_from_mol
from data_utils import get_mol_from_array
from deep_scaffold import DeepScaffold


def build_model(ckpt_loc):
    """Building model from checkpoint file"""
    # Configure default parameters
    config = {
        "num_atom_embedding": 16,
        "causal_hidden_sizes": (32, 64),
        "num_bn_features": 96,
        "num_k_features": 24,
        "num_layers": 20,
        "num_output_features": 256,
        "efficient": False,
        "activation": 'elu'
    }

    # Local configuration file
    with open(os.path.join(ckpt_loc, 'config.json')) as f:
        config_update = json.load(f)
    # Update default configuration with config

    for key in config_update:
        if key in config:
            config[key] = config_update[key]

    # Build model
    mdl = DeepScaffold(**config)

    # Load checkpoint
    mdl = nn.DataParallel(mdl)
    mdl.load_state_dict(torch.load(os.path.join(ckpt_loc, 'mdl.ckpt')))

    # Unwrap from nn.DataParallel, move to GPU
    mdl = mdl.module.cuda(0).eval()

    return mdl


def sample(mdl, scaffold_smi, num_samples):
    """Generate `num_samples` samples from the model `mdl` based on a given scaffold with SMILES `scaffold_smi`.

    Args:
        mdl (DeepScaffold): The scaffold-based molecule generative model
        scaffold_smi (str): The SMILES string of the given scaffold
        num_samples (int): The number of samples to generate

    Returns:
        t.Tuple[t.List[t.Union[str, None]], float, float]: The generated molecules. Molecules that does not satisfy the
                                                           validity requirements are returned as `None`
    """
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Convert SMILES to molecule
    scaffold = Chem.MolFromSmiles(scaffold_smi)

    # Convert molecule to numpy array
    # shape: 1, ..., 5
    scaffold_array, _ = get_array_from_mol(mol=scaffold,
                                           scaffold_nodes=range(scaffold.GetNumHeavyAtoms()),
                                           nh_nodes=[], np_nodes=[], k=1, p=1.0)

    # Convert numpy array to torch tensor
    # shape: 1, ..., 5
    scaffold_tensor = torch.from_numpy(scaffold_array).long().cuda()

    # Generate
    with torch.no_grad():
        # Expand the first dimension
        # shape: num_samples, ..., 5
        scaffold_tensor = scaffold_tensor.expand(num_samples, -1, -1)
        # Generate samples
        # shape: [num_samples, -1, 5]
        mol_array = mdl.generate(scaffold_tensor)

    # Move to CPU
    mol_array = mol_array.detach().cpu().numpy()

    # Convert numpy array to Chem.Mol object
    mol_list = get_mol_from_array(mol_array, sanitize=True)

    # Convert Chem.Mol object to SMILES
    def _to_smiles(_mol):
        if _mol is None:
            return None
        try:
            _smiles = Chem.MolToSmiles(_mol)
        except ValueError:
            # If the molecule can not be converted to SMILES, return None
            return None

        # If the output SMILES is None, return None
        if _smiles is None:
            return None

        # Make sure that the SMILES can be convert back to molecule
        try:
            _mol = Chem.MolFromSmiles(_smiles)
        except ValueError:
            # If there are any error encountered during the process,
            # return None
            return None

        # If the output molecule object is None, return None
        if _mol is None:
            return None
        return _smiles

    smiles_list = list(map(_to_smiles, mol_list))

    # Get the validity statistics
    num_valid = sum(1 for _ in smiles_list if _ is not None)
    percent_valid = float(num_valid) / len(smiles_list)

    # Get the uniqueness statistics
    num_unique = len(set(smiles_list)) - 1
    percent_unique = float(num_unique) / num_valid

    return smiles_list, percent_valid, percent_unique


def sample_batch(mdl, scaffold_smi, num_samples, batch_size):
    """Sample (a relatively large amount of) molecules by splitting the total number into smaller batches

    Args:
        mdl (DeepScaffold): The scaffold-based molecule generative model
        scaffold_smi (str): The SMILES string of the given scaffold
        num_samples (int): The number of samples to generate
        batch_size (int): The number of samples to generate at each time

    Returns:
        t.List[str]: The list of all molecules sampled
    """
    sample_list = []
    while len(sample_list) < num_samples:
        new_samples, _, _ = sample(mdl, scaffold_smi, batch_size)

        # Filter None molecules
        new_samples = list(filter(lambda _x: _x is not None, new_samples))

        # Append to sample list
        sample_list = sample_list + new_samples
    sample_list = sample_list[:num_samples]
    return sample_list


def get_fingerprints(smiles_list, mapper):
    """Getting the fingerprint for a list of molecules

    Args:
        smiles_list (t.List[str]): The list of molecule to get fingerprint from
        mapper (t.Callable): The mapping function. Could be map, pool.map or pool.imap

    Returns:
        csr_matrix: The fingerprint information of the molecule stored inside a sprase matrix.
                    dtype: np.float32, shape: [num_samples, 1024]
    """
    # Defining the length of the fingerprint
    fp_length = 1024

    def get_on_bits(smiles: str) -> t.List[int]:
        """Function for getting on-bits from smiles string

        Args:
            smiles (str): The smiles string of the input molecule

        Returns:
            t.List[int]: The location of the on-bits
        """
        mol = Chem.MolFromSmiles(smiles)  # Parse SMILES string
        assert mol is not None

        # Get fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, fp_length)
        on_bits = list(fp.GetOnBits())  # Store on bits only
        return on_bits

    def reduce_fn(row_col_list):
        row_list, col_list = [], []
        for row_i, col_i in row_col_list:
            row_list.append(row_i)
            col_list.append(col_i)
        row_cat, col_cat = np.concatenate(row_list), np.concatenate(col_list)
        return row_cat, col_cat

    def map_fn(args):
        row_id, smiles = args
        col = np.array(get_on_bits(smiles), dtype=np.int32)
        row = np.full_like(col, row_id)
        return row, col

    # Get the row and column indices
    mapped = list(mapper(map_fn, enumerate(smiles_list)))
    row, col = reduce_fn(mapped)

    # Wrap `row` and `col` into a sparse matrix
    d = np.ones_like(row, dtype=np.float32)
    shape = (len(smiles_list), fp_length)
    fp_mat = csr_matrix((d, (row, col)), shape=shape, dtype=np.float32)

    return fp_mat


def get_tanimoto(fp_1, fp_2):
    """Get the matrix of tanimoto similarity between two molecule sets

    Args:
        fp_1 (csr_matrix_cu)
        fp_2 (csr_matrix_cu): The two sets of molecules represented as sparse fingerprint
                                                    matrices (in gpu)

    Returns:
        cp.ndarray: The similarity matrix calculated
    """
    # Calculate the dot product
    dot_prod = fp_1.dot(fp_2.T).A

    # Calculate the tanimoto similarity
    sim_mat = dot_prod / (fp_1.sum(-1) + fp_2.sum(-1).T - dot_prod)
    return sim_mat


def get_mmd(smiles_list_1, smiles_list_2, mapper):
    """Calculate the MMD between two molecule sets

    Args:
        smiles_list_1 (t.List[str])
        smiles_list_2 (t.List[str]): The two molecule sets represented as SMILES lists
        mapper (t.Callable): The mapper, can be map, pool.map or pool.imap

    Returns:
        t.Tuple[float, float, float]: The diversity of the two molecule sets, as well as the MMD calculated
    """
    # Get the number of molecules
    size_1, size_2 = len(smiles_list_1), len(smiles_list_2)
    # Get fingerprints
    fp_1 = get_fingerprints(smiles_list_1, mapper)
    fp_2 = get_fingerprints(smiles_list_2, mapper)

    # Moving fingerprint to gpu
    fp_1, fp_2 = csr_matrix_cu(fp_1), csr_matrix_cu(fp_2)

    # Calculating similarity matrix (kernel matrix)
    k_11 = get_tanimoto(fp_1, fp_1)
    k_12 = get_tanimoto(fp_1, fp_2)
    k_22 = get_tanimoto(fp_2, fp_2)

    # Calculate diversity
    diversity_1 = (k_11.sum() - size_1) / (size_1 * (size_1 - 1))
    diversity_2 = (k_22.sum() - size_2) / (size_2 * (size_2 - 1))
    # Calculate MMD
    mmd = diversity_1 + diversity_2 - 2 * k_12.sum() / (size_1 * size_2)

    # Convert to float, return
    return diversity_1.item(), diversity_2.item(), mmd.item()


def get_properties(smiles_list, mapper):
    """Get the properties of a give list of molecules

    Args:
        smiles_list (t.List[str]): The list of molecule to process (represented as molecular SMILES)
        mapper (t.Callable): The mapping function. Could be map, pool.map or pool.imap

    Returns:
        t.List[t.Tuple[float, float, float]]: The calculated properties (molecular weight, logp and QED)
    """

    def get_mol_props(smiles):
        """Get the molecular properties of a single molecule"""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        mol_wt = Descriptors.MolWt(mol)
        log_p = Descriptors.MolLogP(mol)
        qed = QED.qed(mol)
        assert mol_wt is not None and log_p is not None and qed is not None

        return mol_wt, log_p, qed

    prop_list = mapper(get_mol_props, smiles_list)
    return prop_list


def get_prop_stat(smiles_list, mapper):
    """Get the statistics for two molecule sets based on several molecular properties

    Args:
        smiles_list (t.List[str]): Input molecule sets
        mapper (t.Callable): The mapper, can be map, pool.map or pool.imap

    Returns:
        t.Tuple[np.ndarray, ...]: The calculated statistics
    """
    # Get the sample size
    sample_size = len(smiles_list)

    # Get molecular properties for each set
    prop_list = get_properties(smiles_list, mapper)

    # Convert to numpy array
    prop_array = np.array(prop_list)

    # Calculate mean and variance of each property
    mu_mean, mu_var = prop_array.mean(0), prop_array.var(0, ddof=1)

    # Calculate the confidence of the mean and variance estimator
    sigma_mean = np.sqrt(mu_var / (sample_size - 1))
    sigma_var = np.sqrt(2.0 / (sample_size - 1)) * mu_var

    # Concat together
    prop_stat = np.stack([mu_mean, mu_var, sigma_mean, sigma_var], axis=0)
    return prop_stat


class Dispatcher:
    """Task dispatcher"""

    def __init__(self, scaffold_loc, molecule_loc, scaffold_network_loc):
        """Constructor

        Args:
            scaffold_loc (str): The location of scaffold SMILES file
            molecule_loc (str): The location of the molecule SMILES file
            scaffold_network_loc (str): The location of the scaffold-molecule network file
        """
        # Load SMILES list
        self.scaffold_list = []

        with open(scaffold_loc) as f:
            for line in f:
                self.scaffold_list.append(line.rstrip().split('\t')[0])
        self.molecule_list = []

        with open(molecule_loc) as f:
            for line in f:
                self.molecule_list.append(line.rstrip())

        # Load scaffold-molecule network
        gc.disable()
        with gzip.open(scaffold_network_loc, 'rb') as f:
            # Compile the mapping between scaffold and molecule
            self.scaffold2molecule = pickle.load(io.BufferedReader(f))
        gc.enable()

        self.task_ids = list(self.scaffold2molecule.keys())
        random.shuffle(self.task_ids)
        print('Dispatcher ready, waiting request from clients ...')

    def dispatch(self):
        """Dispatch task to client"""
        if self.task_ids:
            scaffold_id = self.task_ids.pop()
            scaffold_smiles = self.scaffold_list[scaffold_id]
            molecule_ids = self.scaffold2molecule[scaffold_id]
            molecule_smiles_list = [self.molecule_list[molecule_id]
                                    for molecule_id in molecule_ids]
            message = {
                'message_type': 'task_message',
                'scaffold_smiles': scaffold_smiles,
                'molecule_smiles_list': molecule_smiles_list
            }
            print(f'Sending message to client, with task id {scaffold_id}')
            return message

        message = {
            'message_type': 'none_message'
        }
        return message


class Collector:
    """Collecting results from the clients"""

    def __init__(self, save_loc):
        """Constructor

        Args:
            save_loc (str): The location to save the results
        """
        self.f = open(save_loc, 'w')

    def __enter__(self):
        """Entering the context"""
        print('Collector ready, waiting request from clients ...')
        return self

    def __exit__(self, type, value, traceback):
        """Exit the context"""
        print('Exit collector')
        self.f.__exit__()

    def collect(self, results):
        """Save the message to files"""
        result_str = json.dumps(results)
        result_str = result_str.replace('\r', '').replace('\n', '')
        self.f.write(result_str + '\n')
        self.f.flush()
        print('Message received and saved!')


def worker(url_dispatcher, url_collector, ckpt_loc, num_workers, num_samples):
    """The worker definition

    Args:
        url_dispatcher: The url for the dispatcher
        url_collector: The url for the collector
        ckpt_loc: The location for the checkpoint
        num_workers: The number of CPU workers used
        num_samples: The number of samples
    """
    # Initialize clients
    client_dispatcher = zerorpc.Client()
    client_collector = zerorpc.Client()

    # Connect to client
    client_dispatcher.connect(url_dispatcher)
    client_collector.connect(url_collector)

    # Load model
    mdl = build_model(ckpt_loc)

    # Build multiprocessing pool
    pool = mp.Pool(num_workers)

    # Create mapper
    def mapper(func, iterable):
        return list(pool.map(func, iterable, chunksize=100))

    while True:
        message = client_dispatcher.dispatch()
        if message['message_type'] == 'none_message':
            break
        else:
            # Get sample the list of molecules
            smiles_sampled, percent_valid, percent_unique = sample(mdl=mdl,
                                                                   scaffold_smi=message['scaffold_smiles'],
                                                                   num_samples=num_samples)
            # Filter out None molecules
            smiles_sampled = list(filter(lambda _x: _x is not None,
                                         smiles_sampled))

            # Calculate property statistics
            prop_stat = np.stack((get_prop_stat(smiles_sampled, mapper),
                                  get_prop_stat(message['molecule_smiles_list'], mapper)),
                                 axis=0)
            prop_stat = prop_stat.tolist()

            # Get diversity and MMD
            diversity_1, diversity_2, mmd = get_mmd(smiles_list_1=smiles_sampled,
                                                    smiles_list_2=message['molecule_smiles_list'],
                                                    mapper=mapper)
            result_message = {
                'message_type': 'result_message',
                'scaffold_smiles': message['scaffold_smiles'],
                'molecule_sampled': (smiles_sampled[:100]
                                     if len(smiles_sampled) > 100
                                     else smiles_sampled),
                'num_test': len(message['molecule_smiles_list']),
                'percent_valid': percent_valid,
                'percent_unique': percent_unique,
                'prop_stat': prop_stat,
                'diversity_test': diversity_2,
                'diversity_sample': diversity_1,
                'mmd': mmd
            }
            client_collector.collect(result_message)


def main(args):
    """Entrypoint"""
    command = args[0]
    if command == 'dispatch':
        server = zerorpc.Server(Dispatcher(scaffold_loc='data_utils/scaffolds.smi',
                                           molecule_loc='data_utils/molecules.smi',
                                           scaffold_network_loc='scaffolds_molecules_test.pkl.gz'))
        server.bind("tcp://0.0.0.0:4242")
        server.run()
    elif command == 'collect':
        with Collector('results.txt') as collector:
            server = zerorpc.Server(collector)
            server.bind("tcp://0.0.0.0:4243")
            server.run()
    elif command == 'worker':
        url_dispatcher, url_collector = args[1:]
        num_samples = 10000
        num_workers = 5
        worker(url_dispatcher=url_dispatcher,
               url_collector=url_collector,
               ckpt_loc="ckpt/ckpt-default",
               num_workers=num_workers,
               num_samples=num_samples)
    elif command == 'sample':
        # Get arguments
        scaffold_smi, output_loc, ckpt_loc = args[1:]

        # Build model
        mdl = build_model(ckpt_loc)

        # Perform sampling
        num_samples = 100000
        batch_size = 10000
        sampled_smiles = sample_batch(mdl=mdl,
                                      scaffold_smi=scaffold_smi,
                                      num_samples=num_samples,
                                      batch_size=batch_size)

        # Save to file
        with open(output_loc, 'w') as f:
            for smiles_i in sampled_smiles:
                f.write(f'{smiles_i}\n')


if __name__ == "__main__":
    main(sys.argv[1:])
