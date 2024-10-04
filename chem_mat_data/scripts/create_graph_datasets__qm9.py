import os
from rich import print as pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'QM9'
__TESTING__ = False

experiment = Experiment.extend(
        'create_graph_datasets.py',
        base_path=folder_path(__file__),
        namespace=file_namespace(__file__),
        glob=globals(),
    )


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    We add the xyz_name of the molecule for identification.
    """
    graph['graph_id'] = data['xyz_name']



@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    df = load_smiles_dataset('QM9')

    dataset: dict[int, dict] = {}

    for index, data in enumerate(df.to_dict('records')):
        data['smiles'] = data['smiles']
        #We have 16 targets
        data['targets'] = [data['tag'],data['rotational_constant_A[GHz]'],data['rotational_constant_B[GHz]'],data['rotational_constant_C[GHz]'],data['dipole_moment[Debye]'],data['polarizability[Bohr³]'],data['homo_energy[Hartree]'],data['lumo_energy[Hartree]'],data['gap[Hartree]'],data['electronic_spatial_extent[Bohr²]'],data['zpve[Hartree]'],data['internal_energy_0K[Hartree]'],data['internal_energy_298K[Hartree]'],data['enthalpy_298K[Hartree]'],data['free_energy_298K[Hartree]'],data['heat_capacity_298K[cal/(mol K)]']]
        dataset[index] = data

    pprint(data)
    return dataset

experiment.run_if_main()
