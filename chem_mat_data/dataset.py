from chem_mat_data.processing import MoleculeProcessing


# === DATSET IMPLEMENTATION ===

class StreamingDataset:
    
    def __init__(self):
        pass
    
    def __iter__(self):
        raise NotImplementedError()


class SmilesDataset:

    def __init__(
        self,
        dataset: str,
    ) -> None:
        
        self.dataset = dataset
        
        
class GraphDataset:
    
    def __init__(
        self,
        dataset: str,
        num_workers: int = 2,
        processing_class: type = MoleculeProcessing,
    ) -> None:
        
        self.dataset = dataset
        