model = None
class Chem:
    def Mol():
        pass
Descriptors = None

def desired_properties(mol: Chem.Mol) -> float:
    """
    Objective function to be minimized
    """
    
    # Deterministic Properties
    logp = Descriptors.MolLogP(mol)
    objective_value = abs(10 - logp)
    
    # AI Predictions
    homo = model.predict(mol)
    objective_value += homo
    
    return objective_value