import pandas as pd


def load_smiles_csv(path: str) -> pd.DataFrame:
    smiles_csv = pd.read_csv(path, index_col=0)

    if "Smiles" not in smiles_csv.columns:
        raise KeyError("Expected column Smiles not found")

    if "Indices" not in smiles_csv:
        smiles_csv.reset_index(inplace=True)
        smiles_csv.rename(columns={"index": "Indices"}, inplace=True)

    return smiles_csv
