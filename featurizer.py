from abc import ABC, abstractmethod
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, DataStructs
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
import logging
logger = logging.getLogger(__name__)

class Featurizer(ABC):
    """Abstract base class for featurizers."""

    @abstractmethod
    def featurize(self, mol):
        """Featurize a single molecule."""
        pass

    def featurize_df(self, df, smiles_col, label_col=None):
        """Featurize a DataFrame of SMILES strings."""
        features = []
        labels = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row[smiles_col]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                features.append(None)
                if label_col is not None:
                    labels.append(row[label_col])
                continue
            fp = self.featurize(mol)
            features.append(fp)
            if label_col is not None:
                labels.append(row[label_col])

        features = np.array(features)
        if label_col is not None:
            return features, np.array(labels)
        else:
            return features

class SMILESFeaturizer(Featurizer):
    def featurize(self, mol):
        """Return the SMILES string as a feature."""
        return Chem.MolToSmiles(mol) if mol is not None else None

class PhysChemFeaturizer(Featurizer):
    def __init__(self):
        # List of physicochemical descriptors to calculate
        self.descriptors = [
            Descriptors.MolWt,  # Molecular weight
            Descriptors.MolLogP,  # LogP (lipophilicity)
            Descriptors.NumHDonors,  # Number of hydrogen bond donors
            Descriptors.NumHAcceptors,  # Number of hydrogen bond acceptors
            Descriptors.TPSA,  # Topological polar surface area
        ]
        # Names for the descriptors (for readability)
        self.descriptor_names = [
            "MolWt",
            "MolLogP",
            "NumHDonors",
            "NumHAcceptors",
            "TPSA",
        ]

        # For parallel processing - no extra parameters needed
        self._parallel_init_kwargs = {}

    def featurize(self, mol):
        """Calculate physicochemical descriptors for a single molecule."""
        return [desc(mol) for desc in self.descriptors]


class PharmacophoreFeaturizer(Featurizer):
    """
    Currently slow
    """

    def __init__(self):
        self.factory = Gobbi_Pharm2D.factory

    # Number of bits in the pharmacophore fingerprint
    #  self.n_bits = self.factory.GetNumBits()
    def featurize(self, mol):
        fp = Generate.Gen2DFingerprint(mol, self.factory)
        return list(fp)


class MACCSFeaturizer(Featurizer):
    def __init__(self):
        self.n_bits = 167  # MACCS keys are always 167 bits

    def featurize(self, mol):
        """Calculate MACCS keys for a single molecule."""
        fp = MACCSkeys.GenMACCSKeys(mol)
        return list(fp)


class SubstructureCountFeaturizer(Featurizer):
    def __init__(self):
        # Define substructures using SMARTS patterns
        self.substructures = {
            "aromatic_rings": "[a]",  # Aromatic atoms
            "halogens": "[F,Cl,Br,I]",  # Halogens
            "hydroxyl": "[O;H1]",  # Hydroxyl groups (-OH)
            "amines": "[N;H2,N;H1]",  # Primary/secondary amines
            "carboxylic_acid": "[C](=O)[O;H1]",  # Carboxylic acids
            "ketones": "[C](=O)[C]",  # Ketones
            "esters": "[C](=O)[O;H0]",  # Esters
            "amides": "[C](=O)[N]",  # Amides
            # "nitro": "[N](=O)(=O)",  # Nitro groups
            # "cyano": "[C]#N",  # Cyano groups
        }

    def featurize(self, mol):
        """Count substructures in a single molecule."""
        counts = []
        for smarts in self.substructures.values():
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            counts.append(len(matches))
        return counts


class MorganFeaturizer(Featurizer):
    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits

    def featurize(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.n_bits)
        return list(fp)

    # def featurize(df, smiles_col, label_col=None):
    #     fingerprinter = CircularFingerprint()
    #     fps = []
    #     for i, row in tqdm(df.iterrows(), total=len(df)):
    #         smiles = row[smiles_col]
    #         mol = Chem.MolFromSmiles(smiles)
    #         fp = fingerprinter._featurize(mol)
    #         fps.append(fp)
    #     if label_col is not None:
    #         return np.array(fps), np.array(df[label_col])
    #     else:
    #         return np.array(fps)


#################### Trasnformers
class ChemBERTaFeaturizer(Featurizer):
    """Featurizer for generating ChemBERTa embeddings."""

    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MLM"):
        """Initialize the ChemBERTa model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        if torch.cuda.is_available():
            self.model.to("cuda")  # Move model to GPU if available

    def featurize(self, mol):
        """Generate ChemBERTa embedding for a molecule object."""
        # Convert molecule object to SMILES string
        smiles = Chem.MolToSmiles(mol) if mol is not None else None
        if smiles is None:
            return None

        # Tokenize the SMILES string
        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean pooling to get a fixed-size embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

# class MolFormerFeaturizer(Featurizer):
#     def __init__(self):
#     def featurize(self, smiles):
#         """Generate MolFormer embedding for a SMILES string."""
