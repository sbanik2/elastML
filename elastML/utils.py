from dscribe.descriptors import MBTR, SOAP
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor


# ---------------------------------
def StructureToAse(structure):
    """
    Used to create an ASE object from the structure dictionary of MCTS.
    """
    ASEobject = AseAtomsAdaptor().get_atoms(structure)
    ASEobject.set_cell(
        [
            structure.lattice.a,
            structure.lattice.b,
            structure.lattice.c,
            structure.lattice.alpha,
            structure.lattice.beta,
            structure.lattice.gamma,
        ]
    )

    return ASEobject


# ----------------------------------


def get_species(structure):
    """
    Returns all the species as a list
    """
    species = [cord.specie.symbol for cord in structure.sites]

    return species


# ---------------------------------


def Get_SOAP(structure, rcut=6, nmax=3, lmax=3):
    """
    For creation of SOAP Fingerpeinting
    """
    ASEobject = StructureToAse(structure)
    species = list(set(get_species(structure)))
    periodic_soap = SOAP(
        species=species,
        r_cut=rcut,
        n_max=nmax,
        l_max=nmax,
        periodic=True,
        sparse=False,
        average="inner",
    )

    return periodic_soap.create(ASEobject)
