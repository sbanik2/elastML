import os

import joblib
import numpy as np
import pandas as pd
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition.element import Stoichiometry
from matminer.featurizers.conversions import StrToComposition
from pymatgen.core import Structure

from .utils import Get_SOAP

pd.options.mode.chained_assignment = None


class FeaExtraction:
    def __init__(self, materials_id, api_key):
        self.materials_id = materials_id
        self.api_key = api_key
        self.data = self.getMatProjectData(materials_id, api_key)
        self.data = StrToComposition().featurize_dataframe(self.data, "pretty_formula")
        self.local_fea = self.getLocalFea()
        self.local_soap = self.getSOAPfea()
        self.global_fea = self.getGlobalFea()

    def getMatProjectData(self, materials_id, api_key):
        mpr = MPDataRetrieval(api_key)

        df = mpr.get_dataframe(
            criteria=materials_id,
            properties=[
                "material_id",
                "pretty_formula",
                "structure",
                "energy_per_atom",
                "formation_energy_per_atom",
                "e_above_hull",
                "volume",
                "nsites",
                "spacegroup.number",
                "spacegroup.point_group",
                "spacegroup.crystal_system",
                "band_gap",
                "density",
                "total_magnetization",
                "elasticity.K_VRH",
                "elasticity.G_VRH",
                "warnings",
            ],
            index_mpid=False,
        )

        df = df.rename(
            columns={
                "spacegroup.number": "SG",
                "spacegroup.point_group": "point_group",
                "spacegroup.crystal_system": "crystal_system",
                "elasticity.K_VRH": "K_VRH",
                "elasticity.G_VRH": "G_VRH",
            }
        )

        df = df.set_index("material_id").loc[materials_id["material_id"]["$in"]]
        df = df.reset_index(drop=True)

        # df.to_csv("df_mat.csv")

        return df

    def getGlobalFea(self):
        df_subset = self.data[
            [
                "structure",
                "composition",
            ]
        ]

        st = Stoichiometry()
        global_stoichiometric = st.featurize_dataframe(df_subset, col_id="composition")
        global_stoichiometric = global_stoichiometric.drop(
            columns=["structure", "composition"]
        )

        crystal_system = [
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic",
        ]
        point_groups = [
            "1",
            "-1",
            "2",
            "m",
            "2/m",
            "222",
            "mm2",
            "mmm",
            "4",
            "-4",
            "4/m",
            "422",
            "4mm",
            "-42m",
            "4/mmm",
            "3",
            "-3",
            "32",
            "3m",
            "-3m",
            "6",
            "-6",
            "6/m",
            "622",
            "6mm",
            "-6m2",
            "6/mmm",
            "23",
            "m-3",
            "432",
            "-43m",
            "m-3m",
        ]
        laue_centrosymmetric = [
            "-1",
            "2/m",
            "mmm",
            "4/m",
            "4/mmm",
            "-3",
            "-3m",
            "6/m",
            "6/mmm",
            "m-3",
            "m-3m",
        ]
        rotation_axes_enantiomorphic = [
            "1",
            "2",
            "222",
            "4",
            "422",
            "3",
            "32",
            "6",
            "622",
            "23",
            "432",
        ]
        polar_type = [
            "m",
            "mm2",
            "-4",
            "4mm",
            "-42m",
            "3m",
            "-6",
            "6mm",
            "-6m2",
            "-43m",
        ]

        ptgroups = [point_groups.index(val) for val in self.data.point_group]
        laue = []
        for val in self.data.point_group:
            if val in laue_centrosymmetric:
                laue.append(laue_centrosymmetric.index(val) + 1)
            else:
                laue.append(0)

        enantiomorphic = []
        for val in self.data.point_group:
            if val in rotation_axes_enantiomorphic:
                enantiomorphic.append(rotation_axes_enantiomorphic.index(val) + 1)
            else:
                enantiomorphic.append(0)

        polar = []

        for val in self.data.point_group:
            if val in polar_type:
                polar.append(polar_type.index(val) + 1)
            else:
                polar.append(0)

        crys_system = [crystal_system.index(val) for val in self.data.crystal_system]

        columns = [
            "crystal_system",
            "spacegroup",
            "pointgroup",
            "laue",
            "enantiomorphic",
            "polar",
        ]
        global_structural = pd.DataFrame(
            np.array(
                [
                    crys_system,
                    self.data.SG.values.tolist(),
                    ptgroups,
                    laue,
                    enantiomorphic,
                    polar,
                ]
            ).T,
            columns=columns,
        )

        global_property = self.data[
            [
                "energy_per_atom",
                "formation_energy_per_atom",
                "e_above_hull",
                "band_gap",
                "density",
                "total_magnetization",
            ]
        ]
        global_property["vpa"] = self.data.volume.values / self.data.nsites.values

        global_features = pd.concat(
            [global_structural, global_property, global_stoichiometric], axis=1
        )

        return global_features

    def getLocalFea(self):
        df_subset = self.data[
            [
                "structure",
                "composition",
            ]
        ]

        stats = ["minimum", "maximum", "range", "mean", "avg_dev", "mode"]

        # --------features from magpie----------------
        feat_magpie = ElementProperty.from_preset(preset_name="magpie")
        df_magpie = feat_magpie.featurize_dataframe(
            df_subset, col_id="composition", ignore_errors=True
        )

        # --------features from deml----------------
        feat_deml = ElementProperty.from_preset(preset_name="deml")
        feat_deml.stats = stats
        df_deml = feat_deml.featurize_dataframe(
            df_subset, col_id="composition", ignore_errors=True
        )

        # ----------features from matminer----------------
        feat_matminer = ElementProperty.from_preset(preset_name="matminer")
        feat_matminer.stats = stats
        df_matminer = feat_matminer.featurize_dataframe(
            df_subset, col_id="composition", ignore_errors=True
        )

        # ------------catagorical features----------

        periodic_table = [
            "Number",
            "MendeleevNumber",
            "AtomicWeight",
            "Column",
            "Row",
            "group",
            "block",
            "CovalentRadius",
        ]

        electronic_structure = [
            "NsValence",
            "NpValence",
            "NdValence",
            "NfValence",
            "NValence",
            "NsUnfilled",
            "NpUnfilled",
            "NdUnfilled",
            "NfUnfilled",
            "NUnfilled",
        ]

        property_based = [
            "MeltingT",
            "Electronegativity",
            "GSvolume_pa",
            "GSbandgap",
            "GSmagmom",
            "SpaceGroupNumber",
            "heat_fusion",
            "boiling_point",
            "heat_cap",
            "first_ioniz",
            "electric_pol",
            "electrical_resistivity",
            "thermal_conductivity",
            "velocity_of_sound",
            "coefficient_of_linear_thermal_expansion",
        ]

        df_temp = pd.concat([df_magpie, df_deml, df_matminer], axis=1)

        total_features = periodic_table + electronic_structure + property_based

        # ===========rename and cleaning features=================

        sel_columns = []

        for feat in total_features:
            for col in df_temp.columns:
                if " " + feat in col:
                    #            print(feat)
                    sel_columns.append(col)

        df_local = df_temp[sel_columns]

        columns = {}

        to_replace = ["MagpieData ", "DemlData ", "PymatgenData "]

        for col in sel_columns:
            newcol = None
            for rep in to_replace:
                if rep in col:
                    newcol = col.replace(rep, "")

            newcol = newcol.replace(" ", "_")
            columns[col] = newcol

        df_local = df_local.rename(columns=columns)

        to_remove = [
            "coefficient_of_linear_thermal_expansion",
            "electric_pol",
            "heat_cap",
            "electrical_resistivity",
            "velocity_of_sound",
        ]

        col_to_remove = []

        for rm in to_remove:
            for col in df_local.columns:
                if rm in col:
                    col_to_remove.append(col)

        df_local = df_local.drop(columns=col_to_remove)

        return df_local

    def getSOAPfea(self):
        df_subset = self.data[["structure"]]

        FP = []

        for i, struct in enumerate(df_subset.structure):
            pos = [list(site.frac_coords) for site in struct]
            species = ["Al" for _ in range(len(pos))]
            newstruct = Structure(struct.lattice, species, pos, to_unit_cell=True)
            FP.append(Get_SOAP(newstruct, rcut=6, nmax=3, lmax=3))

        FP = np.vstack(FP)

        df_local_soap = pd.DataFrame(
            FP, columns=["F{}".format(i) for i in range(FP.shape[1])]
        )

        return df_local_soap

    def get_features(self):
        imputer_path = os.path.join(os.path.dirname(__file__), "transform/Imputer")
        imputer = joblib.load(imputer_path)

        predict_imputed = imputer.transform(self.local_fea.values)

        local = pd.DataFrame(predict_imputed, columns=self.local_fea.columns)

        df = pd.concat([local, self.local_soap, self.global_fea], axis=1)

        return df

    def get_predict_data(
        self,
        feature_relevance="mrmr_precomputed",
        target="Bulk",  # or "Shear"
        n_features=150,
    ):
        feature_relevance_path = os.path.join(
            os.path.dirname(__file__),
            "mrmr_relevance/relevance_{}_modulus.csv".format(target),
        )
        feature_relevance = pd.read_csv(feature_relevance_path)

        transform_path = os.path.join(
            os.path.dirname(__file__), "transform/Transform_{}".format(target)
        )

        transform = joblib.load(transform_path)

        df = self.get_features()

        if target == "Bulk":
            target = self.data[["K_VRH"]]
        elif target == "Shear":
            target = self.data[["G_VRH"]]
        else:
            raise Exception("target can either be Bulk or Shear")

        names = feature_relevance.iloc[:, 0].values
        imporatnce = feature_relevance.iloc[:, 1].values
        c = zip(names, imporatnce)
        c = sorted(c, key=lambda x: -x[1])
        names, imporatnce = zip(*c)

        predict = df[list(names[:n_features])]

        # predict.to_csv("predict.csv")

        predict_transform = transform.transform(predict)

        # pd.DataFrame(predict_transform, columns=predict.columns).to_csv(
        #    "predict_transform.csv"
        # )

        return predict_transform, target
