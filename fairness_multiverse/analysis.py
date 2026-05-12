"""
This module contains wrapper functions to analyze the output from a multiverse
analysis esp. in regards to conducting a FANOVA.
"""

from typing import Optional
import pandas as pd

from pathlib import Path
#from smac.configspace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.preprocessing import OrdinalEncoder

from fanova import fANOVA


class MultiverseFanova:
    def __init__(self, features: pd.DataFrame, outcome: pd.Series) -> None:
        """
        Initializes a new MultiverseFanova instance.

        MiltiverseFanova is a helper class to perform a fANOVA analysis on
        data from a multiverse analysis.

        Args:
        - features: A pandas DataFrame containing the features / decisions to
            be used in the fANOVA analysis.
        - outcome: A pandas Series containing the outcome variable to be
            used in the fANOVA analysis.
        """
        self.fanova_features = self.process_features(features)
        self.fanova_outcome = outcome

        self.configuration_space = self.generate_configuration_space()
        self.fanova = self.init_fanova()


    def process_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses a set of features for a fANOVA analysis.

        Args:
        - features: A pandas DataFrame containing the features to be encoded.

        Returns:
        - A pandas DataFrame containing the encoded features.
        """
        return pd.DataFrame(
            OrdinalEncoder().fit(features).transform(features), columns=features.columns
        )

    def generate_configuration_space(self) -> ConfigurationSpace:
        """
        Generates a ConfigurationSpace object based on the instance features.

        Returns:
        - A ConfigurationSpace object.
        """
        cs = ConfigurationSpace()
        for colname in self.fanova_features.columns:
            col = self.fanova_features[colname]
            cs.add_hyperparameter(
                CategoricalHyperparameter(name=colname, choices=list(col.unique()))
            )
        return cs

    def init_fanova(self):
        """
        Initializes a fanova.fANOVA object.

        This will not yet run the analysis to compute importance measures.

        Returns:
        - An fANOVA object.
        """
        return fANOVA(
            self.fanova_features,
            self.fanova_outcome,
            config_space=self.configuration_space,
        )

    def quantify_importance(self, save_to: Optional[Path] = None):
        """
        Quantifies the joint importance of features in the MultiverseFanova.

        Args:
        - save_to: A Path specifying where to save the results. (optional)

        Returns:
        - A pandas DataFrame containing the importance of each feature.
        """
        param_list = [
            param.name for param in self.configuration_space.get_hyperparameters()
        ]
        fanova_all_effects = self.fanova.quantify_importance(param_list)
        df_importance_interactions = (
            pd.DataFrame(fanova_all_effects).transpose().reset_index(drop=False)
        )
        df_importance_interactions.sort_values(
            by="individual importance", ascending=False, inplace=True
        )

        if save_to is not None:
            df_importance_interactions.to_csv(save_to)

        return df_importance_interactions

    def quantify_individual_importance(self, save_to: Optional[Path] = None):
        """
        Quantifies the individual importance of each feature in the MultiverseFanova.

        Args:
        - save_to: A Path specifying where to save the results. (optional)

        Returns:
        - A pandas DataFrame containing the individual importance of each feature.
        """
        param_list = [
            param.name for param in self.configuration_space.get_hyperparameters()
        ]

        main_effects = {}
        for param in param_list:
            param_fanova = (param,)
            main_effects[param] = self.fanova.quantify_importance(param_fanova)[
                param_fanova
            ]

        df_main_effects = pd.DataFrame(main_effects).transpose()
        df_main_effects.sort_values(by="individual importance", ascending=False)

        if save_to is not None:
            df_main_effects.to_csv(df_main_effects)

        return df_main_effects
    

def pretty_label(raw: str) -> str:
    """Turn raw feature names like 'universe_model_gbm' into 'Model: GBM'."""
    # strip pipeline prefix if present
    if "__" in raw:
        raw = raw.split("__", 1)[1]

    # remove 'universe_' prefix if present
    if raw.startswith("universe_"):
        raw = raw[len("universe_"):]

    # split factor and level
    parts = raw.split("_", 1)
    if len(parts) == 1:
        return raw.replace("_", " ").title()
    factor, level = parts[0], parts[1]

    # friendly mapping
    if factor == "scale":
        level_map = {"no-scale": "No scaling", "not-scaled": "No scaling",
                     "scaled": "Scaled", "yes": "Scaled"}
        return f"Scaling: {level_map.get(level, level.replace('_',' ').title())}"

    if factor == "model":
        level_map = {
            "elasticnet": "Elastic Net",
            "logreg": "Logistic Regression",
            "pen_logreg": "Penalized Logistic Regression",
            "gbm": "GBM",
            "rf": "Random Forest",
            "rand_forest": "Random Forest",
        }
        return f"Model: {level_map.get(level, level.replace('_',' ').title())}"

    if factor == "training":
        if level.startswith("size_"):
            return f"Training size: {level.split('size_')[1]}"
        if level.startswith("year_"):
            return f"Training period: {level.split('year_')[1].replace('_', '–')}"
        return f"Training: {level.replace('_',' ').title()}"

    if factor == "training_size":
        return f"Training size: {level}"

    if factor == "training_year":
        return f"Training period: {level.replace('_','–')}"

    if factor == "exclude_features":
        lv_map = {
            "none": "none", "age": "age", "sex": "sex",
            "nationality": "nationality", "nationality-sex": "nationality×sex",
        }
        return f"Excluded features: {lv_map.get(level, level.replace('_',' '))}"

    if factor == "exclude_subgroups":
        lv_map = {"keep-all": "keep all", "drop-non-german": "drop non-German"}
        return f"Subgroups: {lv_map.get(level, level.replace('_',' '))}"

    # fallback
    return raw.replace("_", " ").title()


def pretty_baseline(col, level):
    # strip 'universe_' prefix
    key = col.replace("universe_", "")
    if key == "scale":
        return f"Scaling={ 'Scaled' if level=='scale' else 'No scaling' }"
    if key == "model":
        name = {
            "logreg": "Logistic Regression",
            "elasticnet": "Elastic Net",
            "pen_logreg": "Penalized Logistic Regression",
            "gbm": "GBM",
            "rf": "Random Forest",
        }.get(level, level)
        return f"Model={name}"
    if key == "training_size":
        return f"Training size={level}"
    if key == "training_year":
        return f"Training period={level.replace('_','–')}"
    if key == "exclude_features":
        name = {
            "none": "none",
            "age": "age",
            "sex": "sex",
            "nationality": "nationality",
            "nationality-sex": "nationality×sex",
        }.get(level, level)
        return f"Excluded features={name}"
    if key == "exclude_subgroups":
        name = {"keep-all": "keep all", "drop-non-german": "drop non-German"}.get(level, level)
        return f"Subgroups={name}"
    return f"{key}={level}"
