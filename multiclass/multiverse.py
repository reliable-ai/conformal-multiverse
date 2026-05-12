from multiversum import Config

# Configure the multiverse itself
config = Config(
    # Specify the decisions / dimensions and their options
    dimensions={
        "training_size": ["25k", "5k", "1k"],
        "training_year": ["2014", "2012_14", "2010_14"],
        "scale": ["scale", "do-not-scale"],
        # "encode_categorical": ["ordinal", "one-hot"],
        "model": ["logreg", "penalized_logreg", "rf", "gbm", "elasticnet"],
        "exclude_features": [
            "none",
            "nationality",
            "sex",
            "nationality-sex",
            "age",
        ],
        "exclude_subgroups": [
            "keep-all",
            "drop-non-german",
        ],

        # Evaluation
        "eval_fairness_grouping": ["majority-minority"],

        # Post-Deployment
        "cutoff": ["none"],
    },
)
