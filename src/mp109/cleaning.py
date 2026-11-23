import pandas as pd


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean the training dataset using the rules in params.yaml.

    Args:
        df: Original training DataFrame.
        config: Cleaning configuration dictionary.

    Returns:
        A cleaned copy of the DataFrame.
    """

    data = df.copy()
    age_config = config["age"]

    age_column = age_config["column"]
    affected_zone = age_config["affected_zone"]
    conversion_factor = age_config["conversion_factor"]

    mask = data["city_zone"] == affected_zone

    data.loc[mask, age_column] = (
        data.loc[mask, age_column] / conversion_factor
    )
        # Replace sentinel values with NaN
    sentinel_config = config["sentinel"]

    sentinel_column = sentinel_config["column"]
    sentinel_value = sentinel_config["value"]

    data[sentinel_column] = data[sentinel_column].replace(
        sentinel_value, float("nan")
    )

        # Normalize brand category variants
    brand_config = config["brand"]
    brand_column = brand_config["column"]

    brand_mapping = {
        "budget": "budget",
        "bud.": "budget",
        "mid": "mid",
        "mid.": "mid",
        "premium": "premium",
        "pre.": "premium"
    }

    data[brand_column] = (
        data[brand_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(brand_mapping)
    )

        # Remove duplicate rows
    duplicate_config = config["duplicates"]

    data = data.drop_duplicates(
        keep=duplicate_config["keep"]
    )

    return data