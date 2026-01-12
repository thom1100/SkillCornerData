import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(df: pd.DataFrame, feature_cols: list):
    """
    Objective : Get the similarity matrix for our dataframe
    ------------------------
    Input:
    - df : original dataframe
    - feature_cols : list of the stats we aim at considering to filter df
    ------------------------
    Output :
    - sim_df : similarity matrix
    """
    clean_features = df[feature_cols].dropna(axis=1)

    sim_matrix = cosine_similarity(clean_features)
    sim_df = pd.DataFrame(sim_matrix, index=df["player"], columns=df["player"])

    return sim_df


def feature_contributions(df, player1, player2, feature_cols):
    """
    Objective : What features have the most impact
    ------------------------
    Input:
    - df : original dataframe, filtered by features cols
    - player1 : reference of the player
    - player2 : most similar player, used for contributions
    - feature_cols : list of the stats we aim at considering to filter df
    ------------------------
    Output :
    - contrib_df : dataframe with feature contributions, sorted by importance
    """

    # Extraire les vecteurs
    v1 = df.loc[df["player"] == player1, feature_cols].values.flatten()
    v2 = df.loc[df["player"] == player2, feature_cols].values.flatten()

    # Calculer la contribution brute
    contributions = v1 * v2

    # Normalisation (comme dans la cosine similarity)
    norm_factor = np.linalg.norm(v1) * np.linalg.norm(v2)
    contributions_normalized = contributions / norm_factor

    # Mettre dans un DataFrame
    contrib_df = pd.DataFrame(
        {"feature": feature_cols, "contribution": contributions_normalized}
    ).sort_values(by="contribution", ascending=False)

    return contrib_df


def get_similar_players(df, similarity_df, player_name, top_n=5, filter_cols=None):
    """
    Objective : Listing all similar players to the one considered
    ------------------------
    Input:
    - df : original dataframe
    - similarity_df : similarity matrix computed
    - player_name : name of the player considered
    - top_n : number of similar player we want to display
    - filter_cols : List of filters we consider
    ------------------------
    Output :
    - result : Final Matrix, with the players ordered per similarity
    - sims : Whole dataframe filtered, with similarity scores added
    """
    if player_name not in similarity_df.index:
        return pd.DataFrame()  # empty

    sims = similarity_df.loc[player_name].drop(player_name, errors="ignore")
    filtered_ids = set(df["player"])
    for filter_col in filter_cols:

        if filter_col is None or filter_col not in df.columns:
            continue

        else:
            if filter_col == "Valeur marchande (euros)":
                continue
            else:
                target_value = df.loc[df["player"] == player_name, filter_col].values[0]
                current_filtered = set(df.loc[df[filter_col] == target_value, "player"])

        # Intersection progressive
        filtered_ids &= current_filtered

        sims = sims.loc[sims.index.intersection(filtered_ids)]

    top_similar = sims.sort_values(ascending=False).head(top_n)

    player_info = df[
        [
            "player",
            "pos",
            "nation",
            "Valeur marchande (euros)",
            "team",
            "age",
            "Performance G+A",
            "Expected xG",
            "Playing Time MP",
        ]
    ].set_index("player")
    result = player_info.loc[top_similar.index].assign(similarity=top_similar.values)
    return result.reset_index(drop=False), sims
