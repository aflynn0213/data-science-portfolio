import pandas as pd 
from sklearn.metrics import r2_score, make_scorer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def normalize_names(names):
    normalized_names = []
    for name in names:
        # Replace special characters
        name = name.replace("Ã¡", "a").replace("Ã©", "e").replace("Ã­", "i").replace("Ã³", "o").replace("Ãº", "u").replace("Ã±", "n").replace("â€™", "'").replace("â€˜", "'")
        # Capitalize each word
        name = ' '.join(word.capitalize() for word in name.split())
        normalized_names.append(name)
    return normalized_names

if __name__ == "__main__":
    # Read the data from the specified sheet
    df = pd.read_csv('player_data.csv')  # Assuming 'LR' is the relevant sheet
    
    # Normalize column names
    df.columns = df.columns.str.lower()
    
    # Normalize player names (if necessary)
    df['name'] = normalize_names(df['name'])
    
    # Filter the DataFrame by seasons
    df_2021 = df[df['season'] == 2021]
    df_2022 = df[df['season'] == 2022]
    df_2023 = df[df['season'] == 2023]
    df_2024 = df[df['season'] == 2024]

    # Set the index for each DataFrame
    df_2021.set_index(['season', 'name'], inplace=True)
    df_2022.set_index(['season', 'name'], inplace=True)
    df_2023.set_index(['season', 'name'], inplace=True)
    df_2024.set_index(['season', 'name'], inplace=True)

    # Identify common players
    players_2021 = df_2021.index.get_level_values('name').unique()
    players_2022 = df_2022.index.get_level_values('name').unique()
    players_2023 = df_2023.index.get_level_values('name').unique()
    players_2024 = df_2024.index.get_level_values('name').unique()

    common_players_2021_2022 = set(players_2021).intersection(players_2022)
    common_players_2022_2023 = set(players_2022).intersection(players_2023)
    common_players_2023_2024 = set(players_2023).intersection(players_2024)

    # Prepare the training data
    # Features from 2021 to predict 2022 wRC+
    x_2021 = df_2021.loc[(2021, list(common_players_2021_2022)), ["o-swing% (sc)", "swing% (sc)", "o-contact% (sc)", "z-contact% (sc)", "ev", "la"]]
    y_2022 = df_2022.loc[(2022, list(common_players_2021_2022)), "wrc+"]
    
    # Features from 2022 to predict 2023 wRC+
    x_2022 = df_2022.loc[(2022, list(common_players_2022_2023)), ["o-swing% (sc)", "swing% (sc)", "o-contact% (sc)", "z-contact% (sc)", "ev", "la"]]
    y_2023 = df_2023.loc[(2023, list(common_players_2022_2023)), "wrc+"]

    x_2023 = df_2023.loc[(2023, list(common_players_2023_2024)), ["o-swing% (sc)", "swing% (sc)", "o-contact% (sc)", "z-contact% (sc)", "ev", "la"]]
    y_2024 = df_2024.loc[(2024, list(common_players_2023_2024)), "wrc+"]

    # Sort the indices
    x_2021 = x_2021.sort_index(level='name')
    y_2022 = y_2022.sort_index(level='name')
    x_2022 = x_2022.sort_index(level='name')
    y_2023 = y_2023.sort_index(level='name')
    x_2023 = x_2023.sort_index(level='name')  # New line for sorting 2023
    y_2024 = y_2024.sort_index(level='name') 

    # Combine training features and targets
    x_train = pd.concat([x_2021, x_2022, x_2023])  # Combine features for 2021, 2022, and 2023
    y_train = pd.concat([y_2022, y_2023, y_2024])

    # Display the shapes of the training sets
    print(f"Training features shape: {x_train.shape}")
    print(f"Training target shape: {y_train.shape}")

    # Set up the pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(include_bias=True)),
        ('regressor', Lasso())
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'regressor': [Lasso(), Ridge()],
        'poly__degree': [1, 2, 3, 4],
        'poly__interaction_only': [False],
        'poly__include_bias': [True, False],
        'regressor__alpha': [0.001, 0.01, 0.1, 0.25, 1.0, 2.5, 5, 10.0, 20]
    }

    # Scorer
    scorer = make_scorer(r2_score)

    # Perform GridSearchCV
    model = GridSearchCV(pipeline, param_grid, scoring=scorer, n_jobs=-1, cv=4)
    model.fit(x_train, y_train)

    # Extract the best estimator and parameters
    best_estimator = model.best_estimator_
    best_regressor = best_estimator.named_steps['regressor']
    print(f"Best regressor: {best_regressor.__class__.__name__}")
    print(f"Best training parameters: {best_regressor.get_params()}")
    print(f"Best poly training parameters: {best_estimator.named_steps['poly'].get_params()}")
    print(f"Best score: {model.best_score_}")

    # Here, you can proceed to test on future data or other analysis as needed

    y_pred = best_estimator.predict(x_train)

    # Create a DataFrame with player names and predicted wRC+
    predictions_df = pd.DataFrame({
        'Predicted wRC+': y_pred
    }, index=x_train.index)

    # Display the predictions DataFrame
    print(predictions_df)

    # Original data
    data = {
        'Season': ['2024', '2024', '2024'],
        'Name': ['Player A', 'Player B', 'Player C'],
        'O-Swing%': [0.30, 0.25, 0.33],  # Decimals instead of percentage strings
        'Swing%': [0.40, 0.45, 0.50],    # Decimals instead of percentage strings
        'O-Contact%': [0.90, 0.60, 0.55], # Decimals instead of percentage strings
        'Z-Contact%': [0.95, 0.85, 0.75], # Decimals instead of percentage strings
        'EV': [88.5, 89.0, 93.0],
        'LA': [12.5, 13.0, 12.0]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)  # Remove the index argument
    df.columns = df.columns.str.lower()

    # Make predictions for the example players
    wrc_predictions = best_estimator.predict(df[["o-swing%", "swing%", "o-contact%", "z-contact%", "ev", "la"]])

    # Display the predicted wRC+
    prediction_results = pd.DataFrame({'Predicted wRC+': wrc_predictions}, index=df.name)
    print(prediction_results)