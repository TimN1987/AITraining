import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from training.network import FootballPredictor

# Load and merge all training CSVs
def load_training_csvs():
    data_dir = Path(__file__).parent / "training_data"
    csv_files = sorted(data_dir.glob("*data.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    print(f"Loading {len(csv_files)} CSV files...")
    all_dfs = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(f"  - Loaded {csv_file.name} ({len(df)} rows)")

        # Stop when upcoming fixtures start (no Result values)
        if df['Result'].isna().any():
            first_empty_idx = df['Result'].isna().idxmax()
            df = df.iloc[:first_empty_idx]

        # Split result into score columns
        scores = df['Result'].str.split(" - ", expand=True)
        df['team_a_score'] = scores[0].astype(int)
        df['team_b_score'] = scores[1].astype(int)

        # Keep only relevant columns
        df = df[['Home Team', 'Away Team', 'team_a_score', 'team_b_score']]
        all_dfs.append(df)

    # Combine all seasons into one DataFrame
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total matches loaded: {len(full_df)}")

    # Create unified team index across all files
    teams = list(set(full_df['Home Team'].tolist() + full_df['Away Team'].tolist()))
    team_to_idx = {team: idx for idx, team in enumerate(sorted(teams))}

    full_df['team_a_idx'] = full_df['Home Team'].map(team_to_idx)
    full_df['team_b_idx'] = full_df['Away Team'].map(team_to_idx)

    # Convert to tensors
    X = torch.tensor(full_df[['team_a_idx', 'team_b_idx']].values, dtype=torch.long)
    y = torch.tensor(full_df[['team_a_score', 'team_b_score']].values, dtype=torch.float32)
    
    return X, y, team_to_idx

# Helper to compute accuracy on match outcomes
def compute_accuracy(y_true, y_pred):
    true_result = torch.sign(y_true[:, 0] - y_true[:, 1])
    pred_result = torch.sign(y_pred[:, 0] - y_pred[:, 1])
    correct = (true_result == pred_result).sum().item()
    total = y_true.size(0)
    return correct / total * 100.0

# Console UI
def main():
    print("Football Score Predictor")
    print("Options:\n1. Train / Retrain the model\n2. Predict a match\n3. Quit")

    predictor = None
    team_to_idx = {}

    while True:
        choice = input("Select an option (1-3): ").strip()
        
        if choice == "1":
            print("Loading training data...")
            X, y, team_to_idx = load_training_csvs()
            print("All data loaded successfully.")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            predictor = FootballPredictor(num_teams=len(team_to_idx))
            predictor.load()

            try:
                epochs = int(input("Enter number of training epochs (default 100): ") or "100")
            except ValueError:
                epochs = 100

            print(f"Training for {epochs} epochs...")
            predictor.train(X_train, y_train, epochs=epochs)

            # --- Evaluate ---
            predictor.model.eval()
            with torch.no_grad():
                y_pred = predictor.model(X_test)
                mse = torch.mean((y_pred - y_test) ** 2).item()
                acc = compute_accuracy(y_test, y_pred)
                print(f"\nEvaluation:")
                print(f"  Test MSE: {mse:.4f}")
                print(f"  Accuracy (correct result direction): {acc:.2f}%\n")

            predictor.save()

        elif choice == "2":
            if predictor is None:
                predictor = FootballPredictor(num_teams=0)
                if not predictor.load():
                    print("No trained model found. Please train first.")
                    continue

            team_a = input("Enter team A name: ").strip()
            team_b = input("Enter team B name: ").strip()

            if team_a not in team_to_idx or team_b not in team_to_idx:
                print("Unknown team names. Please train with these teams first.")
                continue

            idx_a = torch.tensor([team_to_idx[team_a]])
            idx_b = torch.tensor([team_to_idx[team_b]])

            predictor.model.eval()
            with torch.no_grad():
                scores = predictor.model(torch.stack([idx_a, idx_b], dim=1))
                score_a, score_b = scores[0].round().int().tolist()

                if score_a > score_b:
                    result = "Home Win"
                elif score_a < score_b:
                    result = "Away Win"
                else:
                    result = "Draw"

                print(f"Predicted Score: {team_a} {score_a} - {score_b} {team_b}")
                print(f"Predicted Result: {result}")

        elif choice == "3":
            print("Exiting.")
            break

        else:
            print("Invalid option. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
