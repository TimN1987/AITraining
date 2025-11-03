import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from training.network import FootballPredictor

# Helper to convert team names to integers
def encode_teams(df):
    teams = list(set(df['team_a'].tolist() + df['team_b'].tolist()))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    df['team_a_idx'] = df['team_a'].map(team_to_idx)
    df['team_b_idx'] = df['team_b'].map(team_to_idx)
    return df, team_to_idx

# Console UI
def main():
    print("Football Score Predictor")
    print("Options:\n1. Train / Retrain the model\n2. Predict a match\n3. Quit")
    
    while True:
        choice = input("Select an option (1-3): ").strip()
        
        if choice == "1":
            csv_path = input("Enter path to training CSV file: ").strip()
            df = pd.read_csv(csv_path)
            df, team_to_idx = encode_teams(df)
            
            X = torch.tensor(df[['team_a_idx', 'team_b_idx']].values, dtype=torch.long)
            y = torch.tensor(df[['team_a_score', 'team_b_score']].values, dtype=torch.float32)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            predictor = FootballPredictor(num_teams=len(team_to_idx))
            predictor.load()  # load existing model if available
            
            predictor.train(X_train, y_train, epochs=100)
            
            # Evaluate on test set
            predictor.model.eval()
            with torch.no_grad():
                y_pred = predictor.model((X_test[:,0], X_test[:,1]))
                mse = torch.mean((y_pred - y_test) ** 2).item()
                print(f"Test MSE: {mse:.4f}")
            
            predictor.save()
        
        elif choice == "2":
            predictor = FootballPredictor(num_teams=0)  # will load model
            if not predictor.load():
                print("No trained model found. Please train first.")
                continue
            
            team_a = input("Enter team A name: ").strip()
            team_b = input("Enter team B name: ").strip()
            
            # Map team names to indices (from loaded model)
            if team_a not in predictor.team_to_idx or team_b not in predictor.team_to_idx:
                print("Unknown team names. Please train with these teams first.")
                continue
            
            idx_a = torch.tensor([predictor.team_to_idx[team_a]])
            idx_b = torch.tensor([predictor.team_to_idx[team_b]])
            
            predictor.model.eval()
            with torch.no_grad():
                scores = predictor.model((idx_a, idx_b))
                score_a, score_b = scores[0].round().int().tolist()
                
                if score_a > score_b:
                    result = 1
                elif score_a < score_b:
                    result = -1
                else:
                    result = 0
                
                print(f"Predicted Score: {team_a} {score_a} - {score_b} {team_b} | Result: {result}")
        
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("Invalid option. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
