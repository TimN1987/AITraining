import numpy as np
from training.rl_player import TicTacToePlayer

class TicTacToeEnv:
    def __init__(self, ai_player: TicTacToePlayer = None, ai_opponent: TicTacToePlayer = None) -> None:
        self.board = Board()
        self.player = ai_player
        self.opponent = ai_opponent

    def reset(self) -> None:
        self.board.reset()

    def run_game_two_player(self):
        game_over = False
        is_draw = False
        episode_history = []
        current_player = 1
        turn_count = 0
        next_state = self.player.get_state(self.board.grid)
        while not game_over:
            if self.board.check_draw():
                is_draw = True
                break
            state = next_state
            print(f"Player {current_player} to play.")
            turn_count += 1
            invalid_move = True
            while invalid_move:
                try:
                    self.board.display()
                    print("Choose your next move.")
                    print(self.board.get_available_moves())
                    row = int(input("Enter the selected row: "))
                    col = int(input("Enter the selected column: "))
                    game_over = self.board.make_move(current_player, row, col)
                    next_state = self.player.get_state(self.board.grid)
                    episode_history.append({
                        "player": current_player,
                        "state": state,
                        "row": row,
                        "column": col,
                        "turn_count": turn_count,
                        "next_state": next_state,
                        "game_over": game_over
                    })
                    if game_over:
                        episode_history.append({"winner": current_player})
                    current_player = 1 if current_player == 2 else 2
                    invalid_move = False
                except:
                    print("Enter a valid row and column from the available positions.")
        if game_over:
            print(f"The winner is player {1 if current_player == 2 else 2}.")
        if is_draw:
            print("No winner - it's a draw!")
            episode_history.append({"winner": 0})
        return episode_history

    def run_game_one_player(self, player_starts: bool):
        game_over = False
        is_draw = False
        episode_history = []
        current_player = 1
        turn_count = 0
        state = self.player.get_state(self.board.grid)
        next_state = state

        # Opening player move if required.
        if player_starts:
            print(f"Your turn.")
            turn_count += 1
            invalid_move = True
            while invalid_move:
                try:
                    print("Choose your move.")
                    print(self.board.get_available_moves())
                    self.board.display()
                    row = int(input("Enter the selected row: "))
                    col = int(input("Enter the selected column: "))
                    game_over = self.board.make_move(current_player, row, col)
                    next_state = self.player.get_state(self.board.grid)
                    episode_history.append({
                        "player": current_player,
                        "state": state,
                        "row": row,
                        "column": col,
                        "turn_count": turn_count,
                        "next_state": next_state,
                        "game_over": game_over
                    })
                    current_player = 2
                    invalid_move = False
                except:
                    print("Enter a valid row and column from the available positions.")
        
        # Run remaining turns.
        while not game_over:
            print("AI player taking turn...")
            if self.board.check_draw():
                is_draw = True
                break
            turn_count += 1
            state = next_state
            row, col = self.player.choose_action(state, current_player)
            game_over = self.board.make_move(current_player, row, col)
            next_state = self.player.get_state(self.board.grid)
            episode_history.append({
                        "player": current_player,
                        "state": state,
                        "row": row,
                        "column": col,
                        "turn_count": turn_count,
                        "next_state": next_state,
                        "game_over": game_over
                    })
            if game_over:
                    episode_history.append({"winner": current_player})
            current_player = 1 if current_player == 2 else 2
            print(f"AI player played {row},{col}.")
            self.board.display()
            if game_over:
                break

            print(f"Player {current_player} to play.")
            if self.board.check_draw():
                is_draw = True
                break
            turn_count += 1
            invalid_move = True
            state = next_state
            while invalid_move:
                try:
                    print("Choose your next move.")
                    print(self.board.get_available_moves())
                    row = int(input("Enter the selected row: "))
                    col = int(input("Enter the selected column: "))
                    game_over = self.board.make_move(current_player, row, col)
                    next_state = self.player.get_state(self.board.grid)
                    episode_history.append({
                        "player": current_player,
                        "state": state,
                        "row": row,
                        "column": col,
                        "turn_count": turn_count,
                        "next_state": next_state,
                        "game_over": game_over
                    })
                    if game_over:
                        episode_history.append({"winner": current_player})
                    current_player = 1 if current_player == 2 else 2
                    invalid_move = False
                except:
                    print("Enter a valid row and column from the available positions.")
        if game_over:
            print(f"The winner is player {1 if current_player == 2 else 2}.")
        if is_draw:
            print("No winners - it's a draw!")
            episode_history.append({"winner": 0})
        return episode_history

    def simulate_game(self):
        game_over = False
        episode_history = []
        current_player = 1
        turn_count = 0
        next_state = self.player.get_state(self.board.grid)
        while not game_over:
            if self.board.check_draw():
                break
            turn_count += 1
            state = next_state
            row, col = self.player.choose_action(state, current_player) if current_player == 1 else self.opponent.choose_action(state, current_player)
            game_over = self.board.make_move(current_player, row, col)
            next_state = self.player.get_state(self.board.grid)
            episode_history.append({
                        "player": current_player,
                        "state": state,
                        "row": row,
                        "column": col,
                        "turn_count": turn_count,
                        "next_state": next_state,
                        "game_over": game_over
                    })
            if game_over:
                episode_history.append({"winner": current_player})
            current_player = 1 if current_player == 2 else 2
        if not game_over:
            episode_history.append({"winner": 0})
        return episode_history


class Board:
    def __init__(self) -> None:
        self.grid = np.zeros((3, 3), dtype=int)

    # Making moves

    def make_move(self, player: int, row: int, col: int) -> bool:
        """
            Runs a turn for the given player. Updates the grid if the move is valid and returns a boolean value to indicate if the player has won.

            Args:
                player (int): restricted to 1 or 2 for the two players.
                row (int): an integer value between 0 and 2.
                col (int): an integer value between 0 and 2.

            Returns:
                True if the move resulted in a win for the player. False if the game is not yet over.
        """
        if row not in range(3) or col not in range(3):
            raise Exception("Invalid move attempted - row and col must be between 0 and 2.")
        if not self.grid[row][col] == 0:
            raise Exception("Invalid move attempted - the position has already been taken.")
        self.grid[row][col] = player
        return self.check_win(player)

    # End game checks

    def check_draw(self) -> bool:
        """
            Checks if all grid cells are non-zero and so the match is over.
        """
        return np.all(self.grid != 0)

    def check_win(self, player: int) -> bool:
        """ 
            Checks if a player has won the game.

            Args:
                player (int): restricted to 1 or 2 for the two players.

            Returns:
                True if the player has won. False if they have not yet won.
        """
        if player not in [1, 2]:
            raise ValueError("Player number must be 1 or 2.")
        # Check enough moves have been played for a win.
        if len(np.argwhere(self.grid == player)) < 3:
            return False
        # Check each direction for a win.
        return (
            self.check_win_horizontal(player)
            or self.check_win_vertical(player)
            or self.check_win_diagonal(player)
        )

    def check_win_horizontal(self, player: int) -> bool:
        """
            Checks if any row of the playing grid is a winning row (i.e. all the same) for the given player.
        """
        return np.any(np.all(self.grid == player, axis=1))
    
    def check_win_vertical(self, player: int) -> bool:
        """
            Checks if any column of the playing grid is a winning column (i.e. all the same) for the given player.
        """
        return np.any(np.all(self.grid == player, axis=0))

    def check_win_diagonal(self, player: int) -> bool:
        """
            Checks if either diagonal of the playing grid is a winning line (i.e. all the same) for the given player.
        """
        return (
            np.all(np.diag(self.grid) == player)
            or np.all(np.diag(np.fliplr(self.grid)) == player)
        )

    # General methods

    def get_available_moves(self) -> list[int]:
        return np.argwhere(self.grid == 0).tolist()

    def display(self) -> None:
        print("\n".join(" ".join(str(x) for x in row) for row in self.grid))

    def reset(self) -> None:
        self.grid.fill(0)