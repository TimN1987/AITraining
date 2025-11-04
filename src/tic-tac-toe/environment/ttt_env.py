import numpy as np

class TicTacToe:
    def __init__(self, ai_player: TicTacToePlayer) -> None:
        self.board = Board()

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
            raise Exception("Invalid move attempted - the positions has already been taken.")
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
            self.check_win_horizontal(player):
            or self.check_win_vertical(player):
            or self.check_win_diagonal(player):
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

    def display(self):
        print("\n".join(" ".join(str(x) for x in row) for row in self.grid))

    def reset(self):
        self.grid.fill(0)