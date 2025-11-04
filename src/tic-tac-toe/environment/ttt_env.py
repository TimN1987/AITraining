import numpy as np

class TicTacToe:
    pass

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
        if len(np.argwhere(self.grid, player)) < 3:
            return False
        if self.check_win_horizontal(player):
            return True
        if self.check_win_vertical(player):
            return True
        if self.check_win_diagonal(player):
            return True
        return False

    def check_win_horizontal(self, player: int) -> bool:
        """
            Checks if any row of the playing grid is a winning row (i.e. all the same) for the given player.

            Args:
                player (int): restricted to 0 or 1 for the two players.

            Returns:
                True if a winning row is found for the player. False if no winning row is found.
        """
        if player not in [0, 1]:
            raise ValueError("Player number must be 0 or 1.")
        for i in range(3):
            if np.all(self.grid[i] == player):
                return True
        return False
    
    def check_win_vertical(self, player: int) -> bool:
        """
            Checks if any column of the playing grid is a winning column (i.e. all the same) for the given player.

            Args:
                player (int): restricted to 0 or 1 for the two players.

            Returns:
                True if a winning column is found for the player. False if no winning column is found.
        """
        for i in range(3):
            if [self.grid[0][i], self.grid[1][i], self.grid[2][i]] == [player] * 3:
                return True
        return False


    def check_win_diagonal(self, player: int) -> bool:
        """
            Checks if either diagonal of the playing grid is a winning line (i.e. all the same) for the given player.

            Args:
                player (int): restricted to 0 or 1 for the two players.

            Returns:
                True if a winning diagonal is found for the player. False if no winning diagonal is found.
        """
        if player not in [0, 1]:
            raise ValueError("Player number must be 0 or 1.")
        if [self.grid[0][0], self.grid[1][1], self.grid[2][2]] == [player] * 3:
            return True
        if [self.grid[0][2], self.grid[1][1], self.grid[2][0]] == [player] * 3:
            return True
        return False
