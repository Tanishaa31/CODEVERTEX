import copy

# Define the game board
EMPTY = ' '
PLAYER = 'X'
AI = 'O'

def evaluate_board(board):
    """Evaluate the current state of the board"""
    # Check rows
    for i in range(0, 9, 3):
        if board[i] == board[i+1] == board[i+2] != EMPTY:
            return 1 if board[i] == AI else -1
    
    # Check columns
    for i in range(3):
        if board[i] == board[i+3] == board[i+6] != EMPTY:
            return 1 if board[i] == AI else -1
    
    # Check diagonals
    if board[0] == board[4] == board[8] != EMPTY:
        return 1 if board[0] == AI else -1
    if board[2] == board[4] == board[6] != EMPTY:
        return 1 if board[2] == AI else -1
    
    # No winner
    return 0

def game_over(board):
    """Check if the game is over"""
    return EMPTY not in board or evaluate_board(board) != 0

def minimax(board, depth, is_maximizing):
    """Implement the Minimax algorithm with Alpha-Beta pruning"""
    if game_over(board):
        return evaluate_board(board)
    
    if is_maximizing:
        best_score = float('-inf')
        for i, cell in enumerate(board):
            if cell == EMPTY:
                new_board = copy.copy(board)
                new_board[i] = AI
                score = minimax(new_board, depth + 1, False)
                best_score = max(best_score, score)
        return best_score
    else:
        best_score = float('inf')
        for i, cell in enumerate(board):
            if cell == EMPTY:
                new_board = copy.copy(board)
                new_board[i] = PLAYER
                score = minimax(new_board, depth + 1, True)
                best_score = min(best_score, score)
        return best_score

def get_best_move(board):
    """Find the best move for the AI player"""
    best_score = float('-inf')
    best_move = None
    for i, cell in enumerate(board):
        if cell == EMPTY:
            new_board = copy.copy(board)
            new_board[i] = AI
            score = minimax(new_board, 0, False)
            if score > best_score:
                best_score = score
                best_move = i
    return best_move

def play_game():
    """Play a game of Tic-Tac-Toe against the AI"""
    board = [EMPTY] * 9
    current_player = PLAYER

    while not game_over(board):
        print_board(board)
        if current_player == PLAYER:
            move = int(input("Your move (0-8): "))
            if board[move] != EMPTY:
                print("That spot is already taken!")
                continue
            board[move] = PLAYER
        else:
            print("AI's move:")
            move = get_best_move(board)
            board[move] = AI
        current_player = PLAYER if current_player == AI else AI

    print_board(board)
    result = evaluate_board(board)
    if result == 0:
        print("It's a tie!")
    elif result == 1:
        print("AI wins!")
    else:
        print("You win!")

def print_board(board):
    """Print the current state of the Tic-Tac-Toe board"""
    print("---------")
    for i in range(0, 9, 3):
        row = " ".join(board[i:i+3])
        print(f"| {row} |")
    print("---------")

if __name__ == "__main__":
    play_game()