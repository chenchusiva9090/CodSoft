# Define the board
board = [' ' for _ in range(9)]

# Function to print the board
def print_board(board):
    print(f"{board[0]} | {board[1]} | {board[2]}")
    print("--|---|--")
    print(f"{board[3]} | {board[4]} | {board[5]}")
    print("--|---|--")
    print(f"{board[6]} | {board[7]} | {board[8]}")

# Function to check if the board is full
def is_board_full(board):
    return ' ' not in board

# Function to check if a player has won
def check_winner(board, player):
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                            (0, 3, 6), (1, 4, 7), (2, 5, 8),
                            (0, 4, 8), (2, 4, 6)]

    for combination in winning_combinations:
        if board[combination[0]] == board[combination[1]] == board[combination[2]] == player:
            return True
    return False

# Function for the AI to make a move
def minimax(board, depth, is_maximizing):
    if check_winner(board, 'X'):
        return -1
    if check_winner(board, 'O'):
        return 1
    if is_board_full(board):
        return 0

    if is_maximizing:
        max_eval = float('-inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                eval = minimax(board, depth+1, False)
                board[i] = ' '
                max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                eval = minimax(board, depth+1, True)
                board[i] = ' '
                min_eval = min(min_eval, eval)
        return min_eval

def find_best_move(board):
    best_move = -1
    best_eval = float('-inf')
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            move_eval = minimax(board, 0, False)
            board[i] = ' '
            if move_eval > best_eval:
                best_eval = move_eval
                best_move = i
    return best_move

# Main game loop
while True:
    print_board(board)

    # Player's move
    player_move = int(input("Enter your move (1-9): ")) - 1
    if board[player_move] != ' ' or player_move < 0 or player_move > 8:
        print("Invalid move. Try again.")
        continue
    board[player_move] = 'X'

    # Check if player has won
    if check_winner(board, 'X'):
        print_board(board)
        print("Congratulations! You win!")
        break

    # Check if the board is full (draw)
    if is_board_full(board):
        print_board(board)
        print("It's a draw!")
        break

    # AI's move
    ai_move = find_best_move(board)
    board[ai_move] = 'O'

    # Check if AI has won
    if check_winner(board, 'O'):
        print_board(board)
        print("AI wins! Better luck next time.")
        break
