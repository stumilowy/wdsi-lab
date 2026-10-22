import random
from typing import Optional
from IPython.display import clear_output


def create_board(size: int = 8) -> list[str]:
    if size < 2:
        raise ValueError("Rozmiar planszy musi wynosić przynajmniej 2.")
    return ['.'] * size


def get_valid_moves(board: list[str]) -> list[int]:
    return [i for i, cell in enumerate(board) if cell == '.']


def make_move(board: list[str], move: int, player_color: str) -> tuple[list[str], list[int]]:
    opponent = 'O' if player_color == 'X' else 'X'
    new_board = board[:]
    new_board[move] = player_color
    flipped: list[int] = []
    size = len(board)

    # Leftward
    candidates: list[int] = []
    j = move - 1
    while j >= 0 and board[j] == opponent:
        candidates.append(j)
        j -= 1
    if candidates and j >= 0 and board[j] == player_color:
        for idx in candidates:
            new_board[idx] = player_color
        flipped.extend(candidates)

    # Rightward
    candidates = []
    j = move + 1
    while j < size and board[j] == opponent:
        candidates.append(j)
        j += 1
    if candidates and j < size and board[j] == player_color:
        for idx in candidates:
            new_board[idx] = player_color
        flipped.extend(candidates)

    return new_board, sorted(flipped)


def is_game_over(board: list[str]) -> bool:
    return '.' not in board


def calculate_score(board: list[str], player_color: str) -> int:
    opponent = 'O' if player_color == 'X' else 'X'
    return board.count(player_color) - board.count(opponent)


def random_agent(board: list[str], player_color: str, valid_moves: list[int]) -> int:
    return random.choice(valid_moves)


def _indices_to_prose(indices: list[int]) -> str:
    if not indices:
        return "none"
    strs = [str(i) for i in indices]
    if len(strs) == 1:
        return strs[0]
    if len(strs) == 2:
        return f"{strs[0]} and {strs[1]}"
    return ", ".join(strs[:-1]) + f", and {strs[-1]}"


def print_board_status(board: list[str]) -> None:
    size = len(board)
    x_cells = [i for i, c in enumerate(board) if c == 'X']
    o_cells = [i for i, c in enumerate(board) if c == 'O']
    empty_cells = [i for i, c in enumerate(board) if c == '.']

    x_str = _indices_to_prose(x_cells) if x_cells else "none"
    o_str = _indices_to_prose(o_cells) if o_cells else "none"
    e_str = _indices_to_prose(empty_cells) if empty_cells else "none"

    print(f"Rozmiar planszy {size}. "
          f"X kontroluje {x_str}. "
          f"O kontroluje {o_str}. "
          f"Puste pola to {e_str}.")


def announce_move(player: str, move: int, flipped_indices: list[int]) -> None:
    flip_str = _indices_to_prose(flipped_indices)
    if flipped_indices:
        print(f"Gracz {player} umieścił pionek na polu {move}. "
              f"Pionki {flip_str} zmieniły kolor na {player}.")
    else:
        print(f"Gracz {player} umieścił pionek na polu {move}. Żaden pionek nie zmienił koloru.")


def get_human_input(valid_moves: list[int]) -> Optional[int]:
    options_str = _indices_to_prose(valid_moves)
    prompt = f"Dostępne ruchy: {options_str}. Wprowadź swój wybór (lub naciśnij Enter bez wpisywania niczego, aby zakończyć): "
    while True:
        try:
            raw = input(prompt)
        except (KeyboardInterrupt, EOFError):
            # Raised if the Jupyter input widget is cancelled/closed
            return None

        # Check for empty string (pressed Enter), 'escape', 'exit', 'quit', or 'q'
        if raw.strip().lower() in ['\x1b', 'escape', 'exit', 'quit', 'q', '']:
            return None
        
        try:
            choice = int(raw.strip())
        except ValueError:
            print(f"  Wpisz liczbę. Dostępne opcje to: {options_str}.")
            continue
        if choice not in valid_moves:
            print(f"  {choice} nie jest poprawnym ruchem. Dostępne opcje to: {options_str}.")
            continue
        return choice


def display_board(board: list[str], mode: str) -> None:
    if mode == "textual":
        print_board_status(board)
    elif mode == "visual":
        print(board)
    else:
        raise ValueError(f"Nieznany tryb: {mode}")


def play_game(agent_x, agent_o, mode: str = "visual") -> None:  # noqa: ANN001
    board = create_board()
    agents = {'X': agent_x, 'O': agent_o}
    current = 'X'
    opponent_map = {'X': 'O', 'O': 'X'}
    is_human_game = (agent_x == "human" or agent_o == "human")
    premature_exit = False

    print("=" * 50)
    print("Witaj w Othello 1D!")
    print("Zasady: umieszczaj swoje pionki na dowolnym pustym polu.")
    print("       Jeśli po umieszczeniu pionka, Twoje pionki otaczają pionki przeciwnika, otoczone pionki zmieniają kolor.")
    print("=" * 50)
    display_board(board, mode)
    print()

    while not is_game_over(board):
        valid_moves = get_valid_moves(board)
        agent = agents[current]

        if agent == "human":
            print(f"\nRuch gracza {current} (Ty).")
            display_board(board, mode)
            move = get_human_input(valid_moves)
            if move is None:
                premature_exit = True
                print("\nGra została przerwana przez gracza.")
                break
        else:
            move = agent(board, current, valid_moves)
            print(f"\nGracz {current} (komputer) myśli...")

        board, flipped = make_move(board, move, current)
        
        if is_human_game:
            # Dla ruchu człowieka czyścimy ekran zaraz po ruchu
            if agent == "human":
                clear_output(wait=True)
            announce_move(current, move, flipped)
        else:
            # AI vs AI: wyświetlamy wszystko
            announce_move(current, move, flipped)
            display_board(board, mode)

        current = opponent_map[current]

    print()
    print("=" * 50)
    if premature_exit:
        print("Gra została przerwana!")
    else:
        print("Gra zakończona!")
    display_board(board, mode)
    x_score = board.count('X')
    o_score = board.count('O')
    print(f"Ostateczny wynik — X: {x_score}, O: {o_score}.")
    if x_score > o_score:
        print("Gracz X wygrywa!")
    elif o_score > x_score:
        print("Gracz O wygrywa!")
    else:
        print("Remis!")
    print("=" * 50)
