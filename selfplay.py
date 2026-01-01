import os
import random
import pickle
import numpy as np
import torch

from engine import BoardState
from mcts import MCTS
from move_encoder import create_policy_target


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class ReplayBuffer:
    """Simple replay buffer storing (position, policy, value) tuples."""
    def __init__(self, max_size=100000):
        self.max_size = int(max_size)
        self.positions = []  # list of torch.Tensor (13,8,8)
        self.policies = []   # list of numpy arrays (4672,)
        self.values = []     # list of floats (1-dimensional)

    def add_examples(self, examples):
        """Add list of (position_tensor, policy_array, value) examples.

        Position should be a torch.Tensor (13,8,8) or numpy array.
        Policy should be a numpy array shape (4672,).
        Value should be a float or scalar tensor.
        """
        for pos, policy, value in examples:
            # convert types
            if not isinstance(pos, torch.Tensor):
                pos = torch.as_tensor(pos, dtype=torch.float32)
            else:
                pos = pos.detach().cpu()

            policy = np.asarray(policy, dtype=np.float32)
            value = float(value)

            self.positions.append(pos)
            self.policies.append(policy)
            self.values.append([value])

        # Trim oldest
        excess = len(self.positions) - self.max_size
        if excess > 0:
            self.positions = self.positions[excess:]
            self.policies = self.policies[excess:]
            self.values = self.values[excess:]

    def sample(self, batch_size):
        """Sample a batch and return (positions, policies, values) as tensors.
        positions: FloatTensor (B,13,8,8)
        policies: FloatTensor (B,4672)
        values: FloatTensor (B,1)
        """
        size = len(self.positions)
        if size == 0:
            raise ValueError("ReplayBuffer is empty")

        idx = np.random.randint(0, size, size=batch_size)

        pos_batch = torch.stack([self.positions[i] for i in idx]).float()
        pol_batch = torch.from_numpy(np.stack([self.policies[i] for i in idx])).float()
        val_batch = torch.from_numpy(np.array([self.values[i] for i in idx], dtype=np.float32)).float()

        return pos_batch, pol_batch, val_batch

    def __len__(self):
        return len(self.positions)

    def save(self, path):
        data = {
            'positions': [p.numpy() for p in self.positions],
            'policies': self.policies,
            'values': self.values,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.positions = [torch.as_tensor(p, dtype=torch.float32) for p in data['positions']]
        self.policies = [np.asarray(p, dtype=np.float32) for p in data['policies']]
        self.values = data['values']


class SelfPlayWorker:
    """Generates self-play games using the Python MCTS implementation.

    If a C++ backend is available, this class could be extended to use it.
    """
    def __init__(self, model, num_simulations=100, temperature_threshold=30, max_game_length=512):
        self.model = model
        self.num_simulations = int(num_simulations)
        self.temperature_threshold = int(temperature_threshold)
        self.max_game_length = int(max_game_length)

    def generate_games(self, num_games, verbose=False):
        """Generate `num_games` self-play games.

        Returns a list of examples [(pos_tensor, policy_array, value), ...]
        """
        games = []

        for g in range(int(num_games)):
            board = BoardState()
            board.parse_fen(START_FEN)

            mcts = MCTS(self.model, num_simulations=self.num_simulations)

            history = []  # list of (pos_tensor, policy_array, player_side)

            move_count = 0
            while True:
                # Create position tensor
                pos_tensor = board.tensorize_board()

                # Adjust temperature (deterministic near end)
                if move_count >= self.temperature_threshold:
                    mcts.temperature = 0
                else:
                    mcts.temperature = 1.0

                best_move, move_probs = mcts.search(board)
                if best_move is None:
                    break

                # Create policy target (numpy array length 4672)
                policy_target = create_policy_target(move_probs)

                history.append((pos_tensor, policy_target, board.side))

                # Play move
                board.make_move(best_move)
                move_count += 1

                # Terminal / max length checks
                legal = board.generate_legal_moves(board.side)
                if len(legal) == 0:
                    break
                if move_count >= self.max_game_length:
                    break

            # Determine game outcome
            result_value = 0.0
            # If side to move has no legal moves, it's terminal
            legal_after = board.generate_legal_moves(board.side)
            if len(legal_after) == 0:
                if board.is_in_check(board.side):
                    # side to move lost
                    winner = 1 - board.side
                    result_value = 1.0
                else:
                    # stalemate
                    result_value = 0.0
            else:
                # Reached max length -> consider draw
                result_value = 0.0

            # Convert history to examples with values from perspective of player
            examples = []
            for pos, policy, player in history:
                if result_value == 0.0:
                    v = 0.0
                else:
                    # If winner equals player then +1 else -1
                    v = 1.0 if (winner == player) else -1.0
                examples.append((pos, policy, v))

            games.extend(examples)

            if verbose:
                print(f"Generated game {g+1}/{num_games}, moves: {move_count}, examples: {len(examples)}")

        return games


if __name__ == '__main__':
    # quick smoke test
    from model import EchelonNet

    model = EchelonNet()
    worker = SelfPlayWorker(model, num_simulations=10)
    ex = worker.generate_games(1, verbose=True)
    print(f"Examples generated: {len(ex)}")
