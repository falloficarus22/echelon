import numpy as np
import torch
from collections import defaultdict
from engine import BoardState, MoveHistory
from move_encoder import encode_move, decode_index


class MCTSNode:
    """
    Represents a node in the MCTS tree.
    Each node corresponds to a board state.
    """
    def __init__(self, board_state, parent=None, parent_action=None, prior_prob=0.0):
        self.board_state = board_state
        self.parent = parent
        self.parent_action = parent_action  # The move that led to this node
        self.prior_prob = prior_prob  # P(s,a) from neural network
        
        self.children = {}  # Dict: move -> MCTSNode
        self.visit_count = 0  # N(s,a)
        self.value_sum = 0.0  # W(s,a)
        self.is_expanded = False
        
    def get_value(self):
        """Returns the mean value Q(s,a) = W(s,a) / N(s,a)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_leaf(self):
        """Check if this node is a leaf (not expanded)"""
        return not self.is_expanded
    
    def is_root(self):
        """Check if this is the root node"""
        return self.parent is None


class MCTS:
    """
    Monte Carlo Tree Search implementation for chess.
    Uses neural network for policy and value evaluation.
    """
    def __init__(self, model, num_simulations=800, c_puct=1.5, temperature=1.0, 
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        """
        Args:
            model: Neural network (EchelonNet) for evaluation
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant (UCB formula)
            temperature: Temperature for move selection (1.0 = stochastic, 0 = deterministic)
            dirichlet_alpha: Alpha parameter for Dirichlet noise at root
            dirichlet_epsilon: Weight of Dirichlet noise at root
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
    def search(self, board_state):
        """
        Perform MCTS search from the given board state.
        Returns: (best_move, move_probabilities)
        """
        # Create root node
        root = MCTSNode(board_state)
        
        # Add Dirichlet noise to root for exploration
        self._add_exploration_noise(root)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 1. Selection: Traverse tree using UCB until leaf node
            while not node.is_leaf() and len(node.children) > 0:
                node = self._select_child(node)
                search_path.append(node)
            
            # 2. Expansion and Evaluation
            value = self._expand_and_evaluate(node)
            
            # 3. Backpropagation: Update all nodes in search path
            self._backpropagate(search_path, value)
        
        # Get move probabilities from visit counts
        move_probs = self._get_move_probabilities(root)
        
        # Select best move based on visit counts
        if not move_probs:
            # This can happen if the board is terminal or no simulations were run
            return None, {}
            
        best_move = max(move_probs, key=move_probs.get)
        
        return best_move, move_probs
    
    def _select_child(self, node):
        """
        Select child with highest UCB score.
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        best_score = -float('inf')
        best_child = None
        
        for move, child in node.children.items():
            # Q value (mean action value)
            q_value = child.get_value()
            
            # UCB score
            u_score = (self.c_puct * child.prior_prob * 
                      np.sqrt(node.visit_count) / (1 + child.visit_count))
            
            score = q_value + u_score
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand_and_evaluate(self, node):
        """
        Expand the node and evaluate using neural network.
        Returns the value from the perspective of the player to move.
        """
        board = node.board_state
        
        # Check for terminal state (game over)
        legal_moves = board.generate_legal_moves(board.side)
        
        if len(legal_moves) == 0:
            # Game over: checkmate or stalemate
            if board.is_in_check(board.side):
                # Checkmate - loss for current player
                return -1.0
            else:
                # Stalemate - draw
                return 0.0
        
        # Tensorize board and get neural network prediction
        board_tensor = board.tensorize_board()
        value, policy_logits = self.model.predict(board_tensor)
        
        # Convert policy logits to probabilities
        policy = torch.exp(policy_logits).numpy()
        
        # Mask illegal moves and renormalize
        policy = self._mask_and_normalize_policy(policy, legal_moves)
        
        # Expand node by creating children for all legal moves
        for move in legal_moves:
            if move not in node.children:
                # Get prior probability for this move
                move_idx = encode_move(move)
                prior = policy[move_idx]
                
                # Make move to get child state
                child_board = self._copy_board(board)
                child_board.make_move(move)
                
                # Create child node
                child = MCTSNode(child_board, parent=node, 
                               parent_action=move, prior_prob=prior)
                node.children[move] = child
        
        node.is_expanded = True
        
        # Return value from perspective of current player
        # Neural network returns value from perspective of side to move
        return value
    
    def _backpropagate(self, search_path, value):
        """
        Backpropagate value through the search path.
        Value is negated at each level (perspective flips).
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Negate for opponent
    
    def _mask_and_normalize_policy(self, policy, legal_moves):
        """
        Mask illegal moves and renormalize policy to sum to 1.
        """
        # Create mask for legal moves
        mask = np.zeros_like(policy)
        for move in legal_moves:
            move_idx = encode_move(move)
            mask[move_idx] = 1.0
        
        # Apply mask
        masked_policy = policy * mask
        
        # Renormalize
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            # If all legal moves have zero probability, use uniform
            masked_policy = mask / mask.sum()
        
        return masked_policy
    
    def _get_move_probabilities(self, root):
        """
        Get move probabilities based on visit counts.
        Uses temperature parameter for exploration vs exploitation.
        """
        move_visits = {}
        total_visits = 0
        
        for move, child in root.children.items():
            visits = child.visit_count
            move_visits[move] = visits
            total_visits += visits
        
        if total_visits == 0:
            # No visits, return uniform
            num_moves = len(root.children)
            if num_moves == 0:
                return {}
            return {move: 1.0/num_moves for move in root.children.keys()}
        
        # Apply temperature
        if self.temperature == 0:
            # Deterministic: pick move with most visits
            best_move = max(move_visits, key=move_visits.get)
            move_probs = {move: 0.0 for move in move_visits.keys()}
            move_probs[best_move] = 1.0
        else:
            # Stochastic: sample proportional to visit counts
            move_probs = {}
            for move, visits in move_visits.items():
                # Apply temperature
                prob = (visits ** (1.0 / self.temperature))
                move_probs[move] = prob
            
            # Normalize
            prob_sum = sum(move_probs.values())
            for move in move_probs:
                move_probs[move] /= prob_sum
        
        return move_probs
    
    def _add_exploration_noise(self, root):
        """
        Add Dirichlet noise to prior probabilities at root for exploration.
        This encourages the search to try different moves.
        """
        if not root.is_expanded:
            return
        
        # Generate Dirichlet noise
        num_children = len(root.children)
        if num_children == 0:
            return
        
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_children)
        
        # Apply noise to children's prior probabilities
        for i, (move, child) in enumerate(root.children.items()):
            child.prior_prob = (
                (1 - self.dirichlet_epsilon) * child.prior_prob + 
                self.dirichlet_epsilon * noise[i]
            )
    
    def _copy_board(self, board):
        """
        Create a deep copy of the board state.
        """
        new_board = BoardState()
        new_board.bitboards = board.bitboards.copy()
        new_board.occupancies = board.occupancies.copy()
        new_board.side = board.side
        new_board.en_passant_sq = board.en_passant_sq
        new_board.castle_rights = board.castle_rights
        new_board.halfmove_clock = board.halfmove_clock
        return new_board


def play_move_with_mcts(mcts, board_state):
    """
    Helper function to play a single move using MCTS.
    Returns: (best_move, move_probabilities, root_value)
    """
    best_move, move_probs = mcts.search(board_state)
    
    # Get value of root position
    board_tensor = board_state.tensorize_board()
    root_value, _ = mcts.model.predict(board_tensor)
    
    return best_move, move_probs, root_value


def test_mcts():
    """
    Test MCTS implementation with a simple position.
    """
    from model import EchelonNet
    
    print("Testing MCTS Implementation...")
    print("=" * 50)
    
    # Create model
    model = EchelonNet(in_channels=13, num_res_blocks=5, num_filters=128)
    model.eval()
    
    # Create MCTS
    mcts = MCTS(model, num_simulations=100, temperature=1.0)
    
    # Create board
    board = BoardState()
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board.parse_fen(start_fen)
    
    print("\nStarting position:")
    print("Running 100 MCTS simulations...")
    
    # Search
    best_move, move_probs = mcts.search(board)
    
    print(f"\nBest move found: {best_move}")
    print(f"Total moves considered: {len(move_probs)}")
    
    # Print top 5 moves by probability
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 moves by visit probability:")
    for i, (move, prob) in enumerate(sorted_moves[:5]):
        decoded = board.decode_move(move)
        from_sq = decoded['from']
        to_sq = decoded['to']
        move_str = f"{chr(ord('a') + from_sq % 8)}{from_sq // 8 + 1}"
        move_str += f"{chr(ord('a') + to_sq % 8)}{to_sq // 8 + 1}"
        print(f"  {i+1}. {move_str}: {prob:.4f}")
    
    print("\nMCTS test completed successfully!")


if __name__ == "__main__":
    test_mcts()