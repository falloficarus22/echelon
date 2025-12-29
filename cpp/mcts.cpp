#include "mcts.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>

namespace py = pybind11;

int MCTS::encode_move_for_nn(Move move) {
    int from_sq = move.from;
    int to_sq = move.to;
    MoveFlag flag = move.flag;
    
    int from_rank = from_sq / 8;
    int from_file = from_sq % 8;
    int to_rank = to_sq / 8;
    int to_file = to_sq % 8;
    
    int dr = to_rank - from_rank;
    int df = to_file - from_file;
    
    int action_id = -1;
    
    // 1. Underpromotions (Actions 64-72)
    if (flag == MOVE_FLAG_PROMOTION_KNIGHT || flag == MOVE_FLAG_PROMOTION_BISHOP || flag == MOVE_FLAG_PROMOTION_ROOK) {
        int p_idx = (flag == MOVE_FLAG_PROMOTION_KNIGHT) ? 0 : (flag == MOVE_FLAG_PROMOTION_BISHOP ? 1 : 2);
        action_id = 64 + (df + 1) * 3 + p_idx;
    }
    // 2. Knight moves (Actions 56-63)
    else {
        static const int knight_moves[8][2] = {{2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}};
        bool is_knight = false;
        for (int i = 0; i < 8; i++) {
            if (dr == knight_moves[i][0] && df == knight_moves[i][1]) {
                action_id = 56 + i;
                is_knight = true;
                break;
            }
        }
        
        // 3. Queen-like moves (Actions 0-55)
        if (!is_knight) {
            if (dr == 0 || df == 0 || std::abs(dr) == std::abs(df)) {
                int step_r = (dr > 0) ? 1 : (dr < 0 ? -1 : 0);
                int step_f = (df > 0) ? 1 : (df < 0 ? -1 : 0);
                int dist = std::max(std::abs(dr), std::abs(df));
                
                static const int directions[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
                for (int i = 0; i < 8; i++) {
                    if (step_r == directions[i][0] && step_f == directions[i][1]) {
                        action_id = i * 7 + (dist - 1);
                        break;
                    }
                }
            }
        }
    }
    
    if (action_id == -1) return -1;
    return from_sq * 73 + action_id;
}

std::map<int, float> MCTS::search(Board& root_board, py::object& model) {
    auto root = std::make_unique<MCTSNode>(root_board);
    
    // Initial expansion of root
    expand_and_evaluate(root.get(), model);
    add_exploration_noise(root.get());
    
    for (int i = 0; i < num_simulations; i++) {
        MCTSNode* node = root.get();
        std::vector<MCTSNode*> path = {node};
        
        // 1. Selection
        while (!node->is_leaf() && !node->children.empty()) {
            node = select_child(node);
            path.push_back(node);
        }
        
        // 2. Expansion and Evaluation
        float value = expand_and_evaluate(node, model);
        
        // 3. Backpropagation
        backpropagate(path, value);
    }
    
    // Return move probabilities (visit counts normalized)
    std::map<int, float> move_probs;
    for (auto const& [move_idx, child] : root->children) {
        move_probs[move_idx] = (float)child->visit_count / root->visit_count;
    }
    
    return move_probs;
}

MCTSNode* MCTS::select_child(MCTSNode* node) {
    float best_score = -1e9;
    MCTSNode* best_child = nullptr;
    
    float sqrt_total = std::sqrt((float)node->visit_count);
    
    for (auto const& [move_idx, child] : node->children) {
        float q_value = child->get_value();
        float u_score = c_puct * child->prior_prob * sqrt_total / (1 + child->visit_count);
        float score = q_value + u_score;
        
        if (score > best_score) {
            best_score = score;
            best_child = child.get();
        }
    }
    return best_child;
}

float MCTS::expand_and_evaluate(MCTSNode* node, py::object& model) {
    // Check legal moves
    MoveList list;
    node->board.generate_moves(list);
    
    std::vector<Move> legal_moves;
    for (int i = 0; i < list.count; i++) {
        Board temp = node->board;
        temp.make_move(list.moves[i]);
        if (!temp.is_in_check(1 - temp.side)) { 
            legal_moves.push_back(list.moves[i]);
        }
    }
    
    if (legal_moves.empty()) {
        if (node->board.is_in_check()) return -1.0f; // Loss
        return 0.0f; // Draw
    }
    
    // call model.predict(tensor)
    std::vector<float> tensor = node->board.tensorize();
    py::array_t<float> py_tensor({13, 8, 8}, tensor.data());
    
    py::tuple result = model.attr("predict")(py_tensor);
    float value = result[0].cast<float>();
    py::array_t<float> policy_logits = result[1].cast<py::array_t<float>>();
    auto policy_ptr = (float*)policy_logits.data();
    
    // Convert logits to probs and mask
    std::vector<float> masked_policy(4672, 0.0f);
    float sum_p = 0.0f;
    for (auto m : legal_moves) {
        int idx = encode_move_for_nn(m);
        if (idx != -1) {
            float p = std::exp(policy_ptr[idx]);
            masked_policy[idx] = p;
            sum_p += p;
        }
    }
    
    if (sum_p > 0) {
        for (int i = 0; i < 4672; i++) masked_policy[i] /= sum_p;
    } else {
        float uniform = 1.0f / legal_moves.size();
        for (auto m : legal_moves) {
            int idx = encode_move_for_nn(m);
            if (idx != -1) masked_policy[idx] = uniform;
        }
    }
    
    // Create children
    for (auto m : legal_moves) {
        int idx = encode_move_for_nn(m);
        if (idx != -1) {
            Board child_board = node->board;
            child_board.make_move(m);
            node->children[idx] = std::make_unique<MCTSNode>(child_board, node, m, masked_policy[idx]);
        }
    }
    
    node->is_expanded = true;
    return value;
}

void MCTS::backpropagate(std::vector<MCTSNode*>& path, float value) {
    for (int i = path.size() - 1; i >= 0; i--) {
        path[i]->visit_count++;
        path[i]->value_sum += value;
        value = -value;
    }
}

void MCTS::add_exploration_noise(MCTSNode* root) {
    if (root->children.empty()) return;
    
    int n = root->children.size();
    std::gamma_distribution<float> dist(dirichlet_alpha, 1.0f);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::vector<float> noise(n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        noise[i] = dist(gen);
        sum += noise[i];
    }
    
    int i = 0;
    for (auto const& [idx, child] : root->children) {
        child->prior_prob = (1 - dirichlet_epsilon) * child->prior_prob + dirichlet_epsilon * (noise[i++] / sum);
    }
}
