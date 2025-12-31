#ifndef MCTS_HPP
#define MCTS_HPP

#include <vector>
#include <map>
#include <memory>
#include <cmath>
#include <algorithm>
#include "board.hpp"
#include "constants.hpp"

// Forward declaration for Python callback
namespace pybind11 { class object; }

struct MCTSNode {
    Board board;
    MCTSNode* parent;
    Move parent_action;
    float prior_prob;
    
    std::map<int, std::unique_ptr<MCTSNode>> children; // move_idx -> node
    int visit_count;
    float value_sum;
    bool is_expanded;

    MCTSNode(Board b, MCTSNode* p = nullptr, Move move = {0,0,MOVE_FLAG_NORMAL}, float prior = 0.0f)
        : board(b), parent(p), parent_action(move), prior_prob(prior),
          visit_count(0), value_sum(0.0f), is_expanded(false) {}

    float get_value() const {
        if (visit_count == 0) return 0.0f;
        return value_sum / visit_count;
    }

    bool is_leaf() const {
        return !is_expanded;
    }
};

class MCTS {
public:
    int num_simulations;
    float c_puct;
    // Temperature applied ONLY at root move selection
    float temperature;
    float dirichlet_alpha;
    float dirichlet_epsilon;

    MCTS(int sims = 800, float c = 1.5f, float temp = 1.0f, float d_alpha = 0.3f, float d_eps = 0.25f)
        : num_simulations(sims), c_puct(c), temperature(temp),
          dirichlet_alpha(d_alpha), dirichlet_epsilon(d_eps) {}

    // Main search function that takes a Python model as a callback
    std::map<int, float> search(Board& root_board, pybind11::object& model);
    
    // Helper to get moves in the same encoding as Python
    int encode_move_for_nn(Move move);

    void set_temperature(float t) {
        temperature = t;
    }

private:
    MCTSNode* select_child(MCTSNode* node);
    float expand_and_evaluate(MCTSNode* node, pybind11::object& model);
    void backpropagate(std::vector<MCTSNode*>& path, float value);
    void add_exploration_noise(MCTSNode* root);
};

#endif
