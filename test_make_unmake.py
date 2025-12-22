import numpy as np
from engine import BoardState
from constants import *

def perft(engine, depth):
    """
    Perft (Performance Test) function.
    Counts the number of leaf nodes at a given depth.
    Used for debugging move generation and make/unmake.
    """
    if depth == 0:
        return 1
    
    nodes = 0
    moves = engine.generate_all_moves(engine.side)
    
    for move in moves:
        history = engine.make_move(move)
        nodes += perft(engine, depth - 1)
        engine.unmake_move(move, history)
    
    return nodes

def perft_divide(engine, depth):
    """
    Perft with division - shows node count for each move from current position.
    Useful for debugging specific moves.
    """
    if depth == 0:
        return 1
    
    total_nodes = 0
    moves = engine.generate_all_moves(engine.side)
    
    print(f"\nPerft divide at depth {depth}:")
    print(f"Total moves: {len(moves)}\n")
    
    for move in moves:
        decoded = engine.decode_move(move)
        move_str = f"{chr(ord('a') + decoded['from'] % 8)}{decoded['from'] // 8 + 1}"
        move_str += f"{chr(ord('a') + decoded['to'] % 8)}{decoded['to'] // 8 + 1}"
        
        history = engine.make_move(move)
        nodes = perft(engine, depth - 1)
        engine.unmake_move(move, history)
        
        print(f"{move_str}: {nodes}")
        total_nodes += nodes
    
    print(f"\nTotal nodes: {total_nodes}")
    return total_nodes

def test_perft():
    """
    Test perft against known values for the starting position.
    
    Known perft values for starting position:
    Depth 1: 20 nodes
    Depth 2: 400 nodes
    Depth 3: 8,902 nodes
    Depth 4: 197,281 nodes
    Depth 5: 4,865,609 nodes
    Depth 6: 119,060,324 nodes
    """
    engine = BoardState()
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    engine.parse_fen(start_fen)
    
    expected_results = {
        1: 20,
        2: 400,
        3: 8902,
        4: 197281,
        5: 4865609,
        # 6: 119060324  # Takes longer, comment out for quick tests
    }
    
    print("Testing Perft from starting position...")
    print("=" * 50)
    
    all_passed = True
    
    for depth, expected in expected_results.items():
        print(f"\nDepth {depth}:")
        result = perft(engine, depth)
        passed = result == expected
        
        status = "PASS" if passed else "FAIL"
        print(f"  Expected: {expected:,}")
        print(f"  Got:      {result:,}")
        print(f"  {status}")
        
        if not passed:
            all_passed = False
            print(f"  Difference: {result - expected:+,}")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed - debugging needed")
    
    return all_passed

def test_make_unmake_reversibility():
    """
    Test that make/unmake correctly reverses the board state.
    """
    print("\nTesting make/unmake reversibility...")
    print("=" * 50)
    
    engine = BoardState()
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    engine.parse_fen(start_fen)
    
    # Save initial state
    initial_bitboards = engine.bitboards.copy()
    initial_occupancies = engine.occupancies.copy()
    initial_side = engine.side
    
    # Generate and make all moves, then unmake them
    moves = engine.generate_all_moves(engine.side)
    
    print(f"Testing {len(moves)} moves...")
    
    for i, move in enumerate(moves):
        # Make move
        history = engine.make_move(move)
        
        # Unmake move
        engine.unmake_move(move, history)
        
        # Check if state is restored
        if not np.array_equal(engine.bitboards, initial_bitboards):
            print(f"FAIL: Bitboards not restored after move {i}")
            return False
        
        if not np.array_equal(engine.occupancies, initial_occupancies):
            print(f"FAIL: Occupancies not restored after move {i}")
            return False
        
        if engine.side != initial_side:
            print(f"FAIL: Side not restored after move {i}")
            return False
    
    print(f"All {len(moves)} moves correctly reversed")
    print("Make/unmake reversibility test passed!")
    return True

def test_specific_positions():
    """
    Test perft on specific interesting positions.
    """
    print("\nTesting specific positions...")
    print("=" * 50)
    
    test_positions = [
        {
            'name': 'Kiwipete',
            'fen': 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
            'depth': 3,
            'expected': 97862
        },
        {
            'name': 'Position 3',
            'fen': '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
            'depth': 4,
            'expected': 43238
        },
        {
            'name': 'Position 4',
            'fen': 'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1',
            'depth': 3,
            'expected': 9467
        },
    ]
    
    all_passed = True
    
    for test in test_positions:
        engine = BoardState()
        engine.parse_fen(test['fen'])
        
        print(f"\n{test['name']}:")
        print(f"FEN: {test['fen']}")
        print(f"Depth {test['depth']}:")
        
        result = perft(engine, test['depth'])
        passed = result == test['expected']
        
        status = "PASS" if passed else "FAIL"
        print(f"  Expected: {test['expected']:,}")
        print(f"  Got:      {result:,}")
        print(f"  {status}")
        
        if not passed:
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("MAKE/UNMAKE TEST SUITE")
    print("=" * 50)
    
    # Test 1: Reversibility
    test1 = test_make_unmake_reversibility()
    
    # Test 2: Perft from starting position
    test2 = test_perft()
    
    # Test 3: Specific positions
    # Uncomment when basic tests pass
    # test3 = test_specific_positions()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    if test1 and test2:
        print("ALL TESTS PASSED!")
        print("\nYour make_move() and unmake_move() are working correctly!")
    else:
        print("SOME TESTS FAILED")
        print("\nDebugging tips:")
        print("1. Check if all move flags are handled in make/unmake")
        print("2. Verify castling rights updates")
        print("3. Check en passant handling")
        print("4. Use perft_divide() to find which moves are wrong")