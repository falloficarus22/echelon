import numpy as np
from constants import *
from attacks import king_attacks_table, knight_attacks_table, pawn_attacks_table, count_bits
from magic_bitboards import get_rook_attacks, get_bishop_attacks, get_queen_attacks


class BoardState:
    def __init__(self):
        """
        Initialize a new board state.
        # Used np.uint64 instead of python integers to optimize bitwise operations
        """
        # Bitboards: 12 integers (6 piece types * 2 colors)
        # We initialize the bitboard to 0 and assign the intgers later
        self.bitboards = np.zeros(12, dtype=np.uint64)

        # Occupancies: 3 integers (White, Black, All)
        # These are "helper" bitboards used to check if a square is empty
        self.occupancies = np.zeros(3, dtype=np.uint64)

        # Game State Variables
        self.side = WHITE
        self.en_passant_sq = -1  # -1 means no en-passant square available
        self.castle_rights = 0
        self.halfmove_clock = 0
        self.position_history = []

    def get_position_hash(self):
        """Generate a simple hash of the current position for repetition detection."""
        # Hash based on: bitboards, side to move, castling rights, en passant
        hash_data = (
            tuple(self.bitboards.tolist()),
            self.side,
            self.castle_rights,
            self.en_passant_sq
        )
        return hash(hash_data)
    
    def is_threefold_repetition(self):
        """Check if current position has occurred 3+ times."""
        if len(self.position_history) < 8:  # Need at least 4 moves for repetition
            return False
        
        current_hash = self.get_position_hash()
        count = self.position_history.count(current_hash)
        return count >= 2  # Current position + 2 previous = 3 total
    
    def is_fifty_move_rule(self):
        """Check if 50-move rule applies (100 half-moves)."""
        return self.halfmove_clock >= 100
    
    def is_draw(self):
        """Check if position is a draw by repetition or 50-move rule."""
        return self.is_threefold_repetition() or self.is_fifty_move_rule()
    
    def parse_fen(self, fen):
        """
        Resets the board and loads the state from a FEN string.
        """
        # Clear everything first
        self.bitboards.fill(0)
        self.occupancies.fill(0)

        # Split the FEN string into components
        parts = fen.split()
        board_part = parts[0]
        turn_part = parts[1]
        castle_part = parts[2]

        # Parse board layout
        rank = 7
        file = 0

        piece_map = {
            "P": PAWN,
            "N": KNIGHT,
            "B": BISHOP,
            "R": ROOK,
            "Q": QUEEN,
            "K": KING,
            "p": PAWN,
            "n": KNIGHT,
            "b": BISHOP,
            "r": ROOK,
            "q": QUEEN,
            "k": KING,
        }

        for char in board_part:
            if char == "/":
                rank -= 1
                file = 0
            elif char.isdigit():
                # Digits mean empty squares
                empty_skips = int(char)
                file += empty_skips
            else:
                # Actual pieces
                piece_type = piece_map[char]
                color = WHITE if char.isupper() else BLACK

                # Global piece index
                index = piece_type + (color * 6)
                square = rank * 8 + file

                # Set the bit
                self.bitboards[index] |= np.uint64(1 << square)
                file += 1

        # Set the side to move
        self.side = WHITE if turn_part == "w" else BLACK

        # Update occupancies
        self.update_occupancies()

    def update_occupancies(self):
        """
        Combine individual bitboards into global occupancy boards.
        """
        # Clear existing occupancies
        self.occupancies.fill(0)

        # White piece loop
        for piece in range(6):
            self.occupancies[WHITE] |= self.bitboards[piece]

        # Black piece loop
        for piece in range(6):
            self.occupancies[BLACK] |= self.bitboards[piece + 6]

        # All occupancies
        self.occupancies[2] = self.occupancies[WHITE] | self.occupancies[BLACK]

    def get_piece_squares(self, bitboard):
        """
        Extracts all sqaure indices from a bitboard.
        Returns a list of sqaures (0 - 63) where bits are set.
        """
        squares = []

        # Copy of a bitboard
        temp = bitboard

        # While there are still bits set
        while temp:
            least_significant_bit = temp & -temp

            # Convert bits to square index
            # bit_length() gives us the position
            square = int(least_significant_bit).bit_length() - 1
            squares.append(square)

            temp &= temp - 1

        return squares

    def get_piece_at_square(self, square, side):
        """
        Returns the piece present on a given square
        """
        # Bitmask for this specific square
        square_mask = np.uint64(1) << square

        for piece_type in range(6):
            piece_bitboard = self.bitboards[piece_type + (side * 6)]

            if piece_bitboard & square_mask:
                return piece_type

        return None

    def get_king_square(self, side):
        """
        Find the square where the king of the given side is located. Returns the square index (0 - 63), -1 if not found.
        """
        king_bitboard = self.bitboards[KING + (side * 6)]

        if king_bitboard == 0:
            return -1  # Shouldn't happen ideally (in chess King is always present on the board)

        return self.get_piece_squares(king_bitboard)[0]

    def encode_move(self, source, target, piece, captured=None, promotion=0, flag=0):
        """
        Encode a move into a 21-bit integer.
        """
        # Convert None to 0
        # Handle captured piece encoding (0=None, 1=Pawn, etc.)
        captured_val = 0
        if captured is not None:
            captured_val = captured + 1

        if promotion is None:
            promotion = 0
        if flag is None:
            flag = 0

        # Ensure integers
        source = int(source)
        target = int(target)
        piece = int(piece)
        # captured = int(captured) # captured is handled via captured_val
        promotion = int(promotion)
        flag = int(flag)
        
        move = 0
        move |= source
        move |= target << 6
        move |= piece << 12
        move |= captured_val << 15
        move |= flag << 18
        
        return move

    def decode_move(self, move):
        """
        Decode a move integer into its components
        """
        source = move & 0x3F
        target = (move >> 6) & 0x3F
        piece = (move >> 12) & 0x7
        captured = (move >> 15) & 0x7
        flag = (move >> 18) & 0x7

        return {"from": source, "to": target, "piece": piece, "captured": captured, "flag": flag}

    def generate_king_moves(self, side):
        """
        Generates all the pseudo-legal moves for the King.
        """
        moves = []

        # Get king bitboard for this side
        king_bitboard = self.bitboards[KING + (side * 6)]
        king_squares = self.get_piece_squares(king_bitboard)
        own_pieces = self.occupancies[side]

        for king_sq in king_squares:
            attack_bitboard = king_attacks_table[king_sq]
            legal_targets = attack_bitboard & ~own_pieces
            target_squares = self.get_piece_squares(legal_targets)

            for target_sq in target_squares:
                captured_piece = self.get_piece_at_square(target_sq, 1 - side)



                move = self.encode_move(
                    source=king_sq,
                    target=target_sq,
                    piece=KING,
                    captured=captured_piece,
                    promotion=0,
                    flag=MOVE_FLAG_NORMAL,
                )
                moves.append(move)

        return moves

    def generate_knight_moves(self, side):
        """
        Generates all the pseudo-legal moves for the knight.
        """
        moves = []

        # Get knight bitboard for this side
        knight_bitboard = self.bitboards[KNIGHT + (side * 6)]
        knight_squares = self.get_piece_squares(knight_bitboard)
        own_pieces = self.occupancies[side]

        for knight_sq in knight_squares:
            attack_bitboard = knight_attacks_table[knight_sq]
            legal_targets = attack_bitboard & ~own_pieces
            target_squares = self.get_piece_squares(legal_targets)

            for target_sq in target_squares:
                captured_piece = self.get_piece_at_square(target_sq, 1 - side)



                move = self.encode_move(
                    source=knight_sq,
                    target=target_sq,
                    piece=KNIGHT,
                    captured=captured_piece,
                    promotion=0,
                    flag=MOVE_FLAG_NORMAL,
                )
                moves.append(move)

        return moves

    def generate_pawn_moves(self, side):
        moves = []

        pawn_bitboard = self.bitboards[PAWN + (side * 6)]
        pawn_squares = self.get_piece_squares(pawn_bitboard)
        occupancies = self.occupancies[2]

        for source_sq in pawn_squares:
            # Determine direction by side
            direction = 8 if side == WHITE else -8
            target_sq = source_sq + direction

            # Check if square in front is empty
            if 0 <= target_sq <= 63 and not (occupancies & (np.uint64(1) << np.uint64(target_sq))):
                # Is it a promotion
                if (side == WHITE and target_sq >= 56) or (side == BLACK and target_sq <= 7):
                    # We have to generate 4 seperate moves for Q, R, B, N
                    for promo in [
                        MOVE_FLAG_PROMOTION_QUEEN,
                        MOVE_FLAG_PROMOTION_ROOK,
                        MOVE_FLAG_PROMOTION_BISHOP,
                        MOVE_FLAG_PROMOTION_KNIGHT,
                    ]:
                        moves.append(self.encode_move(source_sq, target_sq, PAWN, flag=promo))
                else:
                    # Normal push
                    moves.append(self.encode_move(source_sq, target_sq, PAWN))

                    # Double pawn push (only if single pawn push was empty)
                    # Check if pawn is on it's starting square
                    is_start_rank = (side == WHITE and source_sq >= 8 and source_sq <= 15) or (
                        side == BLACK and source_sq >= 48 and source_sq <= 55
                    )
                    double_target = source_sq + (direction * 2)

                    if is_start_rank and not (
                        occupancies & (np.uint64(1) << np.uint64(double_target))
                    ):
                        moves.append(
                            self.encode_move(
                                source_sq, double_target, PAWN, flag=MOVE_FLAG_DOUBLE_PAWN_PUSH
                            )
                        )

            attacks = pawn_attacks_table[side][source_sq]
            opponent_side = 1 - side
            targets = attacks & self.occupancies[opponent_side]
            target_squares = self.get_piece_squares(targets)

            for target_sq in target_squares:
                # Get captured piece type
                captured_piece = self.get_piece_at_square(target_sq, opponent_side)



                # Check for promotion capture
                if (side == WHITE and target_sq >= 56) or (side == BLACK and target_sq <= 7):
                    for promo in [
                        MOVE_FLAG_PROMOTION_QUEEN,
                        MOVE_FLAG_PROMOTION_ROOK,
                        MOVE_FLAG_PROMOTION_BISHOP,
                        MOVE_FLAG_PROMOTION_KNIGHT,
                    ]:
                        moves.append(
                            self.encode_move(
                                source_sq, target_sq, PAWN, captured=captured_piece, flag=promo
                            )
                        )
                else:
                    moves.append(
                        self.encode_move(source_sq, target_sq, PAWN, captured=captured_piece)
                    )

            # En passant logic
            if self.en_passant_sq != -1:
                # If any of the square we capturing is en passant square
                ep_attacks = attacks & (np.uint64(1) << np.uint64(self.en_passant_sq))
                if ep_attacks:
                    moves.append(
                        self.encode_move(
                            source_sq,
                            self.en_passant_sq,
                            PAWN,
                            captured=PAWN,
                            flag=MOVE_FLAG_EN_PASSANT,
                        )
                    )

        return moves
    
    def make_move(self, move):
        """
        Makes a move on the board and updates all state variables.
        Returns a MoveHistory object containing irreversible state.
        
        This function:
        1. Saves irreversible state (for unmake)
        2. Updates bitboards
        3. Handles special moves (castling, en passant, promotion)
        4. Updates occupancies
        5. Switches sides
        """
        # Create history object to store irreversible state
        history = MoveHistory()
        history.en_passant_sq = self.en_passant_sq
        history.castle_rights = self.castle_rights
        history.halfmove_clock = self.halfmove_clock if hasattr(self, 'halfmove_clock') else 0
        self.position_history.append(self.get_position_hash())
        
        # Decode move
        decoded = self.decode_move(move)
        source = decoded['from']
        target = decoded['to']
        piece = decoded['piece']
        captured = decoded['captured']
        flag = decoded['flag']
        
        # Store captured piece
        history.captured_piece = captured
        
        # Get the piece bitboard index for current side
        piece_idx = piece + (self.side * 6)
        
        # Clear en passant square (will be set again if double pawn push)
        self.en_passant_sq = -1
        
        # Update halfmove clock (for 50-move rule)
        if piece == PAWN or captured != 0:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        
        # Handle different move types
        if flag == MOVE_FLAG_NORMAL:
            # Remove piece from source square
            self.bitboards[piece_idx] &= ~(np.uint64(1) << source)
            
            # Handle capture
            if captured != 0:
                captured_type = captured - 1
                captured_idx = captured_type + ((1 - self.side) * 6)
                self.bitboards[captured_idx] &= ~(np.uint64(1) << target)
            
            # Place piece on target square
            self.bitboards[piece_idx] |= (np.uint64(1) << target)
        
        elif flag == MOVE_FLAG_DOUBLE_PAWN_PUSH:
            # Remove pawn from source
            self.bitboards[piece_idx] &= ~(np.uint64(1) << source)
            # Place pawn on target
            self.bitboards[piece_idx] |= (np.uint64(1) << target)
            
            # Set en passant square (the square behind the pawn)
            if self.side == WHITE:
                self.en_passant_sq = source + 8
            else:
                self.en_passant_sq = source - 8
        
        elif flag == MOVE_FLAG_EN_PASSANT:
            # Remove pawn from source
            self.bitboards[piece_idx] &= ~(np.uint64(1) << source)
            # Place pawn on target
            self.bitboards[piece_idx] |= (np.uint64(1) << target)
            
            # Remove captured pawn (not on target square!)
            if self.side == WHITE:
                captured_sq = target - 8
            else:
                captured_sq = target + 8
            
            captured_idx = PAWN + ((1 - self.side) * 6)
            self.bitboards[captured_idx] &= ~(np.uint64(1) << captured_sq)
        
        elif flag in [MOVE_FLAG_PROMOTION_QUEEN, MOVE_FLAG_PROMOTION_ROOK, 
                    MOVE_FLAG_PROMOTION_BISHOP, MOVE_FLAG_PROMOTION_KNIGHT]:
            # Remove pawn from source
            self.bitboards[piece_idx] &= ~(np.uint64(1) << source)
            
            # Handle capture on target square
            if captured != 0:
                captured_type = captured - 1
                captured_idx = captured_type + ((1 - self.side) * 6)
                self.bitboards[captured_idx] &= ~(np.uint64(1) << target)
            
            # Determine promotion piece
            promo_piece = {
                MOVE_FLAG_PROMOTION_QUEEN: QUEEN,
                MOVE_FLAG_PROMOTION_ROOK: ROOK,
                MOVE_FLAG_PROMOTION_BISHOP: BISHOP,
                MOVE_FLAG_PROMOTION_KNIGHT: KNIGHT
            }[flag]
            
            # Place promoted piece on target
            promo_idx = promo_piece + (self.side * 6)
            self.bitboards[promo_idx] |= (np.uint64(1) << target)
        
        elif flag == MOVE_FLAG_CASTLING:
            # Move king
            self.bitboards[piece_idx] &= ~(np.uint64(1) << source)
            self.bitboards[piece_idx] |= (np.uint64(1) << target)
            
            # Move rook
            rook_idx = ROOK + (self.side * 6)
            
            # Determine rook source and target based on king's movement
            if target > source:  # Kingside castling
                if self.side == WHITE:
                    # White kingside: H1 -> F1
                    self.bitboards[rook_idx] &= ~(np.uint64(1) << H1)
                    self.bitboards[rook_idx] |= (np.uint64(1) << F1)
                else:
                    # Black kingside: H8 -> F8
                    self.bitboards[rook_idx] &= ~(np.uint64(1) << H8)
                    self.bitboards[rook_idx] |= (np.uint64(1) << F8)
            else:  # Queenside castling
                if self.side == WHITE:
                    # White queenside: A1 -> D1
                    self.bitboards[rook_idx] &= ~(np.uint64(1) << A1)
                    self.bitboards[rook_idx] |= (np.uint64(1) << D1)
                else:
                    # Black queenside: A8 -> D8
                    self.bitboards[rook_idx] &= ~(np.uint64(1) << A8)
                    self.bitboards[rook_idx] |= (np.uint64(1) << D8)
        
        # Update castling rights
        # If king moves, remove all castling rights for that side
        if piece == KING:
            if self.side == WHITE:
                self.castle_rights &= 0b1100  # Remove white's rights (bits 0,1)
            else:
                self.castle_rights &= 0b0011  # Remove black's rights (bits 2,3)
        
        # If rook moves from or piece captured on corner squares, update castling rights
        # We'll use a simple lookup to determine which castling rights to remove
        castling_rights_mask = [
            0b1110, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1101,  # Rank 1
            0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111,  # Rank 2-7
            0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111,
            0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111,
            0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111,
            0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111,
            0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111,
            0b1011, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111, 0b0111   # Rank 8
        ]
        
        self.castle_rights &= castling_rights_mask[source]
        self.castle_rights &= castling_rights_mask[target]
        
        # Update occupancies
        self.update_occupancies()
        
        # Switch sides
        self.side = 1 - self.side
        
        return history

    def unmake_move(self, move, history):
        """
        Unmakes a move and restores the board to its previous state.
        
        Args:
            move: The encoded move to unmake
            history: MoveHistory object returned by make_move()
        """
        if self.position_history:
            self.position_history.pop()

        # Switch side back first
        self.side = 1 - self.side
        
        # Decode move
        decoded = self.decode_move(move)
        source = decoded['from']
        target = decoded['to']
        piece = decoded['piece']
        captured = history.captured_piece  # Use captured from history, not move
        flag = decoded['flag']
        
        # Get the piece bitboard index for current side
        piece_idx = piece + (self.side * 6)
        
        # Handle different move types (reverse of make_move)
        if flag == MOVE_FLAG_NORMAL:
            # Remove piece from target square
            self.bitboards[piece_idx] &= ~(np.uint64(1) << target)
            
            # Place piece back on source square
            self.bitboards[piece_idx] |= (np.uint64(1) << source)
            
            # Restore captured piece
            if captured != 0:
                captured_type = captured - 1
                captured_idx = captured_type + ((1 - self.side) * 6)
                self.bitboards[captured_idx] |= (np.uint64(1) << target)
        
        elif flag == MOVE_FLAG_DOUBLE_PAWN_PUSH:
            # Remove pawn from target
            self.bitboards[piece_idx] &= ~(np.uint64(1) << target)
            # Place pawn back on source
            self.bitboards[piece_idx] |= (np.uint64(1) << source)
        
        elif flag == MOVE_FLAG_EN_PASSANT:
            # Remove pawn from target
            self.bitboards[piece_idx] &= ~(np.uint64(1) << target)
            # Place pawn back on source
            self.bitboards[piece_idx] |= (np.uint64(1) << source)
            
            # Restore captured pawn (not on target square!)
            if self.side == WHITE:
                captured_sq = target - 8
            else:
                captured_sq = target + 8
            
            captured_idx = PAWN + ((1 - self.side) * 6)
            self.bitboards[captured_idx] |= (np.uint64(1) << captured_sq)
        
        elif flag in [MOVE_FLAG_PROMOTION_QUEEN, MOVE_FLAG_PROMOTION_ROOK,
                    MOVE_FLAG_PROMOTION_BISHOP, MOVE_FLAG_PROMOTION_KNIGHT]:
            # Determine which piece was promoted to
            promo_piece = {
                MOVE_FLAG_PROMOTION_QUEEN: QUEEN,
                MOVE_FLAG_PROMOTION_ROOK: ROOK,
                MOVE_FLAG_PROMOTION_BISHOP: BISHOP,
                MOVE_FLAG_PROMOTION_KNIGHT: KNIGHT
            }[flag]
            
            # Remove promoted piece from target
            promo_idx = promo_piece + (self.side * 6)
            self.bitboards[promo_idx] &= ~(np.uint64(1) << target)
            
            # Place pawn back on source
            self.bitboards[piece_idx] |= (np.uint64(1) << source)
            
            # Restore captured piece if there was one
            if captured != 0:
                captured_type = captured - 1
                captured_idx = captured_type + ((1 - self.side) * 6)
                self.bitboards[captured_idx] |= (np.uint64(1) << target)
        
        elif flag == MOVE_FLAG_CASTLING:
            # Move king back
            self.bitboards[piece_idx] &= ~(np.uint64(1) << target)
            self.bitboards[piece_idx] |= (np.uint64(1) << source)
            
            # Move rook back
            rook_idx = ROOK + (self.side * 6)
            
            if target > source:  # Was kingside castling
                if self.side == WHITE:
                    self.bitboards[rook_idx] &= ~(np.uint64(1) << F1)
                    self.bitboards[rook_idx] |= (np.uint64(1) << H1)
                else:
                    self.bitboards[rook_idx] &= ~(np.uint64(1) << F8)
                    self.bitboards[rook_idx] |= (np.uint64(1) << H8)
            else:  # Was queenside castling
                if self.side == WHITE:
                    self.bitboards[rook_idx] &= ~(np.uint64(1) << D1)
                    self.bitboards[rook_idx] |= (np.uint64(1) << A1)
                else:
                    self.bitboards[rook_idx] &= ~(np.uint64(1) << D8)
                    self.bitboards[rook_idx] |= (np.uint64(1) << A8)
        
        # Restore irreversible state from history
        self.en_passant_sq = history.en_passant_sq
        self.castle_rights = history.castle_rights
        self.halfmove_clock = history.halfmove_clock
        
        # Update occupancies
        self.update_occupancies()

    def is_square_attacked(self, square, attacking_side):
        """
        Checks if a square is attacked by any piece of the given side.
        
        Args:
            square: Square index (0-63) to check
            attacking_side: Side (WHITE or BLACK) doing the attacking
        
        Returns:
            True if the square is attacked, False otherwise
        
        This function checks for attacks from:
        - Pawns (diagonal attacks)
        - Knights (L-shaped jumps)
        - Bishops (diagonal rays)
        - Rooks (straight rays)
        - Queens (bishop + rook rays)
        - King (one square in any direction)
        """
        
        # Get all pieces of the attacking side
        all_occupancy = self.occupancies[2]
        
        # Check for pawn attacks
        # We check if our square can "attack" enemy pawns (inverse logic)
        # If we're checking white attacks, we look from black pawn perspective
        defending_side = 1 - attacking_side
        pawn_attacks = pawn_attacks_table[defending_side][square]
        attacking_pawns = self.bitboards[PAWN + (attacking_side * 6)]
        if pawn_attacks & attacking_pawns:
            return True
        
        # Check for knight attacks
        knight_attacks = knight_attacks_table[square]
        attacking_knights = self.bitboards[KNIGHT + (attacking_side * 6)]
        if knight_attacks & attacking_knights:
            return True
        
        # Check for bishop attacks (and queen diagonal)
        bishop_attacks = get_bishop_attacks(square, all_occupancy)
        attacking_bishops = self.bitboards[BISHOP + (attacking_side * 6)]
        attacking_queens = self.bitboards[QUEEN + (attacking_side * 6)]
        if bishop_attacks & (attacking_bishops | attacking_queens):
            return True
        
        # Check for rook attacks (and queen straight)
        rook_attacks = get_rook_attacks(square, all_occupancy)
        attacking_rooks = self.bitboards[ROOK + (attacking_side * 6)]
        if rook_attacks & (attacking_rooks | attacking_queens):
            return True
        
        # Check for king attacks
        king_attacks = king_attacks_table[square]
        attacking_king = self.bitboards[KING + (attacking_side * 6)]
        if king_attacks & attacking_king:
            return True
        
        return False

    def is_in_check(self, side):
        """
        Checks if the given side's king is in check.
        
        Args:
            side: Side (WHITE or BLACK) to check
        
        Returns:
            True if the king is in check, False otherwise
        """
        king_square = self.get_king_square(side)
        
        if king_square == -1:
            # No king found (shouldn't happen in valid positions)
            return False
        
        # Check if enemy is attacking the king square
        return self.is_square_attacked(king_square, 1 - side)

    def generate_legal_moves(self, side):
        """
        Generates all legal moves (pseudo-legal moves that don't leave king in check).
        
        This is the function you'll use in your search algorithm.
        """
        if self.is_draw():
            return []
        
        pseudo_legal_moves = self.generate_all_moves(side)
        legal_moves = []
        
        for move in pseudo_legal_moves:
            # Make the move
            history = self.make_move(move)
            
            # Check if our king is in check after the move
            # Note: side has switched in make_move, so we check (1 - side)
            if not self.is_in_check(1 - self.side):
                legal_moves.append(move)
            
            # Unmake the move
            self.unmake_move(move, history)
        
        return legal_moves

    def can_castle_kingside(self, side):
        """
        Checks if kingside castling is legal.
        """
        # Check castling rights
        if side == WHITE:
            if not (self.castle_rights & 0b0001):  # White kingside
                return False
            king_sq = E1
            rook_sq = H1
            squares_between = [F1, G1]
            squares_not_attacked = [E1, F1, G1]
        else:
            if not (self.castle_rights & 0b0100):  # Black kingside
                return False
            king_sq = E8
            rook_sq = H8
            squares_between = [F8, G8]
            squares_not_attacked = [E8, F8, G8]
        
        # Check if squares between are empty
        for sq in squares_between:
            if self.occupancies[2] & (np.uint64(1) << sq):
                return False
        
        # Check if king is in check or passes through check
        enemy_side = 1 - side
        for sq in squares_not_attacked:
            if self.is_square_attacked(sq, enemy_side):
                return False
        
        return True

    def can_castle_queenside(self, side):
        """
        Checks if queenside castling is legal.
        """
        # Check castling rights
        if side == WHITE:
            if not (self.castle_rights & 0b0010):  # White queenside
                return False
            king_sq = E1
            rook_sq = A1
            squares_between = [B1, C1, D1]
            squares_not_attacked = [E1, D1, C1]
        else:
            if not (self.castle_rights & 0b1000):  # Black queenside
                return False
            king_sq = E8
            rook_sq = A8
            squares_between = [B8, C8, D8]
            squares_not_attacked = [E8, D8, C8]
        
        # Check if squares between are empty
        for sq in squares_between:
            if self.occupancies[2] & (np.uint64(1) << sq):
                return False
        
        # Check if king is in check or passes through check
        enemy_side = 1 - side
        for sq in squares_not_attacked:
            if self.is_square_attacked(sq, enemy_side):
                return False
        
        return True

    def generate_bishop_moves(self, side):
        """
        Generates all pseudo-legal bishop moves
        """
        moves = []
        bishop_bitboard = self.bitboards[BISHOP + (side * 6)]
        bishop_squares = self.get_piece_squares(bishop_bitboard)
        own_pieces = self.occupancies[side]
        all_pieces = self.occupancies[2]

        for bishop_sq in bishop_squares:
            # Get bishop attacks using magic bitboards
            attack_bitboard = get_bishop_attacks(bishop_sq, all_pieces)
            legal_targets = attack_bitboard & ~own_pieces
            target_squares = self.get_piece_squares(legal_targets)

            for target_sq in target_squares:
                captured_piece = self.get_piece_at_square(target_sq, 1 - side)
                


                move = self.encode_move(
                    source = bishop_sq,
                    target = target_sq,
                    piece = BISHOP,
                    captured = captured_piece,
                    promotion = 0,
                    flag = MOVE_FLAG_NORMAL
                )
                moves.append(move)

        return moves
    
    def generate_rook_moves(self, side):
        """
        Generate all pseudo-legal rook moves
        """
        moves = []
        rook_bitboard = self.bitboards[ROOK + (side * 6)]
        rook_squares = self.get_piece_squares(rook_bitboard)
        own_pieces = self.occupancies[side]
        all_pieces = self.occupancies[2]

        for rook_sq in rook_squares:
            attack_bitboard = get_rook_attacks(rook_sq, all_pieces)
            legal_targets = attack_bitboard & ~own_pieces
            target_squares = self.get_piece_squares(legal_targets)

            for target_sq in target_squares:
                captured_piece = self.get_piece_at_square(target_sq, 1 - side)
                


                move = self.encode_move(
                    source = rook_sq,
                    target = target_sq,
                    piece = ROOK,
                    captured = captured_piece,
                    promotion = 0,
                    flag = MOVE_FLAG_NORMAL
                )
                moves.append(move)

        return moves
    
    def generate_queen_moves(self, side):
        """
        Generate all pseudo-legal moves for queen
        """
        moves = []
        queen_bitboard = self.bitboards[QUEEN + (side * 6)]
        queen_squares = self.get_piece_squares(queen_bitboard)
        own_pieces = self.occupancies[side]
        all_pieces = self.occupancies[2]

        for queen_sq in queen_squares:
            # Get queen attacks (rook + bishop) using magic bitboards
            attack_bitboard = get_queen_attacks(queen_sq, all_pieces)
            legal_targets = attack_bitboard & ~own_pieces
            target_squares = self.get_piece_squares(legal_targets)

            for target_sq in target_squares:
                captured_piece = self.get_piece_at_square(target_sq, 1 - side)


                
                move = self.encode_move(
                    source = queen_sq,
                    target = target_sq,
                    piece = QUEEN,
                    captured = captured_piece,
                    promotion = 0,
                    flag = MOVE_FLAG_NORMAL
                )
                moves.append(move)

        return moves

    def generate_all_moves(self, side):
        """
        Generates all pseudo-legal moves for all the pieces of a given side.
        This includes captures, quiet moves and castling moves
        """
        moves = []

        # Gather moves from all pieces
        moves.extend(self.generate_bishop_moves(side))
        moves.extend(self.generate_rook_moves(side))
        moves.extend(self.generate_queen_moves(side))
        moves.extend(self.generate_king_moves(side))
        moves.extend(self.generate_pawn_moves(side))
        moves.extend(self.generate_knight_moves(side))

        # Handle Castling
        # Kingside
        if self.can_castle_kingside(side):
            target_sq = G1 if side == WHITE else G8
            source_sq = E1 if side == WHITE else E8
            moves.append(self.encode_move(
                source = source_sq,
                target = target_sq,
                piece = KING,
                flag = MOVE_FLAG_CASTLING
            ))

        # Queenside
        if self.can_castle_queenside(side):
            target_sq = C1 if side == WHITE else C8
            source_sq = E1 if side == WHITE else E8
            moves.append(self.encode_move(
                source = source_sq,
                target = target_sq,
                piece = KING,
                flag = MOVE_FLAG_CASTLING
            ))

        return moves
    
    def evaluate(self):
        """
        Evaluation for the engine which returns a score in centipawns (100 = 1 pawn).
        Positive = Advantage for White, Negatiev = Advantage for Black
        """
        score = 0

        # Material values
        # 100 = pawn, 320 = knight, 330 = bishop, rook = 500, queen = 900
        # 0 = king because it is always there
        PIECE_VALUES = [100, 320, 330, 500, 900, 0]

        for piece_type in range(6):
            # Add white material
            white_count = count_bits(self.bitboards[piece_type + (WHITE * 6)])
            score += white_count * PIECE_VALUES[piece_type]

            # Subtract black material
            black_count = count_bits(self.bitboards[piece_type + (BLACK * 6)])
            score -= black_count * PIECE_VALUES[piece_type]

        # Side to move bonus
        return score if self.side == WHITE else -score
    
    def tensorize_board(self):
        """
        Tensorize current board state in a PyTorch tensor (13, 8, 8)
        12 planes for pieces, 1 plane for the side to move
        """
        import torch

        # Initialize an empty array [channels, height, width]
        data = np.zeros((13, 8, 8), dtype = np.float32)

        for p_idx in range(12):
            bb = self.bitboards[p_idx]
            # Optimization: Only iterate over bits that are SET
            while bb:
                lsb = bb & -bb
                idx = int(lsb).bit_length() - 1
                data[p_idx, idx // 8, idx % 8] = 1.0
                bb &= bb - 1

        # 13th plane (side to move)
        if self.side == WHITE:
            data[12, :, :] = 1.0

        return torch.from_numpy(data)

    def get_greedy_move(self):
        """
        Finds the best move using only material evaluation (1-ply search).
        Used as a baseline for Elo benchmarking.
        """
        legal_moves = self.generate_legal_moves(self.side)
        if not legal_moves:
            return None
            
        best_move = None
        best_score = -99999
        
        # We evaluate from perspective of player to move
        for move in legal_moves:
            hist = self.make_move(move)
            # engine.evaluate() returns score from white's perspective,
            # but usually engines flip it or we handle it.
            # Let's assume white = positive, black = negative.
            score = self.evaluate() 
            self.unmake_move(move, hist)
            
            # Since evaluate() returns score for white, 
            # if we are black we want to minimize it.
            actual_score = score if self.side == WHITE else -score
            
            if actual_score > best_score:
                best_score = actual_score
                best_move = move
                    
        return best_move
    
    
    

class MoveHistory:
    """
    Stores irreversible state that cannot be deduced from the board position.
    This info is needed to unmake moves correctly.
    """
    def __init__(self):
        self.captured_piece = 0
        self.en_passant_sq = -1
        self.castle_rights = 0
        self.halfmove_clock = 0


if __name__ == "__main__":
    # Create the engine
    engine = BoardState()

    # Standard start position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    engine.parse_fen(start_fen)

    # Test 1: Verification
    # Print the white pawns bitboard
    print_bitboard(engine.bitboards[0])

    # Test 2: All occupied squares
    print_bitboard(engine.occupancies[2])

    # Test 3: King on d4
    print_bitboard(king_attacks_table[D4])

    # Test 4: Knight on e5
    print_bitboard(knight_attacks_table[E5])

    # Test 5: White pawn on b3
    print_bitboard(pawn_attacks_table[WHITE][B3])
