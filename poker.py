import random
import time
from collections import defaultdict
import math

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    
    def __repr__(self):
        return f"{self.rank}{self.suit}"
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

class Deck:
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♥', '♦', '♣']
    
    def __init__(self):
        self.cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]
        self.shuffle()
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def draw(self, n):
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn
    
    def remove_cards(self, cards_to_remove):
        self.cards = [card for card in self.cards if card not in cards_to_remove]

class HandEvaluator:
    
    @staticmethod
    def evaluate_hand(hole_cards, community_cards):
        all_cards = hole_cards + community_cards
        best_rank = 0
        best_hand = None
        from itertools import combinations # Check all 5 card combinations
        for five_cards in combinations(all_cards, 5):
            current_rank = HandEvaluator._evaluate_five_card(five_cards)
            if current_rank > best_rank:
                best_rank = current_rank
                best_hand = five_cards
        
        return best_rank
    
    @staticmethod
    def _evaluate_five_card(cards):
        ranks = [card.rank for card in cards]
        suits = [card.suit for card in cards]
        rank_counts = defaultdict(int)
        for rank in ranks:
            rank_counts[rank] += 1
        sorted_groups = sorted(rank_counts.items(), key=lambda x: (-x[1], -HandEvaluator._rank_to_value(x[0])))
        suits_counts = defaultdict(int)
        for suit in suits:
            suits_counts[suit] += 1
        
        # check flush
        flush = any(count >= 5 for count in suits_counts.values())
        
        # check Straight
        unique_ranks = sorted({HandEvaluator._rank_to_value(rank) for rank in ranks}, reverse=True)
        straight = False
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                straight = True
                break
        #Low straight (A-2-3-4-5)
        if set(unique_ranks) >= {14, 2, 3, 4, 5}:
            straight = True
        
        # Straight flush
        if flush and straight:
            return 8_000_000 + max(unique_ranks)
        
        # Four of a kind
        if sorted_groups[0][1] == 4:
            return 7_000_000 + HandEvaluator._rank_to_value(sorted_groups[0][0])
        
        # Full house
        if sorted_groups[0][1] == 3 and sorted_groups[1][1] >= 2:
            return 6_000_000 + HandEvaluator._rank_to_value(sorted_groups[0][0]) * 100 + HandEvaluator._rank_to_value(sorted_groups[1][0])
        
        # Flush
        if flush:
            return 5_000_000 + sum(HandEvaluator._rank_to_value(rank) * (15 ** (4 - i)) for i, rank in enumerate(sorted(ranks, key=lambda x: -HandEvaluator._rank_to_value(x))[:5]))
        
        # Straight
        if straight:
            return 4_000_000 + max(unique_ranks)
        
        # Three of a kind
        if sorted_groups[0][1] == 3:
            return 3_000_000 + HandEvaluator._rank_to_value(sorted_groups[0][0]) * 10000 + sum(HandEvaluator._rank_to_value(r) for r in ranks if r != sorted_groups[0][0])
        
        # Two pair
        if sorted_groups[0][1] == 2 and sorted_groups[1][1] == 2:
            return 2_000_000 + HandEvaluator._rank_to_value(sorted_groups[0][0]) * 10000 + HandEvaluator._rank_to_value(sorted_groups[1][0]) * 100 + HandEvaluator._rank_to_value(sorted_groups[2][0])
        
        # One pair
        if sorted_groups[0][1] == 2:
            return 1_000_000 + HandEvaluator._rank_to_value(sorted_groups[0][0]) * 10000 + sum(HandEvaluator._rank_to_value(r) for r in ranks if r != sorted_groups[0][0])
        
        # High card
        return sum(HandEvaluator._rank_to_value(rank) * (15 ** (4 - i)) for i, rank in enumerate(sorted(ranks, key=lambda x: -HandEvaluator._rank_to_value(x))[:5]))

        # I hope that's all
        # That's all this function is done 
    
    @staticmethod
    def _rank_to_value(rank):
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 
                       '7': 7, '8': 8, '9': 9, '10': 10, 
                       'J': 11, 'Q': 12, 'K': 13, 'A': 14} #card dictionary
        return rank_values[rank]

class MCTSNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action  # 'stay' or 'fold'
        self.children = []
        self.wins = 0
        self.simulations = 0
        self.untried_actions = ['stay', 'fold'] if action is None else []
    
    def select_child(self, exploration_weight=1.0):
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            if child.simulations == 0:
                score = float('inf')
            else:         #use ucb1
                exploitation = child.wins / child.simulations
                exploration = math.sqrt(math.log(self.simulations + 1e-6) / child.simulations + 1e-6) # added the + 1e-6 to avoid division by 0
                score = exploitation + exploration_weight * exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self):
        action = self.untried_actions.pop()
        new_node = MCTSNode(parent=self, action=action)
        self.children.append(new_node)
        return new_node
    
    def update(self, result):
        self.simulations += 1
        self.wins += result
        if self.parent:
            self.parent.update(result)

class PokerBot:    
    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        self.time_limit = 9.5  # seconds (leaving 0.5s buffer) (we we don't exceed 10s)
    
    def decide_action(self, hole_cards, community_cards, game_stage):
        root = MCTSNode()
        start_time = time.time()
        while time.time() - start_time < self.time_limit:
            node = root
            state = self._simulate_state(hole_cards, community_cards, game_stage)
            
            # Selection
            while node.untried_actions == [] and node.children != []:
                node = node.select_child()
            
            # Expansion
            if node.untried_actions != []:
                node = node.expand()
            
            # Simulation
            result = self._simulate_random_outcome(state, node.action)
            
            # Backpropagation
            node.update(result)
        
        # Make decision based on MCTS results
        stay_simulations = 0
        stay_wins = 0
        fold_simulations = 0
        fold_wins = 0
        
        for child in root.children:
            if child.action == 'stay':
                stay_simulations = child.simulations
                stay_wins = child.wins
            elif child.action == 'fold':
                fold_simulations = child.simulations
                fold_wins = child.wins
        
        # default is stay
        if stay_simulations == 0:
            return 'stay'
        
        win_probability = stay_wins / stay_simulations
        return 'stay' if win_probability >= 0.5 else 'fold'
    
    def _simulate_state(self, hole_cards, community_cards, game_stage):
        deck = Deck()
        known_cards = hole_cards + community_cards
        deck.remove_cards(known_cards)
        cards_to_deal = 0
        if game_stage == 'preflop':
            cards_to_deal = 5  # flop(3) + turn(1) + river(1)
        elif game_stage == 'flop':
            cards_to_deal = 2  # turn(1) + river(1)
        elif game_stage == 'turn':
            cards_to_deal = 1  # river(1)
        else:  # river
            cards_to_deal = 0
        
        return {
            'hole_cards': hole_cards,
            'community_cards': community_cards,
            'deck': deck,
            'cards_to_deal': cards_to_deal
        }
    
    def _simulate_random_outcome(self, state, action):
        if action == 'fold':
            return 0  # Folding always loses
        # Verified, Do this

        # Simulate opponent's hole cards
        deck = state['deck']
        remaining_deck = deck.cards.copy()
        possible_opponent_cards = [card for card in remaining_deck 
                                 if card not in state['hole_cards'] 
                                 and card not in state['community_cards']]
        
        # Randomly select opponent's hole cards
        if len(possible_opponent_cards) < 2:
            return 0  # Not enough cards for opponent
            
        opponent_hole = random.sample(possible_opponent_cards, 2)
        #opponent_hole = deck.draw(2)
        for card in opponent_hole:
            remaining_deck.remove(card)
        future_community = state['community_cards'].copy()
        if state['cards_to_deal'] > 0 and len(remaining_deck) >= state['cards_to_deal']:
            future_community += random.sample(remaining_deck, state['cards_to_deal'])
        my_hand_rank = self.hand_evaluator.evaluate_hand(state['hole_cards'], future_community)
        opponent_hand_rank = self.hand_evaluator.evaluate_hand(opponent_hole, future_community)
        
        if my_hand_rank > opponent_hand_rank:
            return 1  # Win
        elif my_hand_rank == opponent_hand_rank:
            return 0.5  # Tie
        else:
            return 0  # Loss

def play_hand(): #this is ONE SINGLEHAND ONLY
    deck = Deck()
    bot = PokerBot()
    bot_hole = deck.draw(2)
    opponent_hole = deck.draw(2)
    stages = ['preflop', 'flop', 'turn', 'river']
    community_cards = []
    
    for stage in stages:
        print(f"\n--- {stage.upper()} ---")
        print(f"Your cards: {bot_hole}")
        print(f"Community cards: {community_cards}")
        
        # Deal community cards if needed
        if stage == 'flop' and len(community_cards) == 0:
            community_cards += deck.draw(3)
        elif stage in ['turn', 'river'] and len(community_cards) < 5:
            community_cards += deck.draw(1)
        
        print(f"Updated community cards: {community_cards}")
        
        decision = bot.decide_action(bot_hole, community_cards, stage)
        print(f"Bot decides to: {decision}")
        
        if decision == 'fold':
            print("Bot folded. Opponent wins!")
            return # Wait, did the instructions say this?
        # It's implied ?
        
    print("\n--- SHOWDOWN ---")
    print(f"Your cards: {bot_hole}")
    print(f"Opponent cards: {opponent_hole}")
    print(f"Community cards: {community_cards}")
    
    bot_rank = HandEvaluator().evaluate_hand(bot_hole, community_cards)
    opponent_rank = HandEvaluator().evaluate_hand(opponent_hole, community_cards)
    
    if bot_rank > opponent_rank:
        print("You win")
    elif bot_rank == opponent_rank:
        print("Tie")
    else:
        print("You Lose")

if __name__ == "__main__":
    print("Starting")
    play_hand()
    # Everthing is fine
    # Do not mess with this anymore