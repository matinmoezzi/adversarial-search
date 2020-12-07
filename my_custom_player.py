
from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def minimax(self, state, depth):
        def min_val(state, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.eval_func(state)
            v = float("inf")
            for action in state.actions():
                v = min(v, max_val(state.result(action), depth - 1))
            return v

        def max_val(state, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.eval_func(state)
            v = float("-inf")
            for action in state.actions():
                v = max(v, min_val(state.result(action), depth - 1))
            return v
        return max(state.actions(), key=lambda x: min_val(state.result(x), depth - 1))

    def alpha_beta_pruning(self, state, depth, heuristic):
        def alpha_beta_min_val(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.eval_func(state, heuristic)
            v = float("inf")
            for action in state.actions():
                v = min(v, alpha_beta_max_val(
                    state.result(action), alpha, beta, depth - 1))
                if alpha >= v:
                    return v
                beta = min(beta, v)
            return v

        def alpha_beta_max_val(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.eval_func(state, heuristic)
            v = float("-inf")
            for action in state.actions():
                v = max(v, alpha_beta_min_val(
                    state.result(action), alpha, beta, depth - 1))
                if beta <= v:
                    return v
                alpha = max(alpha, v)
            return v
        return max(state.actions(), key=lambda x: alpha_beta_min_val(state.result(x), float("-inf"), float("inf"), depth - 1))

    def eval_func(self, state, heuristic):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)

        # Blocking the Opponent Heuristic
        if heuristic == 'BTO':
            return len(own_liberties) - (2 * len(opp_liberties)) + len([action for action in own_liberties if action in opp_liberties])

        # Offensive to Defensive Heuristic
        elif heuristic == 'OTD':
            m = state.ply_count / 99
            if m > 0.5:
                return (2 * len(own_liberties)) - len(opp_liberties)
            elif m <= 0.5:
                return len(own_liberties) - (2 * len(opp_liberties))

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        import random
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            i = 0
            while i <= 12:
                self.queue.put(self.alpha_beta_pruning(
                    state, depth=i, heuristic='BTO'))
                i += 1
