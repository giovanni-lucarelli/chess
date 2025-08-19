# Chess Endgame Solver

The goal of this project is to study learning optimal play in deterministic, perfect-information chess endgames by comparing different RL approaches:

* value iteration and policy iteration (model based algorithm)
* Q-learning (value based model free algorithm)
* REINFORCE (policy based model free algorithm)
* UCT-style Monte-Carlo Tree Search (with and without learned priors) 

on a fixed family of endgames (e.g., King and queen vs king, King and Rook vs King). We model each endgame as a finite discounted MDP with legal chess positions as states and legal moves as actions. 

Formally, two-player chess endgames are deterministic, perfect-information Markov games.
In our experiments, we model from the perspective of one player and incorporate the opponent’s moves into the transition dynamics, yielding a finite deterministic MDP with augmented state (`board`, `side-to-move`).

Agents train via self-play from a fixed start-state distribution; evaluation uses ground-truth labels from endgame tablebases (distance-to-mate, DTM). 

**Assessment:**
We compare methods on optimality gap ($\Delta$ DTM), and computational cost under fixed budgets, with statistical confidence intervals across random seeds.

## MDP formalization

While chess is inherently a two-player zero-sum **Markov game**, in this work we model it from the perspective of the White player only.
The opponent’s moves are treated as part of the environment dynamics, and the state includes the side-to-move flag.
This yields a finite, deterministic **Markov Decision Process** (MDP) with:

* **States**: all legal KQK (or KRK) positions, augmented with side-to-move.
* **Terminal State**: chessmate, once reached the game ends.
* **Actions**: legal moves for the current player.
* **Transition function**: deterministic update given current state and chosen action, followed by the opponent’s deterministic or stochastic reply.
* **Rewards**: +1 for win, −1 for loss, 0 otherwise.


## Markov Game formalization

A chess game can be formalized as a finite, deterministic, turn-based, zero-sum **Markov game** (or **stochastic game**). From [1], as such it consists of:
- set of agents $I=\{W,B\}$
- finite set of states $\mathcal{S}$, with subset of terminal states $\bar{\mathcal{S}}\sube\mathcal{S}$
- for each agent $i\in I$:
    - finite set of actions $\mathcal{A}_i$, in general $\mathcal{A}_i(s)$ where $s\in\mathcal{S}$
    - reward function $R_i:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\to\mathbb{R}$, where $\mathcal{A}=\mathcal{A}_W\times\mathcal{A}_B$ and such that $R_W=-R_B$ (zero-sum game). Hence we only need one, namely $R_W=R$
- state transition probability function $P:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\to[0,1]$
- initial state distribution $\mu:\mathcal{S}\to[0,1]$ such that 
$\sum_{s\in\mathcal{S}}\mu(s)=1$ and $\forall s \in\bar{\mathcal{S}}:\mu(s)=0$

### More Details
  
- $\mathcal{S}$ is the state space, every legally reachable board position plus: side-to-move, castling rights, en-passant square, half-move clock, full-move number. (All that information is needed to determine future legality.) Estimated size $|\mathcal{S}|=4.8 \times 10^{44}$ [(Tromp & Österlund 2022)](https://github.com/tromp/ChessPositionRanking)

    > **Note**: the number of paths through that state space is given by the *Shannon number* $10^{120}$

- $\mathcal{A}_W(s), \mathcal{A}_B(s)$ are the sets of legal moves for White or Black in state $s$. Only one set is non-empty at each turn. (up to $218$ in rare positions; avg around $30–40$).

- $P(s' \mid s, a)$: deterministic rule table of chess, yielding the unique successor position. 

    > Formally $P(s' \mid s, a)$ is a tensor. Usual chess engines do **not** store the entire transition function $P(s' \mid s, a)$ as a giant tensor, since it is infeasable:
    >* $|S| \approx 10^{44}$ legal positions
    >* $|A| \approx 10^2$ moves per position
    >
    >$\Rightarrow$ $|S| \times |A| \times |S|$ would be a tensor with more than $10^{90}$ entries
    >
    >Instead of storing $\mathcal{P}$, chess engines implement functions that *define* $\mathcal{P}$ procedurally:
    >* `legal_moves(s)` — generates valid actions in state $s$
    >* `apply_move(s, a)` — returns the next state $s'$
    >* `is_terminal(s')` — checks if game is over


- $R(s,a,s')=R_W(s,a,s') = -R_B(s,a,s')$: +1/−1/0 only when a terminal position is reached (White win, Black win, draw).


## References

- [1] Albrecht, S. V., Christianos, F., Schäfer, L. (2024). Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. United Kingdom: MIT Press.
