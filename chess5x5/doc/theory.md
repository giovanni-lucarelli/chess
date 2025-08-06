# Introduction to Chess Programing 

![Kasparov vs Deep Blue](/docs/garry-kasparov-deep-blue-ibm.jpg)
(*Garry Kasparov* versus Deep Blue (IBM) - Philadelphia, 1996)

>*Chess has been used as a Rosetta Stone of both human and machine cognition for over a century.* - Garry Kasparov

## About the game of chess

From [Chess Programming](https://www.chessprogramming.org/Chess):

>Chess is a two-player zero-sum abstract strategy board game with perfect information as classified by John von Neumann. Chess has an estimated state-space complexity of $10^{46}$, the estimated game tree complexity of $10^{123}$ is based on an average branching factor of $35$ and an average game length of $80$ ply (*i.e.* half-move).

## Brief Chess Computer History

- Von-Neumann?
- Shannon?
- deep blue (IBM)
- stockfish
- alphazero (DeepMind)
- lela

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
