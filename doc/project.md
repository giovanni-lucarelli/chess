# Chess Endgame Solver

The goal of this project is to study learning optimal play in deterministic, perfect-information chess endgames by comparing three RL approaches:

* UCT-style Monte-Carlo Tree Search (with and without learned priors), 
* REINFORCE, 
* Q-learning

on a fixed family of endgames (e.g., King and queen vs king, King and Rook vs King). We model each endgame as a finite discounted MDP with legal chess positions as states and legal moves as actions. 

Formally, two-player chess endgames are deterministic, perfect-information Markov games.
In our experiments, we model from the perspective of one player and incorporate the opponent’s moves into the transition dynamics, yielding a finite deterministic MDP with augmented state (`board`, `side-to-move`).

Agents train via self-play from a fixed start-state distribution; evaluation uses ground-truth labels from endgame tablebases (distance-to-mate, DTM). 

**Assessment:**
We compare methods on optimality gap (ΔDTM), and computational cost under matched budgets, with statistical confidence intervals across random seeds.

## Problem statement

Learning optimal play in small chess endgames from a *distribution* of legal positions.

Formally: for each endgame $E \in \{\text{KQK},\text{KRK}\}$, define a finite episodic MDP and learn a policy $\pi$ that **minimizes expected mate length** (or equivalently maximizes return under a sparse terminal reward). Concretely:

* Objective (DTM version): $\min_\pi \; \mathbb{E}_{s_0 \sim d_0}[\text{DTM}_\pi(s_0)]$.
* Objective (RL reward version): terminal $+1$ for mating, $0$ otherwise, optional small step penalty $-\lambda$; maximize $\mathbb{E}_\pi[\sum_t \gamma^t r_t]$.

You then **compare three methods** (UCT-MCTS, REINFORCE, Q-learning) on how close they get to optimal play, how much data/compute they need, and how they behave theoretically.

## MDP formalization


While chess is inherently a two-player zero-sum **Markov game**, in this work we model it from the perspective of the White player only.
The opponent’s moves are treated as part of the environment dynamics, and the state includes the side-to-move flag.
This yields a finite, deterministic **Markov Decision Process** (MDP) with:

* **States**: all legal KQK (or KRK) positions, augmented with side-to-move.
* **Actions**: legal moves for the current player.
* **Transition function**: deterministic update given current state and chosen action, followed by the opponent’s deterministic or stochastic reply.
* **Rewards**: +1 for win, −1 for loss, 0 otherwise.

