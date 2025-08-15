# Policy Gradient in Chess

In this document, we'll understand how to use policy gradient algorithms, specifically REINFORCE, in a chess game. This will define the theory later implemented in this project. 

Recalling that the **objective** of this project is to study learning optimal play in a deterministic, perfect-information chess endgame, we train agents via self-play and evaluate performance using DTM and computational cost. 

### RL theory

We enter here in the subset of RL called **Policy-Based RL**, where we do not consider value functions but instead we try to approximate a policy function with parameters. 

$$
\pi_{\theta}(s,a) = \mathbb{P}\left[a | s,\theta\right]
$$

We focus on **model-free RL**.

Before, a policy was generated from the value function, which was approximated using parameters $w$.

- **Value-Based**: 
    
    - Learn Value Function (either model-based or model-free methods)
    - Implicit policy from that value function (e.g. $\epsilon$-greedy)
- **Policy-Based**:

    - No Value Function
    - Learn Policy
- **Actor-Critic** (possible implementation later):

    - Learn Value Function
    - Learn Policy
> **Note**
>Advantages of Policy-Based RL:
>
>- Better convergence properties
>- Effective in high-dimensional or continuous action spaces
>- Can learn stochastic policies
>
>Disadvantages: 
>
>- Typically converge to a local optimum
>- Usually inefficient and with high variance

How do we measure the quality of a policy?
There are many methods, but usually an average reward per time-step is used:

$$
J_{avR}(\theta) = \sum_s d^{\pi_{\theta}}(s)\sum_a \pi_{\theta}(s,a)\mathcal{R}_s^a
$$

where $d^{\pi_{\theta}}$ is a **stationary distribution** of Markov chain for $\pi_{\theta}$.

This is an **optimisation** problem, meaning that we have to find the best parameters $\theta$ that maximise the objective function $J(\theta)$. There are many algorithms for this, but we will focus on SGD.

$$
\Delta \theta = \overbrace{\alpha}^{\text{step-size param}} \cdot \underbrace{\nabla_{\theta}J(\theta)}_{\text{policy gradient}}
$$

>**Important**
>To compute policy gradient analytically (so without using **Finite Differences**), we assume policy $\pi_{\theta}$ is differentiable whenever >it is non-zero an we know the gradient $\Delta_{\theta}\pi_{\theta}(s,a)$. >It is useful to exploit the following identity for later use:
>$$\nabla{\theta}\pi_{\theta}(s,a) = \pi_{\theta}\frac{\nabla{\theta}\pi_{\theta}(s,a)}{\pi_{\theta}(s,a)} = \pi_{\theta}(s,a)\underbrace{\nabla_{\theta}\log\pi_{\theta}(s,a)}_{\text{score function}}$$

Now one has to choose how to represent features and how to weight them, then choose the distribution for the policy (can be Softmax, Gaussian, ...).

Considering as example a one-step MDP:

```math
J(\theta) = \mathbb{E}_{\pi_{\theta}}\left[r\right] = \sum_s d(s) \sum_a \pi_{\theta}(s,a)\mathcal{R}_{s,a}
```

```math
\nabla_{\theta}J(\theta) = \sum_s d(s) \sum_a \pi_{\theta}(s,a)\nabla_{\theta}\log\pi_{\theta}(s,a)\mathcal{R}_{s,a} = \mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta}\log\pi_{\theta}(s,a)r\right]
```

For multi-step MDP instead:

```math
\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta}\log\pi_{\theta}(s,a)Q^{\pi_{\theta}}(s,a)\right] \quad \text{(replace R with Q)}
```

##### REINFORCE algorithm

- Update parameters by SGA
- Using policy gradient theorem
- Using return $v_t$ as an unbiased sample of $Q^{\pi_{\theta}}(s_t,a_t)$

$$
\Delta\theta_t = \alpha\nabla_{\theta}\log\pi_{\theta}(s_t,a_t)v_t
$$

```
REINFORCE
    Initialise parameters arbitrarily
    FOR each episode DO
        FOR each step DO
            update parameters
        END for
    END for
    return parameters
END 
```

>**To do**
>exploit actor-critic algorithms to improve performance

### How to apply theory to chess game

For the formalization of the chess game look [here](./project.md).

> **Note**
>Using a policy-gradient algorithm is enough for this work, meaning we >don't have to implement a minimax search since the opponen move is already >included in the environment and the policy is improved by **maximizing** >the best action rewards for the player taken into consideration and >**minimizing** the action rewards for the opponent. The policy-based >algorithm already does this when learning the best policy. 

- state $s$ is reprented by the *Game* class
- action $a$ is represented by the *Move* class
- reward $r$ is wither $+1,-1,0$

```math
\nabla_{\theta}J(\theta) = \sum_s d(s) \sum_a \pi_{\theta}(s,a)\nabla_{\theta}\log\pi_{\theta}(s_t,a_t)v_t
```
>**Remember** that $v_t$ is an unbiased sample of $Q^{\pi_{\theta}}(s_t,a_t)$

Practically speaking, starting from an endgame position the reasoning of the algorithm is:

- I am in this state with this features defining the state
- I can do some actions, which lead to a reward (0 for most cases) and bring me to another state $s'$
- So I will use a Monte Carlo method to sample trajectories
- For each trajectory I will update parameters of the policy function at each step
- I will repeat this procedure until convergence

The reset_from_fen function at src/game.cpp:547 is used to initialize a
chess game from a FEN (Forsyth-Edwards Notation) string, which is the
standard notation for describing a chess position.

Here's what the function does:

1. Clears the current game state - Resets the board, check flags, and
checkmate status
2. Parses the FEN string into 6 fields:
- placement: Piece positions on the board
- stm: Side to move (w/b)
- castling: Castling rights (KQkq)
- ep: En passant target square
- halfmove: Halfmove clock (ignored)
- fullmove: Fullmove number (ignored)
3. Sets up piece placement - Processes the placement string rank by rank
(8→1), handling:
- / as rank separators
- Numbers as empty squares count
- Letters as pieces (uppercase=white, lowercase=black)
4. Sets side to move - "w"/"W" for white, otherwise black
5. Sets castling rights - Parses KQkq characters for white/black
kingside/queenside rights
6. Sets en passant square - Parses algebraic notation or sets to
NO_SQUARE if "-"
7. Updates check status - Calls update_check() to determine if either
king is in check

The function is used throughout the codebase to reset games to specific
positions, as seen in the notebooks and test code that use the standard
starting position FEN: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w
KQkq - 0 1"

> **IMPORTANT**: 
> For the internal state representation, we will use here a neural network, since hand-crafted features are inefficient and hard to obtain.