# Q-learning algorithm for Chess Endgames

Following [this](./Bachelor_s_Project_AI_Boar.pdf) work, here we try to implement a Q-learning algorithm to solve classical endgames of chess. Endgames with 7 pieces or less in total are already solved, but to know each best move a lot of memory is necessary, thus our work will be in exploiting a Reinforcement Learning algorithm that should reach almost the same results using less memory. 

An endgame is essentially the final stage of a Chess game and it can result either in a Win, a Draw or a Loss for either player. Since we are in a zero-sum game, a Win for a player results in a Loss for the other. For simplicity, in this project we only consider the point of view of the White player, considering the Black one part of the environment and letting him use the best possible move. This is possible since we consider endgames that are theoretically won for the White player, so even if the Black one plays the best moves, he will either draw or loose. The starting position of the endgame can vary, what defines an endgame is just which pieces are on the chessboard. As an example, consider the below image comprising the King and the Rook for the White player and the King for the Black player (KRvK). 

![KRvK](KRvK.png)

## Q-learning

Q-learning is a model-free off-policy RL algorithm. This means that we don't assume a model of the environment (learns value function independently) and it uses an $\epsilon$-greedy approach for the selection of the next move. 

At any time step $t$ the agent is in state $s_t$ and has to choose an action $a_t$, therefore the action-value function is represented as $Q(s_t,a_t)$.

```math
Q^{\text{new}}(s,a) = \underbrace{Q(s_t,a_t)}_{\text{old value}} + \underbrace{\alpha}_{\text{learning rate}} \cdot \overbrace{(\underbrace{\underbrace{r_t}_{\text{reward}} + \underbrace{\gamma}_{\text{discount factor}}\cdot \underbrace{\max_{a}Q(s_{t+1},a)}_{\text{est. future value}}}_{\text{new value}} - \underbrace{Q(s_t,a_t)}_{\text{old value}})}^{\text{temporal difference}}
```

>**NOTE**
>The hyperparameters are a tuple ($\alpha$, $\gamma$, $\epsilon$) which in literature is being studied and has been found an optimal set of value that we will use (at least initially) and is (0.05, 0.95, 0.003).

Two important considerations need to be done before proceding:

- chess is a two-players game, hence the players alternate the moves ($A_w,A_b,A_w,A_b,\dots$)
- the agent (white in this case) needs to assume that the opponent will make the best move he can, otherwise it is not learning correctly

For the second point we already mentioned that we will let the Black player play the best possible move (taken from the oracle), while for the first point we have to introduce the MINIMAX concept.
It essentially tells that the estimate of future value in the Q-learning equation changes this way:

```math
\underbrace{\min_{b}(\overbrace{\max_{a}(Q(\underbrace{s_{t+2}}_{\text{(a)}}, \underbrace{a_b}_{\text{(b)}}))}^{\text{best move after opponent move}})}_{\text{least best move after opponent's possible moves}}
```

(a) - State after both agent and opponent move
(b) - Agent's moves after opponent moves

This way it finds the best move it knows it can do after each of the opponent's moves, and then choosing the minimal Q estimation among them, signifying the opponend playing optimally.

### Reward Function

For this project we want to use simple yet (probably) effective and intuitive reward function. Considering that the White player cannot loose in the subset of endgames we chose (Black does not have pieces to win), the reward should be just for WIN and for DRAW. We decided to keep their values equal in absolute terms but opposite in sign. Moreover, we decided to include a step penalty to encourage the agent to checkmate the fastest possible. Thus, mathematically speaking, the reward function is this:

```math
\begin{align*}
    +10000 & \text{ for WIN}\\
    -10000 & \text{ for DRAW}\\
    -1 & \text{ otherwise}
\end{align*}
```

