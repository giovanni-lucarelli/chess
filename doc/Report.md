# Chess Endgame Solver

The **goal** of this project is to study **learning optimal play** in deterministic, perfect-information chess endgames by comparing different RL approaches on one type of endgame: King and Rook vs King.

Formally, two-player chess endgames are deterministic, perfect-information **Markov games**.
In our experiments, we model from the perspective of one player and incorporate the opponent’s moves into the transition dynamics, yielding a **finite deterministic MDP**. The state includes both the positions of the pieces on the board and the side to move (the side to move is necessary because all checkmate terminal states have black as the side to move).

We have decided to limit the problem to chess endgames to have a smaller finite state set: in this setting it is possible to form approximations of value functions using tables with one entry for each state (or state–action pair). As written in Sutton and Barto book[1], this is called the **tabular case**, and the corresponding methods **tabular solution methods**.

Furthermore, chess endgames have already been solved and thus we have availability online of **endgame tablebases** that contain the optimal move at each possible game state. The availability of ground-truth solutions assures a more precise and complete evaluation of our algorithms through the measuring of the optimality gap ($\Delta$ DTM).

It is important to note that in many cases of practical interest, however, there are far more states than could possibly be entries in a table. In these cases the functions must be approximated, using some sort of more compact parameterized function representation. This is precisely why we decided to approach the problem using also **approximate solution methods**. The original idea was in fact to develop algorithms that could eventually learn policies to play chess games with up to 5 pieces.

## Markov Game formalization

A chess game can be formalized as a finite, deterministic, turn-based, zero-sum **Markov game** (or **stochastic game**). From [2], as such it consists of:
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

## MDP formalization

While chess is inherently a two-player zero-sum **Markov game**, in this work we model it from the perspective of the White player only.
The opponent’s moves are treated as part of the environment dynamics, and the state includes the side-to-move flag.
This yields a finite, deterministic **Markov Decision Process** (MDP) with:

* **States**: all legal KRK positions, augmented with side-to-move. We don't keep track of castling rights or en-passant square (useless at this point of the game) and also half-move clock or full-move number (since we don't consider 50-move draws).
* **Terminal States**: checkmate or stalemate or insufficient pieces, once we reach one of these states the game ends. We do not consider 50-move draw or draw by 3 repeated moves.
* **Actions**: legal moves for the current player.
* **Transition function**: deterministic update given current state and chosen action, followed by the opponent’s deterministic reply.
* **Rewards in classic chess game formalization**: +1 for win, −1 for loss, 0 for draw.
* **Rewards in our formulation**: +1 for loss, -0.01 as step penalty, −1 for draw.

> Remark: All variables in the problem are **discrete**, posing well for a tabular approach.

## Algorithms motivations and results analysis
We tried many algorithms to solve the problem: each has its own pros and cons and was motivated by different reasons.

In particular, we focused on:
* Value Iteration (model based algorithm)
* Q-learning (model free algorithm)
* Actor-Critic (full reinforcement learning algorithm)
* UCT-style Monte-Carlo Tree Search 

### Value Iteration

Value Iteration is a **Dynamic Programming** (DP) algorithm that assure us to find the optimal value function.
As a Dynamic Programming algorithm, it updates estimates of the values of states based on estimates of the values of successor states. That is, it updates estimates on the basis of other estimates. We call this general idea **bootstrapping**.

A major drawback to the DP methods is that they require sweeps of the entire state set. If the state set is very large, then even a single sweep can be prohibitively expensive. We also have to consider that in the worst case, the time that DP methods take to find an optimal policy is **polynomial** in the number of states and actions.

In our specific case the number of states is:
- **182.676** states (including terminal states, 175.168 excluding terminal states) and 3.383.416 state-action pairs (excluding terminal states since there are no possible actions) in the KRvK endgame.

Since, in practice, DP methods can be used with today’s computers to solve MDPs with **millions** of states, DP is a **feasible** solution to our specific problem.

#### Methodology

For this specific algorithm we used -2 as the step penalty, 1 as the checkmate reward and 100 as the draw penalty. This huge draw penalty is used to balance the step penalty: in this way the algorithm will always learn to prefer a long mate rather than a quick draw. Also, since from all states it is possible to win, the final values will exactly correspond to the number of moves to mate (the draw penalty value will never be chosen since it's much lower than the longest mate: 16 turns).

Since the MDP contains terminal states (also called coffin states) we can safely use discount factor $\gamma$=1.

A difficulty we had when implementing this method was finding all possible legal states, in total we had to consider:
- All board positions with the 3 pieces such that it's white turn, but black is not under checkmate (non-terminal states)
- All board positions with 3 pieces such that it's black turn and black is either under checkmate or it can't move (terminal states)
- All board positions with 2 pieces such that it's white turn (terminal states)

We only realized after (when working with Q learning) that we could have avoided saving values for terminal states since they are all going to be 0.

#### Results

The Value Iteration algorithm applied to the MDP against optimal player **converged to the optimal policy**: specifically, it converged after **16** iterations. This comes as no surprise since the longest possible mate with perfect play is exactly 16 turns. At each $i$-th iteration it succesfully found all checkmates up to $i$-th turns.

Both policy iteration and value iteration are widely used, and it is not clear which, if either, is better in general. For this reason, when Value Iteration solved our problem we decided to not implement policy iteration, which would have probably converged to the same optimal result.

One main **disadvantage** of this algorithm is the cost of **memory**: all states must be saved and examined, which would be unfeasible for games with more pieces.

Addionally, this method converges to the optimal policy (it always wins) because we can use the optimal policy as the policy of the black player. We then experimented to see how would the policy learned by value iteration perform against the ideal player if it was instead computed assuming random moves from the opponent.

### TD-Control: Q-Learning

In the previous section we considered transitions from state to state and learned the values of states. Now we consider transitions from state–action pair to state–action pair, and learn the values of state–action pairs. This method can be called one-step, tabular, model-free method.

We decided to try a model-free approach to see how good of a solution would get an algorithm trained with less information. For training, multiple states are used as starting points so that the algorithm can more surely explore the whole space.

We also want to mention that we found online references of other studies using this method to solve chess endgames. So, even if value iteration already converges to the optimal policy we deemed interesting to compare the two solutions.

In Q-Learning, the learned action-value function, Q, directly approximates $q_*$, the optimal action-value function, independent of the policy being followed (off-policy method). Since we don't care about the risks accumulated during training we found no reason to train SARSA on the problem (though for completeness we implemented a flag to activate it in the TD-control code).

We kept the algorithm one-step (i.e., we use no eligibility trace) to keep it simpler and computationally less demanding.

#### Methodology

For episodic algorithms such as Q learning, we had to implement a *max_step* value that would truncate the episode after too many steps.

The initial idea was to make the episode reach its end and keep the step penalty: unfortunately, with episodes taking up to 20 minutes we realized it was unfeasible on our devices. We thus decided to truncate the episodes once it reached 50 steps (we know all states can be won with up to 16 turns against optimal play).

We thought of taking out the step penalty so that the truncated episodes wouldn't affect too much the value of the last action. However, this made the algorithm effectively not learn anything since many episodes ended because of *max_step* and this brought with no updates to the state-action function (since the reward was always 0 until the terminal state). We thus reintroduced a -0.01 step penalty: in fact, the objective is not only to learn to mate but also to obtain the fastest mate.

We also introduced a discount factor $\gamma$=0.99: this was introduced when we took out the step penalty to assure that quick mates would be preferred to slower ones and in the end it was kept even when we reintroduced back the step penalty. The episodes are quite short anyway so it doesn't affect much the estimates.

As for the parameters $\epsilon$ and $\alpha$ we chose to apply an epsilon decay so that the moves would become increasingly less randomized and a constant learning rate $\alpha$.

We wanted to change to a decaying $\alpha$ but we didn't manage to train the algorithm again. A costant learning rate unfortunately doesn't assure convergence (as asked by the Robbins Monro conditions), but we chose it small ($\alpha$=0.05) so that the variance would at least be small (this of course required an high number of steps).

#### Result

After 1 million iterations, the algorithm start to converge. To check for convergence we plotted the return for each episode: it slowly started to stabilize. It is possible that with more training time it would have actually converged to optimum.

When tested against the optimal player we found that it managed to win in many states, though often with more steps than necessary. We didn't have the time to test it over all the state space, but it seems that the the mate-in-1 positions are correctly won, while the more steps are required to win the less probable it is victory.

Overall, there aren't really may reasons why one would prefer Q learning over Value iteration: we still have the disadvantage of memory while obtaining a worse performance.

### Actor Critic

Actor critic learns a parameterized policy that can select actions without consulting a value function. A value function is still used to learn the policy parameters, but is not required for action selection.

The reason we tried this method is to expand our experiments: since all the methods used so far are heavy in memory costs we wanted an approximate solution method.

Initially we thought of the Reinforce algorithm, but since the episodes can last so much we noticed quickly that the variance due to Montecarlo was too high. We thus moved to the actor-critic method.

Since we have a priori knowledge and we know the general structure of the endgame we can manually construct an approximation of the value function. We have tried different ways of approximating (first with a tabular approximation and finally with a linear combination) based on handpicked features.

Specifically, we use various normalized metrics of distance between the pieces, the distance of the black king from the border and side and some additional features of the board (like black king on light/dark tile). More about it in the **results.ipynb**.

As for the policy function, since the policy space is quite complex (thousand of possible actions, but only a few legal for each state) we decided to use a simple neural network.

#### Methodology

Since it is an episodic algorithm we again have to introduce the variable *max_steps* to assure that episodes don't go on *ad infinitum*. We kept the step penalty to assure that those partial episodes would still contribute to the learning. Since the terminal state is not assured to be reached we also slowly lowered the discount factor to 0.99.

We are using an on-policy algorithm (otherwise we would fall in the deadly triad) and as for the learning rates we chose $\alpha_w$=0.02 and $\alpha_{\theta}$=0.003. We need the critic to change estimates for the weights of the approximation value function faster than the weights of the policy: this is because the estimates of the value have to guide the estimates of the policy.

The neural network that approximates the policy function is composed of 3 fully connected layers with ReLu activations functions and a final softmax to output the probabilities over all the actions.

We tried to train it over a million episodes but the algorithm was highly unstable and never converged.
For this reason, we tried to use Curriculum learning, as the episodes that terminated in a win were too sparse and we couldn't manage to reach convergence. We trained the algorithm only over episodes with distance-to-mate (DTM) of 1 until convergence and then over episodes of DTM 3.

#### Results

After many iterations, the algorithm seems to converge to a solution for those two DTM. However, it generalizes very badly to longer checkmates, making it not very useful for our problem. The only advantage is that it uses less memory as it doesn't save explicitely a value for each possible state, but only the weights of the neural networks (182.676 states vs 36.864 weights).

The strong difficulty we found to make the algorithm converge poses a strong limitation to this method, though the advantage of being able of learning a good value approximation makes this method still interesting. In fact, we examined the weights of each features and they were aligned with human knowledge: they incentivize small distances of the black king (BK) from any side, large Manhattan distance between WR and BK and small Chebyshev distance between WR and BK.

## References

- [1] Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction. Cambridge: MIT press.
- [2] Albrecht, S. V., Christianos, F., Schäfer, L. (2024). Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. United Kingdom: MIT Press.