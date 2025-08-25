# Chess Endgame Solver

The goal of this project is to study learning optimal play in deterministic, perfect-information chess endgames by comparing different RL approaches:

* value iteration and policy iteration (model based algorithm)
* Q-learning (value based model free algorithm)
* REINFORCE (policy based model free algorithm)
* UCT-style Monte-Carlo Tree Search (with and without learned priors) 

on a fixed family of endgames (e.g., King and queen vs king, King and Rook vs King). We model each endgame as a finite discounted MDP with legal chess positions as states and legal moves as actions. 

Formally, two-player chess endgames are deterministic, perfect-information Markov games.
In our experiments, we model from the perspective of one player and incorporate the opponent’s moves into the transition dynamics, yielding a finite deterministic MDP with augmented state (`board`, `side-to-move`).

We have decided to limit the problem to chess endgames to have a smaller finite state set: in this setting it is possible to form approximations of value functions using tables with one entry for each state (or state–action pair). As written in Sutton and Barto book, this is called the **tabular case**, and the corresponding methods tabular methods.

It is important to note that in many cases of practical interest, however, there are far more states than could possibly be entries in a table. In these cases the functions must be approximated, using some sort of more compact parameterized function representation. Reinforcement learning adds to MDPs a focus on approximation and incomplete information for realistically large problems.

Agents train against a perfect opponent from a fixed start-state distribution; evaluation uses ground-truth labels from endgame tablebases (distance-to-mate, DTM). 

**Assessment:**
We compare methods on optimality gap ($\Delta$ DTM), and computational cost under fixed budgets, with statistical confidence intervals across random seeds.

## MDP formalization

While chess is inherently a two-player zero-sum **Markov game**, in this work we model it from the perspective of the White player only.
The opponent’s moves are treated as part of the environment dynamics, and the state includes the side-to-move flag.
This yields a finite, deterministic **Markov Decision Process** (MDP) with:

* **States**: all legal KQK (or KRK) positions, augmented with side-to-move.
* **Terminal State**: checkmate or stalemate or insufficient pieces, once we reach one of these states the game ends.
* **Actions**: legal moves for the current player.
* **Transition function**: deterministic update given current state and chosen action, followed by the opponent’s deterministic reply.
* **Rewards**: +1 for win, −1 for loss, 0 otherwise.
* **Rewards**: -2 per ply, −1000 for draw


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

## Algorithms analysis

### Value Iteration

![alt text](ValueIteration.png)

Value Iteration is a **Dynamic Programming** (DP) algorithm that assure us to find the optimal value function.
As a Dynamic Programming algorithm, it updates estimates of the values of states based on estimates of the values of successor states. That is, it updates estimates on the basis of other estimates. We call this general idea **bootstrapping**.

A major drawback to the DP methods is that they involve operations over the entire state set of the MDP, that is, they require sweeps of the state set. If the state set is very large, then even a single sweep can be prohibitively expensive. DP is sometimes thought to be of limited applicability because of the **curse of dimensionality**, the fact that the number of states often grows exponentially with the number of state variables.

Large state sets do create difficulties, but these are inherent diffculties of the problem, not of DP as a solution method. In fact, DP is comparatively better suited to handling large state spaces than competing methods such as direct search and linear programming.

In our specific case the number of states is:
- 182676 states (including terminal states) and 3383416 state-action pairs (excluding terminal states since there are no possible actions) in the KRvK endgame
-  in the KQvK endgame
-  in the KBBvK endgame

In practice, DP methods can be used with today’s computers to solve MDPs with millions of states, which make it a feasible solution to our specific problem. If we ignore a few technical details, then, in the worst case, the time that DP methods take to find an optimal policy is **polynomial** in the number of states and actions.

Both policy iteration and value iteration are widely used, and it is not clear which, if either, is better in general. In practice, these methods usually converge much faster than their theoretical worst-case run times, particularly if they are started with good initial value functions or policies.


#### Possible alternative: Asynchronous Value Iteration
Asynchronous Value Iteration is an in-place iterative DP algorithm that is not organized in terms of systematic sweeps of the state set. This algorithm updates the values of states in any order whatsoever, using whatever values of other states happen to be available. The values of some states may be updated several times before the values of others are updated once. To converge correctly, however, an asynchronous algorithm must continue to update the values of all the states: it can’t ignore any state after some point in the computation.

Of course, avoiding sweeps does not necessarily mean that we can get away with less computation. It just means that an algorithm does not need to get locked into any hopelessly long sweep before it can make progress improving a policy.

We can try to order the updates to let value information propagate from state to state in an e cient way. Some states may not need their values updated as often as others. We might even try to skip updating some states entirely if they are not relevant to optimal behavior.

### TD-Control: Q-Learning and SARSA

![alt text](Sarsa.png)
![alt text](QLearning.png)

In the previous section we considered transitions from state to state and learned the values of states. Now we consider transitions from state–action pair to state–action pair, and learn the values of state–action pairs. This method can be called one-step, tabular, model-free method.

Obviously, TD methods have an advantage over DP methods in that they do not require a model of the environment, of its reward and next-state probability distributions. The next most obvious advantage of TD methods is that they are naturally implemented in an online, fully incremental fashion. With Monte Carlo methods, for example, one must wait until the end of an episode, because only then is the return known, whereas with TD methods one need wait only one time step.

We face the need to trade off exploration and exploitation, thus approaches fall into two main classes: on-policy (Sarsa) and off-policy (Q-Learning).

In Sarsa, the learned action-value function, Q, directly approximates $q_*$, the optimal action-value function, independent of the policy being followed.

In Q-Learning, the learned action-value function, Q, directly approximates $q_*$, the optimal action-value function, independent of the policy being followed. This dramatically simplifies the analysis of the algorithm and enabled early convergence proofs. The policy still has an effect in that it determines which state–action pairs are visited and updated. However, all that is required for correct convergence is that all pairs continue to be updated.

To generalize from different starting states we specifically sample different starting positions.

The **convergence properties** of the Sarsa algorithm depend on the nature of the policy’s dependence on Q. For example, one could use "-greedy" or "-soft" policies. Sarsa converges with probability 1 to an optimal policy and action-value function, under the usual conditions on the step sizes, as long as all state–action pairs are visited an infinite number of times and the policy converges in the limit to the greedy policy (which can be arranged, for example, with "-greedy" policies by setting  $\alpha= 1/t$).

#### Training improvement: prioritized sweeping
Simulated transitions are started in state–action pairs selected uniformly at random from all previously experienced pairs. But a uniform selection is usually not the best; planning can be much more efficient if simulated transitions and updates are focused on particular state–action pairs.

Consider how in chess the positive rewards come only from the terminal state: at first only an update along a transition into the goal will meaningfully change values. This example suggests that search might be usefully focused by working backward from goal states. In general, we want to work back not just from goal states but from any state whose value has changed.

Suppose now that the agent discovers a change in the environment and changes its estimated value of one state, either up or down. Typically, this will imply that the values of many other states should also be changed, but the only useful one-step updates are those of actions that lead directly into the one state whose value has been changed. If the values of these actions are updated, then the values of the predecessor states may change in turn. If so, then actions leading into them need to be updated, and then their predecessor states may have changed. In this way one can work backward from arbitrary states that have changed in value, either performing useful updates or terminating the propagation. This general idea might be termed backward focusing of planning computations.

A queue is maintained of every state–action pair whose estimated value would change nontrivially if updated , prioritized by the size of the change. When the top pair in the queue is updated, the e↵ect on each of its predecessor pairs is computed. If the e↵ect is greater than some small threshold, then the pair is inserted in the queue with the new priority.

#### Possible alternative: TD-control with afterstate value functions

A conventional state-value function evaluates states in which the agent has the option of selecting an action. For this reason we considered only the boards where the side-to-move is white, i.e., our agent and we assumed deterministic perfect black player.

An alternative, however, would have been to have the state-value function evaluate board positions after our agent has made its move. These are called **afterstates**, and the value functions over these are called **afterstate value functions**.

Afterstates are useful when we have knowledge of an initial part of the environment’s dynamics but not necessarily of the full dynamics. For example, in real life chess we typically know the immediate effects of our moves. We know for each possible chess move what the resulting position will be, but not how our opponent will reply. Afterstate value functions are a natural way to take advantage of this kind of knowledge and thereby produce a more effcient learning method.

Why could it be more efficient? A conventional action-value function would map from positions and moves to an estimate of the value. But many position—move pairs produce the same resulting position (example: "7R/8/7k/8/4K3/8/8/8 b - - 0 1" -> h6g6 -> "7R/8/6k1/8/4K3/8/8/8 w - - 0 1" and "7R/8/8/7k/4K3/8/8/8 b - - 0 1" -> h5g6 -> "7R/8/6k1/8/4K3/8/8/8 w - - 0 1"). In such cases the  position–move pairs are different but produce the same “afterposition,” and thus must have the same value. A conventional action-value function would have to separately assess both pairs, whereas an afterstate value function would immediately assess both equally.

## References

- [1] Albrecht, S. V., Christianos, F., Schäfer, L. (2024). Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. United Kingdom: MIT Press.
