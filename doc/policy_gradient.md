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
- reward $r$ is either $+1,-1,0$

Recalling that the REINFORCE update rule is

```math
\Delta\theta_t = \alpha\nabla_{\theta}\log\pi_{\theta}(s_t,a_t)v_t
```

where $v_t$ is the return from time step $t$, we need to implement the **self-play** concept. 
Self-play RL is a paradigm where an agent learns by playing games against copies of itself, rather than against fixed opponents or human players. The fundamental insight is that the environment itself becomes adaptive as the agent improves. Each agent is essentially solving

>**Remember** that $v_t$ is an unbiased sample of $Q^{\pi_{\theta}}(s_t,a_t)$

```math
\max_{\theta}J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta},\pi_{-\theta}}\left[R(\tau)\right]
```
where $J(\theta)$ represents our objective function and $\tau$ is a trajectory generated by the current policy playing against the opponents. 

The polict gradient becomes:

```math
\nabla_{\theta}J(\theta) = \sum_s d(s) \sum_a \pi_{\theta}(s,a)\nabla_{\theta}\log\pi_{\theta}(s_t,a_t)G_t
```
where $G_t$ is the return from the outcome of the self-play game.

Differences from standard REINFORCE:

- **Non-Stationary Environment** - in self-play the expected reward depends on the opponent's policy, which is constantly evolving
- **Game-Theoretic Objective** - instead of maximizing expected return against a fixed environment, we're seeking:
```math
\pi^* = \arg \max_{\pi} \min_{\pi^{'}}\mathbb{E}_{\tau \sim \pi,\pi^{'}}\left[R(\tau)\right]
```
- **Symmetric Learning** - both plauers use the same update rule simultaneusly:
```math
\theta_1^{(t+1)} = \theta_1^{(t)} + \alpha\nabla_{\theta_1}\log\pi_{\theta_1}(a_1|s)\cdot R_1 
```

```math
\theta_2^{(t+1)} = \theta_2^{(t)} + \alpha\nabla_{\theta_2}\log\pi_{\theta_2}(a_1|s)\cdot R_2
```
where $R_1 = -R_2$ in zero-sum games.

```latex
function SELF_PLAY_REINFORCE:
    Initialize θ randomly
    for episode = 1 to N:
        # Generate self-play game
        τ = generate_trajectory(π_θ, π_θ)  # Both players use same policy
        
        # Compute returns for both players
        for t = 0 to |τ|-1:
            G_t = Σ_{k=t}^{|τ|-1} γ^{k-t} r_{k+1}
            
        # Update policy based on player 1's perspective
        for t = 0 to |τ|-1:
            if player_turn[t] == 1:
                θ ← θ + α ∇_θ log π_θ(a_t|s_t) · G_t
            else:  # player 2's turn
                θ ← θ + α ∇_θ log π_θ(a_t|s_t) · (-G_t)  # opponent's reward
```

> **IMPORTANT**: 
> For the internal state representation, we will use here a neural network, since hand-crafted features are inefficient and hard to obtain.

Another important concept that we have implemented in our REINFORCE algorithm is bias reduction using a **baseline**.
Essentially we are subtracting a baseline function $B(s)$ from the return:
```math
\Delta\theta_t = \alpha\nabla_{\theta}\log\pi_{\theta}(s_t,a_t)\cdot\underbrace{(v_t - B(s_t))}_{\text{advantage function }A^{\pi_{\theta}}(s,a)}
```
The idea here is that *any baseline that depends only on the state (not the action) leaves the expectation unchanged*:
```math
\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta}\log\pi_{\theta}(s,a)\cdot B(s)\right] = 0
```
The optimal baseline is the state value function:
```math
B(s) = V^{\pi_{\theta}}(s)
```

Intuitively,

- $A(s,a) > 0$: action is better than average $\to$ increase probability
- $A(s,a) < 0$: action worse than average $\to$ decrease probability
- $A(s,a) = 0$: action is average $\to$ no update needed

This leads to the **Actor-Critic** framework:
```math
\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta}\log\pi_{\theta}(s,a)\cdot A^{\pi_{\theta}}(s,a)\right]
```

where we learn:

- **Actor**: $\pi_{\theta}(a|s)$
- **Critic**: $V_{w}(s) \approx V^{\pi_{\theta}}(s)$

> **IMPORTANT**
>Finally, a **batch training** has been applied.