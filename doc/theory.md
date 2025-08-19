# Introduction to Chess Programing 

![Kasparov vs Deep Blue](/docs/garry-kasparov-deep-blue-ibm.jpg)
(*Garry Kasparov* versus Deep Blue (IBM) - Philadelphia, 1996)

>*Chess has been used as a Rosetta Stone of both human and machine cognition for over a century.* - Garry Kasparov

## About the game of chess

From [Chess Programming](https://www.chessprogramming.org/Chess):

>Chess is a two-player zero-sum abstract strategy board game with perfect information as classified by John von Neumann. Chess has an estimated state-space complexity of $10^{46}$, the estimated game tree complexity of $10^{123}$ is based on an average branching factor of $35$ and an average game length of $80$ ply (*i.e.* half-move).

## Brief Chess Computer History
The idea of teaching a machine to play games has been around for a long time and chess has been a popular choice as for the game to master.

It is interesting to remark that even the first game-playing machine (built around 1890 by the Spanish engineer Leonardo Torres y Quevedo) specialized in the “KRK” (king and rook versus king) chess endgame, guaranteeing a win when the side with the rook has the move.

However, it was only in 1950 that Claude Shannon published the first article about programming a computer for playing chess. It laid out all the major ideas: a representation for board positions, an evaluation function,  quiescence search, and some ideas for selective game-tree search.

Many algorithms have been subsequently developed involving game-tree search, from alpha-beta search to expectiminimax to Monte Carlo tree search.

As for the specific chess milestones, many programs have reached the master status: starting from **Belle** in 1982 to **Deep Thought** in 1990 to **Deep Blue** which was the first computer program to beat the current world champion in 1997. Deep Blue ran *alpha–beta search* at over 100 million positions per second, and could generate singular extensions to occasionally reach a depth of 40 ply.

> **Remark**: a ply is one turn taken by one of the players. The word is used to clarify what is meant when one might otherwise say "turn". For example, in standard chess terminology, one move consists of a turn by each player; therefore a ply in chess is a half-move.

The top chess programs today (e.g., Stockfish, Komodo, Houdini) far exceed any human player. They ran alpha-beta search with some additional pruning techniques like the null move heuristic, which generates a good lower bound on the value of a position, using a shallow search in which the opponent gets to move twice at the beginning, or the futility pruning, which helps decide in advance which moves will cause a beta cutoff in the successor nodes.

In parallel, Bellman in 1965 developed the idea of retrograde analysis for computing endgame tables. Using this idea, Thompson and Stiller in the nineties solved all chess endgames with up to five pieces.

In 2012 Makhnychev and Zakharov compiled the **Lomonosov Endgame Tablebase**, which solved all endgame positions with up to seven pieces—some require over 500 moves without a capture. The 7-piece table consumes 140 terabytes; an 8-piece table would be 100  times larger.

In 2017, **Alphazero** defeated STOCKFISH (the 2017 TCEC computer chess champion) in a 1000-game trial, with 155 wins and 6 losses. Additional matches also resulted in decisive wins for ALPHAZERO, even when it was given only 1/10th the time allotted to STOCKFISH.

In 2018, ALPHAZERO defeated top programs in chess and shogi, learning through self-play without any expert human knowledge and without access to any past games. (It does, of course, rely on humans to define the basic architecture as Monte Carlo tree search with deep neural networks and reinforcement learning, and to encode the rules of the game.)

In 2018, a free open search chess engine **Leela Chess Zero** was also developed based on Alphazero.

Many competitions are being held between chess engines: though computers have definetely surpassed human ability, the game is not solved and further strategic improvements are still possible.

## References

- [1] Russel, S., Norvig, P. (2021). Artificial Intelligence A Modern Approach. Pearson Education.