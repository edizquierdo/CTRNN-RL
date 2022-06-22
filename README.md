# Dynamic Recurrent Neural Network trained using Reward-Modulated Learning

This is an implementation of reward-modulated learning implemented on continuous-time recurrent neural networks, following previous work by Wei and Webb "<a href="https://www.sciencedirect.com/science/article/pii/S0893608018302260">A model of operant learning based on chaotically varying synaptic strength</a>" (2018).

The main part of the code is contained within the CTRNN class. The rest of the classes are use to evolve the neural network to produce oscillations.

We are using this to better understand biologically-inspired forms of reinforcement learning for dynamical neural networks.

This is work in collaboration with colleague, Dr. Jason Yoder. A first part of this work was published (see ref below). We are continuing to work on this project.

Yoder J, Cooper A, Cehong W, Izquierdo EJ (2022) <a href="https://www.frontiersin.org/articles/10.3389/fncom.2022.818985/full">Reinforcement learning for central pattern generation in dynamical recurrent neural networks</a>. Frontiers in Computational Neuroscience.  doi: 10.3389/fncom.2022.818985

## Instructions for use

1. Compile using the Makefile:
```
$ make
```
2. Perform an evolutionary run to produce neural circuit that can oscillate some (but not perfectly). Then test reward-modulating that circuit 10 times (takes ~6 seconds):
```
$ ./main
```
3. Visualize the evolutionary progress and the resulting dynamics of the evolved neural circuit improving during its lifetime with rewards:
```
$ Mathematica viz.nb
```
