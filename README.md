This project is a reproduction of a series of PBD (Position Based Dynamics) papers, implemented using the Taichi library.
It features several solvers: the original PBD, a global PBD solver (which considers all constraints simultaneously—a method that is nearly infeasible in practice and was implemented for experimental purposes), and an XPBD solver.

For PBD, I have implemented constraints for point-point distance, point-triangle collision, and bending. 
For XPBD, the point-point distance and bending constraints are implemented.

这是一个基于taichi库复现的pbd系列论文的项目。
复现了多种求解器，有原始pbd、全局pbd（同时考虑所有constraint，这在实践中几乎不可行，我自己搞着玩的）、以及xpbd的求解器。

对于pbd，我实现了两点间距离、点与三角面碰撞、bending等constraint。
对于xpbd，实现了两点间距离、bending的constraint。
