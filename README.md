Gym Environment for Dyadic Slider Task


The task includes two agents that apply forces on an object to follow a target which moves in one dimension. 
 
State is specified by 4 continuous variables,e(t), e′(t), fn(t), f′n(t). 
 e(t), fn(t) are the positional error and the normal force felt by the agent.
 e′(t), f′n(t)are their first derivatives over time.
 
The variables are computed for each agent separately, complying with their point of view.
Action is defined as the force produced by an agent.

The motion of the object as a result of the applied forces is simulated using Newtonian laws.
