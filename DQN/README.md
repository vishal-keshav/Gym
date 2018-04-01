# Deep Q-Learning

### Structural overview
A global Q network and local Q network.
Both network are identical with respect to parameters where input is observation
and output is Q values corresponding to each possible discrete actions.
State or observation need not be discrete.

Network forward connection and gradient connection will be updated soon!

### Functional overview
Global Q network will be updated at a slower rate as
compared to local Q Network. Updation of local Q network follows Bellman equation
whereas global Q network just copies the local parameter once sufficiently trained

### Implementation details
Below is the algorithm which is being implemented:
![Pseudo-code][algorithm]

More updates here...

### Implementation caveats

[algorithm]: Algorithm.png
