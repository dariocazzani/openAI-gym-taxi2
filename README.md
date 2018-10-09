# openAI-gym-taxi2

### Install requirements
```
pip install -r requirements.txt
```

### Train with Expected SARSA [2] and let the agent play:
```
python main.py
```

### The environment

* **The Taxi Problem**:<br />
  From [Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition](https://arxiv.org/abs/cs/9905014) by Tom Dietterich

* **Description**: <br />
  There are four designated locations in the grid world indicated by 
  * R(ed) 
  * B(lue)
  * G(reen)
  * Y(ellow). 

  When the episode starts, the taxi starts off at a random square and the passenger is at a random location. 
  The taxi drive to the passenger's location, pick up the passenger, drive to the passenger's destination 
  (another one of the four specified locations), and then drop off the passenger. 
  Once the passenger is dropped off, the episode ends.

* **Observations**: <br /> 
  There are 500 discrete actions since there are 25 taxi positions, 5 possible locations of the passenger 
  (including the case when the passenger is the taxi), and 4 destination locations. 

* **Actions**: <br />
  There are 6 discrete deterministic actions:
    * 0: move south
    * 1: move north
    * 2: move east 
    * 3: move west 
    * 4: pickup passenger
    * 5: dropoff passenger
    
* **Rewards**: <br />
  There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
    
* **Rendering**: <br />
  * blue: passenger
  * magenta: destination
  * yellow: empty taxi
  * green: full taxi
  * other letters: locations

#### References
[1] Sutton, R. S. & Barto, A. G. [Reinforcement learning: an introduction](http://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), 1988.

[2] Harm van Seijen, Hado van Hasselt, Shimon Whiteson and Marco Wiering [A Theoretical and Empirical Analysis of Expected Sarsa](http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf)
