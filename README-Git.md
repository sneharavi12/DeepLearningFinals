
# DeepLearningFinals

# Train an autonomous vehicle to avoid obstacles


Necessary dependencies should be installed through the docker file.
#execute this to pull the required files

sudo docker pull sneharavi12/deeplearning:latest1

Pymunk version 4 for python2 is installed. Perform the following to convert it to Python3

Update from Python 2 to 3:

cd pymunk-pymukn-4.0.0/pymunk

2to3 -w *.py

Install it:

cd .. 
python3 setup.py install

Running the Game:

Train the model in the first step.

python3 learning.py

Run the trained model in the next step.

python3 playing.py

The car drives around through obstacles and restarts on bumping into any.

## Building Gym Torcs

##Why torcs?
TORCS, The Open Racing Car Simulator is a highly portable multi platform car racing simulation. 
It is used as ordinary car racing game, as AI racing game and as research platform. It runs on Linux (x86, AMD64 and PPC), FreeBSD, OpenSolaris, MacOSX and Windows.

You can visualize how the neural network learns over time and inspect its learning process, rather than just looking at the final result
TORCS can help us simulate and understand machine learning technique in automated driving, which is important for self-driving car technologies

There are three steps  to have this agent running.
-Server for Torcs
-lient for Torcs
-An environment, built like Gym environments that gives the observations and rewards based on the agent st.

#Server
v-Torcs
This is an all in one package of TORCS. 
The link below gives a complete overview of how this can be installed and set up on a linux machine.
https://github.com/giuse/vtorcs
This captures various sensor information that can be used to train the agent once we build the environment

#Client
SnakeOil is a Python library for interfacing with a TORCS race car simulator
Its as simle as creating the client as shown and implementing the custom 
drive function	
Involves mechanics of driving the car & not its implementation



