# DeepLearningFinals
Reinforcement Learning for Self-Driving Cars

1. OBSTACLE DETECTION
Necessary dependencies should be installed through the docker file.

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

