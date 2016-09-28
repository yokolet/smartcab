import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q_table = {}
        # for epsilon calculation
        self.prev_reward = -10
        self.epsilon_policies = [lambda x: 0.1 if x['deadline'] % 5 == 0 else 0.9,
                                 lambda x: 0.1 if x['reward'] < -0.7 else 0.9]

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def learn(self, alpha, gamma, reward, action):
        # gets next state
        next_inputs = self.env.sense(self)
        next_waypoint = self.planner.next_waypoint()
        next_state = (next_inputs['light'], next_inputs['oncoming'],
                      next_inputs['left'], next_waypoint)
        # finds max among next state
        q_max = max([self.q_table.get((next_state, a), 0.0) for a in self.env.valid_actions])
        # caclulates Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max Q(s',a') - Q(s,a))
        q_sa = self.q_table.get((self.state, action), 0.0)
        self.q_table[(self.state, action)] = q_sa + alpha * (reward + gamma * q_max - q_sa)

    def best_action(self, actions):
        # gets values in q_table
        q_values = [(i, self.q_table.get((self.state, a), 0.0)) for i, a in enumerate(actions)]
        # finds max value
        q_values = sorted(q_values, key=lambda x: x[1], reverse=True)
        q_max = q_values[0][1]
        # finds actions who have max values
        q_maxes = filter(lambda x: x[1]==q_max, q_values)
        # choose action index randomly from actions of max value
        action_index = random.choice(map(lambda x: x[0], q_maxes))
        return actions[action_index]

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from
        # route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        env_state = self.env.agent_states[self]

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        
        # TODO: Select action according to your policy
        policy_no = 1
        epsilon = self.epsilon_policies[policy_no]({'deadline': deadline, 'reward': self.prev_reward})
        if epsilon < 0.5:
            action = random.choice(self.env.valid_actions)
        else:
            action = self.best_action(self.env.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        alpha = 0.5
        gamma = 0.8
        self.learn(alpha, gamma, reward, action)

        # saves reward for epsilon value
        self.prev_reward = reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
