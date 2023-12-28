#!/usr/bin/env python
# coding: utf-8


#*** Recurrent Reinforcement Learning for Cyber Threat Detection on IoT Devices ***#
'''
This code covers the implementation of the RRIoT algorithm as described in the manuscript 
"RRIoT: Recurrent Reinforcement Learning for Cyber Threat Detection on IoT devices." 
This code was adapted from another repository, https://github.com/gcamfer/Anomaly-ReactionRL 
and edited to reflect changes per our manuscript. For our code, we have included a single 
IoT device; other IoT devices follow a similar programmatic convention.  
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, clone_model
from keras.layers import Dense, SimpleRNN, LSTM, Reshape
from keras import optimizers
from keras import backend as K

import json
import os
import sys
import time


#### Data Preprocessing

data = pd.read_csv('C:\\Users\\crookard\\Desktop\\TON-IoT\\Train_Test_IoT_Weather.csv', sep=',')

# Keep Only Relevant Features
data = data[['ts', 'temperature', 'pressure' 'humidity' ,'type']] 

# Creating Histogram 
data['type'].value_counts().plot(kind='bar', title='Weather Traffic Count')

# Columns to be transformed (excluding 'type')
columns_to_transform = data.columns.drop('type')

# Applying Standard Scaler to these columns
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[columns_to_transform])

# Convert the scaled data back to a DataFrame
data_scaled_df = pd.DataFrame(data_scaled, columns=columns_to_transform)

# Reset index for the 'type' column to align the data
type_series = data['type'].reset_index(drop=True)

# Concatenating the scaled data with the 'type' column
final_data = pd.concat([data_scaled_df, type_series], axis=1)
print(final_data.head())



#Split the Data and then Write to CSV
train, test = train_test_split(final_data, test_size=0.20)

train.to_csv('RRIoT\\TON_Weather_train.txt',sep=',',index=False)
test.to_csv('RRIoT\\TON_Weather_test.txt',sep=',',index=False)

train.to_csv('RRIoT\\TON_Weather_train.csv',sep=',',index=False)
test.to_csv('RRIoT\\TON_Weather_test.csv',sep=',',index=False)

train_path = 'C:\\Users\\crookard\\Desktop\\RRIoT\\Notebooks\\RRIoT\\TON_Weather_train.txt'
test_path = 'C:\\Users\\crookard\\Desktop\\RRIoT\\Notebooks\\RRIoT\\TON_Weather_test.txt'

class data_cls:
    def __init__(self,train_test,**kwargs):
        col_names = ['ts', 'temperature', 'pressure' 'humidity' ,'type']
        
        # Data formated path and test path. 
        self.train_test = train_test
     
        self.index = 0
        if(train_test=="train"):
          self.train_path = train_path
        else:
          self.train_path = test_path
        
        self.df = pd.read_csv(self.train_path,sep=',')    
        
        #self.attack_types = ['normal','attack']
        #self.attack_names = ['normal','attack']
        #self.attack_map = {'normal':'normal', 'attack':'attack'}
               
        self.attack_types = ['normal', 'ddos', 'backdoor', 'injection', 'ransomware', 'password', 'xss', 'scanning']
        self.attack_names = ['normal', 'ddos', 'backdoor', 'injection', 'ransomware', 'password', 'xss', 'scanning']
        self.attack_map = {'normal':'normal', 
                           'backdoor':'backdoor',
                           'ddos':'ddos', 
                           'injection':'injection', 
                           'ransomware':'ransomware', 
                           'password':'password', 
                           'xss':'xss', 
                           'scanning':'scanning'}

        self.all_attack_names = list(self.attack_map.keys())
        
        self.df = pd.concat([self.df.drop('type', axis=1), pd.get_dummies(self.df['type'])], axis=1)
        
    def get_shape(self):
              
        self.data_shape = self.df.shape
        # stata + labels
        return self.data_shape
    
    ''' Get n-rows from loaded data 
        The dataset must be loaded in RAM
    '''
    def get_batch(self,batch_size=100):
                
        # Read the df rows
        indexes = list(range(self.index,self.index+batch_size))    
        if max(indexes)>self.data_shape[0]-1:
            dif = max(indexes)-self.data_shape[0]
            indexes[len(indexes)-dif-1:len(indexes)] = list(range(dif+1))
            self.index=batch_size-dif
            batch = self.df.iloc[indexes]
        else: 
            batch = self.df.iloc[indexes]
            self.index += batch_size    
            
        labels = batch[self.attack_names]    
        batch = batch.drop(self.all_attack_names,axis=1)
        return batch,labels
    
    def get_full(self):
        labels = self.df[self.attack_names]
        batch = self.df.drop(self.all_attack_names,axis=1)       
        return batch,labels
 
# Huber loss function        
def huber_loss(y_true, y_pred, clip_value=1):
    '''
    Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    for more details about the Huber loss function.
    '''
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

# Needed for keras huber_loss locate
import keras.losses
keras.losses.huber_loss = huber_loss

class QNetwork():
    """
    Q-Network Estimator
    Represents the global model for the table
    """

    def __init__(self,obs_size,num_actions,hidden_size = 100,
                 hidden_layers = 1,learning_rate=.2):
        """
        Initialize the network with the provided shape
        """
        self.obs_size = obs_size
        self.num_actions = num_actions
        
        # Network arquitecture
        self.model = Sequential()
        
        # Add input layer
        self.model.add(Dense(hidden_size, input_shape=(obs_size,),
                             activation='relu'))
        
        
        # Add hidden layers -> Edited to Reflect Architecture Changes
        for layers in range(hidden_layers):
            self.model.add(Dense(hidden_size, activation='relu'))
                
        # Recurrent layer
            self.model.add(Reshape(input_shape=(hidden_size,), target_shape=(hidden_size, 1)))
            self.model.add(LSTM(hidden_size, activation='tanh'))      
            
        # Add output layer    
        self.model.add(Dense(num_actions))
        
        #optimizer = optimizers.SGD(learning_rate)
        #optimizer = optimizers.Adam(alpha=learning_rate)
        optimizer = tf.keras.optimizers.Adam(0.00025)
        #optimizer = optimizers.RMSpropGraves(learning_rate, 0.95, self.momentum, 1e-2)
        
        # Compilation of the model with optimizer and loss
        self.model.compile(loss=huber_loss,optimizer=optimizer) 
        
        # Model Summary
        self.model.summary()

    def predict(self,state,batch_size=1):
        """
        Predicts action values.
        """
        return self.model.predict(state,batch_size=batch_size)

    def update(self, states, q):
        """
        Updates the estimator with the targets.

        Args:
          states: Target states
          q: Estimated values

        Returns:
          The calculated loss on the batch.
        """
        loss = self.model.train_on_batch(states, q)
        return loss
    
    def copy_model(model):
        """Returns a copy of a keras model."""
        model.save('tmp_model')
        return keras.models.load_model('tmp_model')


class QNetwork_DDPG():
    """
    Q-Network Estimator
    Represents the global model for the table
    """

    def __init__(self, obs_size, num_actions, 
                 actor_hidden_size=100, critic_hidden_size=100,
                 actor_hidden_layers=1, critic_hidden_layers=1, 
                 actor_learning_rate=.0001, critic_learning_rate=.001):
        """
        Initialize the network with the provided shape
        """
        self.obs_size = obs_size
        self.num_actions = num_actions

        # Actor Network arquitecture
        self.actor_model = Sequential()

        # Add input layer
        self.actor_model.add(Dense(actor_hidden_size, input_shape=(obs_size,), activation='relu'))

        # Add hidden layers -> Edited to Reflect Architecture Changes
        for _ in range(1):
            self.actor_model.add(Dense(actor_hidden_size, activation='relu'))

            # Recurrent layer
            self.actor_model.add(Reshape(input_shape=(actor_hidden_size,), target_shape=(actor_hidden_size, 1)))
            #self.actor_model.add(SimpleRNN(actor_hidden_size, activation='tanh'))
            self.actor_model.add(LSTM(actor_hidden_size, activation='tanh'))
            
        # Add output layer
        self.actor_model.add(Dense(num_actions, activation='tanh'))

        # Critic Network arquitecture
        self.critic_model = Sequential()

        # Add input layer
        self.critic_model.add(Dense(critic_hidden_size, input_shape=(obs_size,), activation='relu'))

        # Add hidden layers -> Edited to Reflect Architecture Changes
        for _ in range(1):
            self.critic_model.add(Dense(critic_hidden_size, activation='relu'))
            
            # Recurrent Layer 
            self.critic_model.add(Reshape(input_shape=(critic_hidden_size,), target_shape=(critic_hidden_size, 1)))
            #self.critic_model.add(SimpleRNN(critic_hidden_size,))
            self.critic_model.add(LSTM(critic_hidden_size,))            
            
        # Add output layer
        self.critic_model.add(Dense(1))

        # Define the optimizer for both actor and critic networks
        actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
        critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)

        # Compilation of the models with optimizer and loss
        self.actor_model.compile(loss='huber_loss', optimizer=actor_optimizer)
        self.critic_model.compile(loss='huber_loss', optimizer=critic_optimizer)

        # Model Summary
        self.actor_model.summary()
        self.critic_model.summary()

    def predict(self, state, batch_size=1):
        """
        Predicts action values using the actor network.
        """
        return self.actor_model.predict(state, batch_size=batch_size)

    def update(self, states, actions, q_targets):
        """
        Updates the actor and critic networks.

        Args:
          states: Target states
          actions: Estimated actions
          q_targets: Estimated values

        Returns:
          The calculated loss on the batch.
        """
        # Update critic network
        critic_loss = self.critic_model.train_on_batch([states, actions], q_targets)

        # Update actor network
        with tf.GradientTape() as tape:
            # Compute the predicted actions
            predicted_actions = self.actor_model(states)

            # Compute the critic values for the predicted actions
            critic_values = self.critic_model([states, predicted_actions])

            # Compute actor loss as negative critic value
            actor_loss = -tf.math.reduce_mean(critic_values)

        # Compute actor gradients
        actor_gradients = tape.gradient(actor_loss, self.actor_model.trainable_variables)

        # Apply actor gradients
        self.actor_model.optimizer.apply_gradients(zip(actor_gradients, self.actor_model.trainable_variables))

        return critic_loss, actor_loss

    def copy_model(model):
        """Returns a copy of a keras model."""
        model.save('tmp_model')
        return keras.models.load_model('tmp_model')

#Policy interface
class Policy:
    def __init__(self, num_actions, estimator):
        self.num_actions = num_actions
        self.estimator = estimator
    
class Epsilon_greedy(Policy):
    def __init__(self,estimator ,num_actions ,epsilon,min_epsilon,decay_rate, epoch_length):
        Policy.__init__(self, num_actions, estimator)
        self.name = "Epsilon Greedy"
        
        if (epsilon is None or epsilon < 0 or epsilon > 1):
            print("EpsilonGreedy: Invalid value of epsilon", flush = True)
            sys.exit(0)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.actions = list(range(num_actions))
        self.step_counter = 0
        self.epoch_length = epoch_length
        self.decay_rate = decay_rate
        
        #if epsilon is up 0.1, it will be decayed over time
        if self.epsilon > 0.01:
            self.epsilon_decay = True
        else:
            self.epsilon_decay = False
    
    def get_actions(self,states):
        # get next action
        if np.random.rand() <= self.epsilon:
            actions = np.random.randint(0, self.num_actions,states.shape[0])
        else:
            self.Q = self.estimator.predict(states,states.shape[0])
            actions = []
            for row in range(self.Q.shape[0]):
                best_actions = np.argwhere(self.Q[row] == np.amax(self.Q[row]))
                actions.append(best_actions[np.random.choice(len(best_actions))].item())
            
        self.step_counter += 1 
        # decay epsilon after each epoch
        if self.epsilon_decay:
            if self.step_counter % self.epoch_length == 0:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate**self.step_counter)
            
        return actions


class ReplayMemory(object):
    """Implements basic replay memory"""

    def __init__(self, observation_size, max_size):
        self.observation_size = observation_size
        self.num_observed = 0
        self.max_size = max_size
        self.samples = {
                 'obs'      : np.zeros(self.max_size * 1 * self.observation_size,
                                       dtype=np.float32).reshape(self.max_size,self.observation_size),
                 'action'   : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
                 'reward'   : np.zeros(self.max_size * 1).reshape(self.max_size, 1),
                 'terminal' : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
               }

    def observe(self, state, action, reward, done):
        index = self.num_observed % self.max_size
        self.samples['obs'][index, :] = state
        self.samples['action'][index, :] = action
        self.samples['reward'][index, :] = reward
        self.samples['terminal'][index, :] = done

        self.num_observed += 1

    def sample_minibatch(self, minibatch_size):
        max_index = min(self.num_observed, self.max_size) - 1
        sampled_indices = np.random.randint(max_index, size=minibatch_size)

        s      = np.asarray(self.samples['obs'][sampled_indices, :], dtype=np.float32)
        s_next = np.asarray(self.samples['obs'][sampled_indices+1, :], dtype=np.float32)

        a      = self.samples['action'][sampled_indices].reshape(minibatch_size)
        r      = self.samples['reward'][sampled_indices].reshape((minibatch_size, 1))
        done   = self.samples['terminal'][sampled_indices].reshape((minibatch_size, 1))

        return (s, a, r, s_next, done)


'''
Reinforcement learning Agent definition
'''

class Agent(object):  
        
    def __init__(self, actions,obs_size, policy="EpsilonGreedy", **kwargs):
        self.actions = actions
        self.num_actions = len(actions)
        self.obs_size = obs_size
        
        self.epsilon = kwargs.get('epsilon', 1)
        self.min_epsilon = kwargs.get('min_epsilon', .1)
        self.gamma = kwargs.get('gamma', .001)
        self.minibatch_size = kwargs.get('minibatch_size', 2)
        self.epoch_length = kwargs.get('epoch_length', 100)
        self.decay_rate = kwargs.get('decay_rate',0.99)
        self.ExpRep = kwargs.get('ExpRep',True)
        if self.ExpRep:
            self.memory = ReplayMemory(self.obs_size, kwargs.get('mem_size', 10))
        
        self.ddqn_time = 100
        self.ddqn_update = self.ddqn_time
        
        #self.model_network = QNetwork(self.obs_size, self.num_actions,
        #                              kwargs.get('hidden_size', 100),
        #                              kwargs.get('hidden_layers',1),
        #                              kwargs.get('learning_rate',.2))
        #self.target_model_network = QNetwork(self.obs_size, self.num_actions,
        #                              kwargs.get('hidden_size', 100),
        #                              kwargs.get('hidden_layers',1),
        #                              kwargs.get('learning_rate',.2))
        #self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
        
        self.model_network = QNetwork_DDPG(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers',1),
                                      kwargs.get('learning_rate',.2))
        self.target_model_network = QNetwork_DDPG(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers',1),
                                      kwargs.get('learning_rate',.2))
        self.target_model_network.model = QNetwork_DDPG.copy_model(self.model_network.actor_model)
        
        if policy == "EpsilonGreedy":
            self.policy = Epsilon_greedy(self.model_network,len(actions),
                                         self.epsilon,self.min_epsilon,
                                         self.decay_rate,self.epoch_length)
        
        
    def learn(self, states, actions,next_states, rewards, done):
        if self.ExpRep:
            self.memory.observe(states, actions, rewards, done)
        else:
            self.states = states
            self.actions = actions
            self.next_states = next_states
            self.rewards = rewards
            self.done = done        
    def update_model(self):
        if self.ExpRep:
            (states, actions, rewards, next_states, done) = self.memory.sample_minibatch(self.minibatch_size)
        else:
            states = self.states
            rewards = self.rewards
            next_states = self.next_states
            actions = self.actions
            done = self.done
        
        next_actions = []
        # Compute Q targets
        
        #Q_prime = self.model_network.predict(next_states,self.minibatch_size)
        Q_prime = self.target_model_network.predict(next_states,self.minibatch_size)
        
        for row in range(Q_prime.shape[0]):
            best_next_actions = np.argwhere(Q_prime[row] == np.amax(Q_prime[row]))
            next_actions.append(best_next_actions[np.random.choice(len(best_next_actions))].item())
        sx = np.arange(len(next_actions))
        
        # Compute Q(s,a)
        Q = self.model_network.predict(states,self.minibatch_size)
        
        # Q-learning update
        targets = rewards.reshape(Q[sx,actions].shape) +                   self.gamma * Q[sx,next_actions] *                   (1-done.reshape(Q[sx,actions].shape))   
        Q[sx,actions] = targets  
        
        loss = self.model_network.actor_model.train_on_batch(states,Q)#inputs,targets    #UPDATED HERE     
        
        # Update Timer 
        self.ddqn_update -= 1
        if self.ddqn_update == 0:
            self.ddqn_update = self.ddqn_time
            #self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
            self.target_model_network.actor_model.set_weights(self.model_network.actor_model.get_weights()) #UPDATED HERE
        return loss    

    def act(self, state,policy):
        raise NotImplementedError


class DefenderAgent(Agent):      
    def __init__(self, actions, obs_size, policy="EpsilonGreedy", **kwargs):
        super().__init__(actions,obs_size, policy="EpsilonGreedy", **kwargs)
        
    def act(self,states):
        # Get actions under the policy
        actions = self.policy.get_actions(states)
        return actions
    
class AttackAgent(Agent):      
    def __init__(self, actions, obs_size, policy="EpsilonGreedy", **kwargs):
        super().__init__(actions,obs_size, policy="EpsilonGreedy", **kwargs)
        
    def act(self,states):
        # Get actions under the policy
        actions = self.policy.get_actions(states)
        return actions

'''
Reinforcement learning Enviroment Definition
'''
class RLenv(data_cls):
    def __init__(self,train_test,**kwargs):
        data_cls.__init__(self,train_test,**kwargs)
        self.data_shape = data_cls.get_shape(self)
        self.batch_size = kwargs.get('batch_size',1) # experience replay -> batch = 1
        self.iterations_episode = kwargs.get('iterations_episode',10)

    '''
    _update_state: function to update the current state
    Returns:
        None
    Modifies the self parameters involved in the state:
        self.state and self.labels
    Also modifies the true labels to get learning knowledge
    '''
    def _update_state(self):        
        self.states,self.labels = data_cls.get_batch(self)
        
        # Update statistics
        self.true_labels += np.sum(self.labels).values

    '''
    Returns:
        + Observation of the environment
    '''
    def reset(self):
        # Statistics
        self.def_true_labels = np.zeros(len(self.attack_types),dtype=int)
        self.def_estimated_labels = np.zeros(len(self.attack_types),dtype=int)
        self.att_true_labels = np.zeros(len(self.attack_names),dtype=int)
        
        self.state_numb = 0
        
        self.states,self.labels = data_cls.get_batch(self,self.batch_size)
        
        self.total_reward = 0
        self.steps_in_episode = 0
        return self.states.values 
   
    '''
    Returns:
        State: Next state for the game
        Reward: Actual reward
        done: If the game ends (no end in this case)
    
    In the adversarial enviroment, it's only needed to return the actual reward
    '''    
    def act(self,defender_actions,attack_actions):
        # Clear previous rewards        
        self.att_reward = np.zeros(len(attack_actions))       
        self.def_reward = np.zeros(len(defender_actions))
                
        attack = [self.attack_types.index(self.attack_map[self.attack_names[att]]) for att in attack_actions]
    
        self.def_reward = (np.asarray(defender_actions)==np.asarray(attack))*1
        self.att_reward = (np.asarray(defender_actions)!=np.asarray(attack))*1

        self.def_estimated_labels += np.bincount(defender_actions,minlength=len(self.attack_types))      
        for act in attack_actions:
            self.def_true_labels[self.attack_types.index(self.attack_map[self.attack_names[act]])] += 1

        # Get new state and new true values 
        attack_actions = attacker_agent.act(self.states)
        self.states = env.get_states(attack_actions)
        
        # Done allways false in this continuous task       
        self.done = np.zeros(len(attack_actions),dtype=bool)
            
        return self.states, self.def_reward,self.att_reward, attack_actions, self.done
    
    '''
    Provide the actual states for the selected attacker actions
    Parameters:
        self:
        attacker_actions: optimum attacks selected by the attacker
            it can be one of attack_names list and select random of this
    Returns:
        State: Actual state for the selected attacks
    '''
    def get_states(self,attacker_actions):
        first = True
        for attack in attacker_actions:
            if first:
                minibatch = (self.df[self.df[self.attack_names[attack]]==1].sample(1))
                first = False
            else:
                minibatch=minibatch.append(self.df[self.df[self.attack_names[attack]]==1].sample(1))
        
        self.labels = minibatch[self.attack_names]
        minibatch.drop(self.all_attack_names,axis=1,inplace=True)
        self.states = minibatch
        
        return self.states


#### Training Phase

if __name__ == "__main__":
  
    
    # Train batch
    batch_size = 1
    # batch of memory ExpRep
    minibatch_size = 100
    ExpRep = True
    
    iterations_episode = 200
  
    # Initialization of the enviroment
    env = RLenv('train',batch_size=batch_size,
                iterations_episode=iterations_episode)    
    # obs_size = size of the state
    obs_size = env.data_shape[1]-len(env.all_attack_names)
    print(obs_size)
    num_episodes = 100  
    
    '''
    Definition for the defensor agent.
    '''
    defender_valid_actions = list(range(len(env.attack_types))) # only detect type of attack
    defender_num_actions = len(defender_valid_actions)    
    
	
    def_epsilon = 1 # exploration
    min_epsilon = 0.01 # min value for exploration
    def_gamma = 0.001
    def_decay_rate = 0.99
    
    def_hidden_size = 200
    def_hidden_layers = 2
    
    def_learning_rate = .01
    
    defender_agent = DefenderAgent(defender_valid_actions,obs_size,"EpsilonGreedy",
                          epoch_length = iterations_episode,
                          epsilon = def_epsilon,
                          min_epsilon = min_epsilon,
                          decay_rate = def_decay_rate,
                          gamma = def_gamma,
                          hidden_size=def_hidden_size,
                          hidden_layers=def_hidden_layers,
                          minibatch_size = minibatch_size,
                          mem_size = 1000,
                          learning_rate=def_learning_rate,
                          ExpRep=ExpRep)
    
    #Pretrained defender
    #defender_agent.model_network.model.load_weights("models/RRIoT_model.h5")    
    
    '''
    Definition for the attacker agent.
    In this case the exploration is better to be greater
    The correlation sould be greater too so gamma bigger
    '''
    attack_valid_actions = list(range(len(env.attack_names)))
    attack_num_actions = len(attack_valid_actions)
	
    att_epsilon = 1
    min_epsilon = 0.5 # min value for exploration

    att_gamma = 0.001
    att_decay_rate = 0.99
    
    att_hidden_layers = 1
    att_hidden_size = 100
    
    att_learning_rate = 0.2
    
    attacker_agent = AttackAgent(attack_valid_actions,obs_size,"EpsilonGreedy",
                          epoch_length = iterations_episode,
                          epsilon = att_epsilon,
                          min_epsilon = min_epsilon,
                          decay_rate = att_decay_rate,
                          gamma = att_gamma,
                          hidden_size=att_hidden_size,
                          hidden_layers=att_hidden_layers,
                          minibatch_size = minibatch_size,
                          mem_size = 1000,
                          learning_rate=att_learning_rate,
                          ExpRep=ExpRep)
    
        
    
    # Statistics
    att_reward_chain = []
    def_reward_chain = []
    att_loss_chain = []
    def_loss_chain = []
    def_total_reward_chain = []
    att_total_reward_chain = []
    
	# Print parameters
    print("-------------------------------------------------------------------------------")
    print("Total epoch: {} | Iterations in epoch: {}"
          "| Minibatch from mem size: {} | Total Samples: {}|".format(num_episodes,
                         iterations_episode,minibatch_size,
                         num_episodes*iterations_episode))
    print("-------------------------------------------------------------------------------")
    print("Dataset shape: {}".format(env.data_shape))
    print("-------------------------------------------------------------------------------")
    print("Attacker parameters: Num_actions={} | gamma={} |" 
          " epsilon={} | ANN hidden size={} | "
          "ANN hidden layers={}|".format(attack_num_actions,
                             att_gamma,att_epsilon, att_hidden_size,
                             att_hidden_layers))
    print("-------------------------------------------------------------------------------")
    print("Defense parameters: Num_actions={} | gamma={} | "
          "epsilon={} | ANN hidden size={} |"
          " ANN hidden layers={}|".format(defender_num_actions,
                              def_gamma,def_epsilon,def_hidden_size,
                              def_hidden_layers))
    print("-------------------------------------------------------------------------------")

    # Main loop
    training_start_time = time.time()
    attacks_by_epoch = []
    attack_labels_list = []
    for epoch in range(num_episodes):
        start_time = time.time()
        att_loss = 0.
        def_loss = 0.
        def_total_reward_by_episode = 0
        att_total_reward_by_episode = 0
        # Reset enviromet, actualize the data batch with random state/attacks
        states = env.reset()
        
        # Get actions for actual states following the policy
        attack_actions = attacker_agent.act(states)
        states = env.get_states(attack_actions)    
        
        done = False
       
        attacks_list = []
        # Iteration in one episode
        for i_iteration in range(iterations_episode):
            
            attacks_list.append(attack_actions[0])
            # Apply actions, get rewards and new state
            act_time = time.time()  
            defender_actions = defender_agent.act(states)
            # Environment actuation for this actions
            next_states,def_reward, att_reward,next_attack_actions, done = env.act(defender_actions,attack_actions)
            # If the epoch*batch_size*iterations_episode is largest than the df

            
            attacker_agent.learn(states,attack_actions,next_states,att_reward,done)
            defender_agent.learn(states,defender_actions,next_states,def_reward,done)
            
            act_end_time = time.time()
            
            # Train network, update loss after at least minibatch_learns
            if ExpRep and epoch*iterations_episode + i_iteration >= minibatch_size:
                def_loss += defender_agent.update_model()
                att_loss += attacker_agent.update_model()
            elif not ExpRep:
                def_loss += defender_agent.update_model()
                att_loss += attacker_agent.update_model()
                

            update_end_time = time.time()

            # Update the state
            states = next_states
            attack_actions = next_attack_actions
            
            
            # Update statistics
            def_total_reward_by_episode += np.sum(def_reward,dtype=np.int32)
            att_total_reward_by_episode += np.sum(att_reward,dtype=np.int32)
        
        attacks_by_epoch.append(attacks_list)
        # Update user view
        def_reward_chain.append(def_total_reward_by_episode) 
        att_reward_chain.append(att_total_reward_by_episode) 
        def_loss_chain.append(def_loss)
        att_loss_chain.append(att_loss) 

        
        end_time = time.time()
        print("\r\n|Epoch {:03d}/{:03d}| time: {:2.2f}|\r\n"
                "|Def Loss {:4.4f} | Def Reward in ep {:03d}|\r\n"
                "|Att Loss {:4.4f} | Att Reward in ep {:03d}|"
                .format(epoch, num_episodes,(end_time-start_time), 
                def_loss, def_total_reward_by_episode,
                att_loss, att_total_reward_by_episode))
        
        
        print("|Def Estimated: {}| Att Labels: {}".format(env.def_estimated_labels,
              env.def_true_labels))
        attack_labels_list.append(env.def_true_labels)
        
    # Training Time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    hours, rem = divmod(total_training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\r|Total Training Time: {:0>2}:{:0>2}:{:05.2f}|".format(int(hours),int(minutes),seconds))


# Training Visualization

if not os.path.exists('models'):
    os.makedirs('models')
# Save trained model weights and architecture, used in test
defender_agent.model_network.actor_model.save_weights("models/RRIoT_model_scaled.h5", overwrite=True)
with open("models/RRIoT_model_scaled.json", "w") as outfile:
    json.dump(defender_agent.model_network.actor_model.to_json(), outfile)

    
if not os.path.exists('results'):
    os.makedirs('results')    
# Plot training results
plt.figure(1)
plt.subplot(211)
plt.plot(np.arange(len(def_reward_chain)),def_reward_chain,label='Defense')
plt.plot(np.arange(len(att_reward_chain)),att_reward_chain,label='Attack')
plt.title('Total reward by episode')
plt.xlabel('n Episode')
plt.ylabel('Total reward')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)

plt.subplot(212)
plt.plot(np.arange(len(def_loss_chain)),def_loss_chain,label='Defense')
plt.plot(np.arange(len(att_loss_chain)),att_loss_chain,label='Attack')
plt.title('Loss by episode')
plt.xlabel('n Episode')
plt.ylabel('loss')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
plt.tight_layout()
#plt.show()
plt.savefig('results/train_adv.eps', format='eps', dpi=1000)


#### Testing Phase

from keras.models import model_from_json

import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import  confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

with open("models/RRIoT_model_scaled.json", "r") as jfile:
    model = model_from_json(json.load(jfile))
model.load_weights("models/RRIoT_model_scaled.h5")

model.compile(loss=huber_loss,optimizer="sgd")


# Define environment, game, make sure the batch_size is the same in train
env_test = RLenv('test')

total_reward = 0    


true_labels = np.zeros(len(env_test.attack_types),dtype=int)
estimated_labels = np.zeros(len(env_test.attack_types),dtype=int)
estimated_correct_labels = np.zeros(len(env_test.attack_types),dtype=int)

states , labels = env_test.get_full()

start_time=time.time()
q = model.predict(states)
actions = np.argmax(q,axis=1)        


#### SAGE Analysis

import sage
from sage import PermutationEstimator, MarginalImputer

# Load the model
with open("models/RRIoT_model_scaled.json", "r") as jfile:
    model = model_from_json(json.load(jfile))
model.load_weights("models/RRIoT_model_scaled.h5")

model.compile(loss=huber_loss,optimizer="sgd")

# Define environment 
env_test = RLenv('test')

total_reward = 0    

true_labels = np.zeros(len(env_test.attack_types),dtype=int)
estimated_labels = np.zeros(len(env_test.attack_types),dtype=int)
estimated_correct_labels = np.zeros(len(env_test.attack_types),dtype=int)

# Get the states and labels
states , labels = env_test.get_full()

# Setup the SAGE imputer
imputer = sage.MarginalImputer(model, states[:128])

# Now we can use SAGE to interpret the policy
estimator = sage.PermutationEstimator(imputer, 'mse')

start_time = time.time()
q = model.predict(states)
actions = np.argmax(q,axis=1) 

end_time = time.time()
print("Total Test Time: ", end_time-start_time)

sage_values = estimator(np.array(states), np.array(labels))

# Plot results
print(sage_values)

feature_names = ['ts', 'temperature', 'pressure' 'humidity']

sage_values.plot(feature_names)


maped=[]
for indx,label in labels.iterrows():
    maped.append(env_test.attack_types.index(env_test.attack_map[label.idxmax()]))

labels,counts = np.unique(maped,return_counts=True)
true_labels[labels] += counts

for indx,a in enumerate(actions):
    estimated_labels[a] +=1              
    if a == maped[indx]:
        total_reward += 1
        estimated_correct_labels[a] += 1

action_dummies = pd.get_dummies(actions)
posible_actions = np.arange(len(env_test.attack_types))
for non_existing_action in posible_actions:
    if non_existing_action not in action_dummies.columns:
        action_dummies[non_existing_action] = np.uint8(0)
labels_dummies = pd.get_dummies(maped)

normal_f1_score = f1_score(labels_dummies[0].values,action_dummies[0].values)
dos_f1_score = f1_score(labels_dummies[1].values,action_dummies[1].values)
probe_f1_score = f1_score(labels_dummies[2].values,action_dummies[2].values)
r2l_f1_score = f1_score(labels_dummies[3].values,action_dummies[3].values)
u2r_f1_score = f1_score(labels_dummies[4].values,action_dummies[4].values)
    

Accuracy = [normal_f1_score,dos_f1_score,probe_f1_score,r2l_f1_score,u2r_f1_score]
Mismatch = estimated_labels - true_labels

acc = float(100*total_reward/len(states))
print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {:.2f}%'.format(total_reward,
      len(states),acc))
outputs_df = pd.DataFrame(index = env_test.attack_types,columns = ["Estimated","Correct","Total"])#,"F1_score"])
for indx,att in enumerate(env_test.attack_types):
   outputs_df.iloc[indx].Estimated = estimated_labels[indx]
   outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
   outputs_df.iloc[indx].Total = true_labels[indx]
   #outputs_df.iloc[indx].F1_score = Accuracy[indx]*100
   outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])
    
    
print(outputs_df)


# Testing Visualization

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
width = 0.35
pos = np.arange(len(true_labels))
p1 = plt.bar(pos, estimated_correct_labels, width, color='g') #pos+width

p2 = plt.bar(pos, (np.abs(estimated_correct_labels-true_labels)),width, 
             bottom=estimated_correct_labels, 
             color='r') 

p3 = plt.bar(pos, np.abs(estimated_labels-estimated_correct_labels),width, 
             bottom=estimated_correct_labels+(np.abs(estimated_correct_labels-true_labels)),
             color='b') 

#p2 = plt.bar(pos+width, (np.abs(estimated_correct_labels-true_labels)),width,color='r') #p1
#p3 = plt.bar(pos+width, np.abs(estimated_labels-estimated_correct_labels),width, 
#             bottom=(np.abs(estimated_correct_labels-true_labels)),color='b') #p2

ax.yaxis.set_tick_params(labelsize=15)
ax.set_xticks(pos) #pos+width/2
ax.set_xticklabels(env.attack_types,rotation='vertical',fontsize = 'xx-large')

#ax.set_yscale('log')

#ax.set_ylim([0, 100])
#ax.set_title('Test set scores',fontsize = 'xx-large')
#ax.set_title('Test set scores, Acc = {:.2f}'.format(acc))
plt.legend(('Correct estimated','False negative','False positive'),fontsize = 'x-large')
plt.tight_layout()
#plt.show()
plt.savefig('results/test_adv_imp.svg', format='svg', dpi=1000)

aggregated_data_test = np.array(maped)

print('Performance measures on Test data')
print('Accuracy =  {:.4f}'.format(accuracy_score(aggregated_data_test,actions)))
print('F1 =  {:.4f}'.format(f1_score(aggregated_data_test,actions, average='weighted')))
print('Precision_score =  {:.4f}'.format(precision_score(aggregated_data_test,actions, average='weighted')))
print('Recall_score =  {:.4f}'.format(recall_score(aggregated_data_test,actions, average='weighted')))
print('G_Mean_Score = {:.4f}'.format(geometric_mean_score(aggregated_data_test,actions, average='weighted')))

cnf_matrix = confusion_matrix(aggregated_data_test,actions)
np.set_printoptions(precision=2)
plt.figure()
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=env.attack_types, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('results/confusion_matrix_adversarial.svg', format='svg', dpi=1000)


mapa = {0:'normal', 1:'backdoor', 2:'ddos', 3:'injection', 4:'ransomware', 5:'password', 6:'xss', 7:'scanning'}
yt_app = pd.Series(maped).map(mapa)

perf_per_class = pd.DataFrame(index=range(len(yt_app.unique())),columns=['name', 'acc', 'f1', 'pre', 'rec', 'gmean'])
for i,x in enumerate(pd.Series(yt_app).value_counts().index):
    y_test_hat_check = pd.Series(actions).map(mapa).copy()
    y_test_hat_check[y_test_hat_check != x] = 'OTHER'
    yt_app = pd.Series(maped).map(mapa).copy()
    yt_app[yt_app != x] = 'OTHER'
    ac=accuracy_score( yt_app,y_test_hat_check)
    f1=f1_score( yt_app,y_test_hat_check,pos_label=x, average='binary')
    pr=precision_score( yt_app,y_test_hat_check,pos_label=x, average='binary')
    re=recall_score( yt_app,y_test_hat_check,pos_label=x, average='binary')
    gm=geometric_mean_score( yt_app,y_test_hat_check,pos_label=x, average='binary')
    perf_per_class.iloc[i]=[x,ac,f1,pr,re,gm]
    
print("\r\nOne vs All metrics: \r\n{}".format(perf_per_class))
