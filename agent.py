import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot, get_game_image, get_action
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
from SL_train import SLnet

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self, slmodel):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, slmodel=slmodel, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self, game):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, game)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done, game):
        self.trainer.train_step(state, action, reward, next_state, done, game)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    data = []
    plot_scores = []
    plot_mean_scores = []
    plot_scores_after_train = []
    total_scores_after_train = 0
    total_score = 0
    record = 0
    old_reward = 0
    slmodel = torch.load("C:/Users/panxi/PycharmProjects/NeuroScience/650/data/best_model.pt")
    agent = Agent(slmodel)
    game = SnakeGameAI()
    while True:
        past_dir = game.direction

        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move, old_reward)
        state_new = agent.get_state(game)
        new_dir = game.direction
        current_img = get_game_image(game.snake, game.food)
        action = get_action(past_dir, new_dir)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done, game)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # if game over, collect previous step and train again.
        if done:
            if game.frame_iteration < 100 * len(game.snake):
                data.append({'image': current_img, 'act': action, 'ifend': int(1)})
            if len(data) % 10 == 9:
                try:
                    with open('./data/train_data_fail.pkl', 'rb') as f:
                        data_load = pickle.load(f)
                        f.close()
                except IOError:
                    print('No such pickle file, create a new one!')
                    data_load = []
                    pass
                with open('./data/train_data_fail.pkl', 'wb') as f:
                    pickle.dump(data_load+data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()
                data = []
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory(game)

            if score > record:
                record = score
                agent.model.save()

            if agent.n_games > 75:
                total_scores_after_train += score
                mean_score_after_train = total_scores_after_train / (agent.n_games - 75)
            else:
                total_scores_after_train += 0
                mean_score_after_train = 0

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games

            plot_mean_scores.append(mean_score)
            plot_scores_after_train.append(mean_score_after_train)
            plot(plot_scores, plot_mean_scores, plot_scores_after_train)


if __name__ == '__main__':
    train()
