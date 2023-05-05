import copy
import time
import pandas as pd

import pygame
import random
from enum import Enum
from collections import namedtuple
from helper import get_game_image, get_action
import pickle
from game import Direction

##########################
# Snake for human players#

##########################
pygame.init()

font = pygame.font.Font('arial.ttf', 25)
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 30  # Moving speed, recommend 30-40


class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        past_direction = copy.deepcopy(self.direction)
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # 2. move

        self._move(self.direction)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        pygame.time.delay(100)
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score

    def _is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for i, pt in enumerate(self.snake):
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            if i == 0:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            else:
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


if __name__ == '__main__':

    try:
        with open('data/train_data.pkl', 'rb') as f:
            data_list = pickle.load(f)
            f.close()
    except IOError:
        data_list = []

    print("--------Game start--------")
    game = SnakeGame()
    # game loop
    actions_collect = []
    score = 0
    counts_time = []
    counts = 0
    num_actions = 0
    while True:
        past_dir = game.direction
        game_over, new_score = game.play_step()
        counts = counts + 1
        current_img = get_game_image(game.snake, game.food)
        new_dir = game.direction
        action = get_action(past_dir, new_dir)
        actions_collect.append({'image': current_img, 'act': action, 'ifend': int(game_over)})
        num_actions += 1 if action != [1, 0, 0] else 0

        # save data in different scenarios
        # 1. game over
        if game_over is True:
            actions_collect = actions_collect[-1]
            data_list.append(actions_collect)
            score = new_score
            new_score = 0
            actions_collect = []
            print('Final Score', score)
            pygame.event.get()
            pygame.quit()
            break
        # 2. score update/get food
        if new_score > score:
            actions_collect = actions_collect[-1]
            data_list.append(actions_collect)
            counts_time.append({'snake': len(game.snake), 'cost': counts, 'num_actions': num_actions})
            num_actions = 0
            counts = 0
            score = new_score
            new_score = 0
            actions_collect = []
            continue

        # long time run
        if len(actions_collect) > 10:
            actions_collect = actions_collect[-2]
            actions_collect['ifend'] = 2
            data_list.append(actions_collect)
            actions_collect = []

    # Save the images and actions to a file
    with open('./data/train_data.pkl', 'wb') as f:
        pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    # Simple player actions record, used for analyze routines of human players
    sum_df = pd.DataFrame.from_records(counts_time)
    sum_df.to_csv('./data/operation1.csv', mode='a', index=False)
