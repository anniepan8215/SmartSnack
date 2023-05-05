import matplotlib.pyplot as plt
from IPython import display
import pygame
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from PIL import Image

BLOCK_SIZE = 20
SPEED = 20
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
plt.ion()


def plot(scores, mean_scores, scores_after_train):
    '''
    Plot Snake score
    :param scores: Score per rounds
    :param mean_scores: average score over all round
    :param scores_after_train: average score of games after 75 rounds
    '''
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='scores')
    plt.plot(mean_scores, label='mean ovel all')
    plt.plot(scores_after_train, label='mean after 75')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)


def get_game_image(snake, food, w=640, h=480):
    # init display
    display = pygame.Surface((w, h))
    display.fill((0, 0, 0))

    # draw snake
    for i, pt in enumerate(snake):
        pygame.draw.rect(display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        if i == 0:
            pygame.draw.rect(display, GREEN, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        else:
            pygame.draw.rect(display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

    # draw food
    pygame.draw.rect(display, (200, 0, 0), pygame.Rect(food.x, food.y, 20, 20))

    # # convert surface to image matrix
    # image_matrix = pygame.surfarray.array3d(display)
    # convert surface to PIL image
    image_surface = pygame.surfarray.make_surface(np.transpose(pygame.surfarray.array3d(display), (1, 0, 2)))
    pil_image = Image.frombytes('RGB', (w, h), pygame.image.tostring(image_surface, 'RGB'))
    return pil_image


def get_data_loader(images, actions, batch_size):
    # Convert images and actions to PyTorch tensors
    # Load the images and actions from the file
    with open('data/train_data.pkl', 'rb') as f:
        data = pickle.load(f)
        images = data['images']
        actions = data['actions']

    images = torch.stack([torch.Tensor(img) for img in images])
    actions = torch.LongTensor(actions)

    # Create a TensorDataset from the tensors
    dataset = TensorDataset(images, actions)

    # Create a DataLoader from the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


from game import Direction


def get_action(past_dir, new_dir):
    '''
    Transforem game.Direction to snake action
    :param past_dir: direction_t
    :param new_dir: direction_t_1
    :return: boolean list [straight, right, left]
    '''
    action = [0, 0, 0]  # [straight, right, left]

    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    past_idx = clock_wise.index(past_dir)
    new_idx = clock_wise.index(new_dir)

    if past_idx == new_idx:
        action = [1, 0, 0]
    elif new_idx == (past_idx + 1) % 4:
        action = [0, 1, 0]
    else:
        action = [0, 0, 1]

    return action
