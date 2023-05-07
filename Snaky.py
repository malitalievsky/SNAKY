import pygame as pg
import time
import numpy as np
import random

COLORS = {'BLACK': (0, 0, 0), 'WHITE': (255, 255, 255), 'RED': (255, 0, 0), 'GREEN': (0, 255, 0)}
DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  #right, left, down, up


class block():
    def __init__(self, block_side, color, pos):
       self.block_side = block_side
       self.color = color
       self.pos = pos  # np array


    def draw_block(self, surface):
       pg.draw.rect(surface, self.color,
                    (self.pos[0] * self.block_side, self.pos[1] * self.block_side, self.block_side, self.block_side))


class snaky(block):
    def __init__(self, body_positions, block_side): #body_positions = [list of first 4 np arrays representing positions of snake generated randomly]
        self.snaky = []  # list of blocks
        self.block_side = block_side
        for Position in body_positions:
            self.snaky.append(block(block_side, COLORS['GREEN'], Position))


    def draw_snake(self, surface):
        for Block in self.snaky:
            Block.draw_block(surface)


    def move_snake(self, move, apple):  # assuming the move given is legal , move = [0,1]
        prev_head = self.snaky[0]
        new_head = block(prev_head.block_side, COLORS['GREEN'], np.add(prev_head.pos, move))
        self.snaky.insert(0, new_head)
        ate_apple = self.ate_apple(self, apple) #ate_apple=boolean
        if not ate_apple:
            self.snaky.pop()

    @staticmethod
    def ate_apple(snake, apple):
        head = snake.snaky[0]
        if (head.pos == apple.pos).all():
            return True
        return False


    @staticmethod
    def crashed_into_wall(snake, HEIGHT, WIDTH):
        head = snake.snaky[0]
        if (head.pos[0] < 0) or (head.pos[0] >= HEIGHT):
            return True
        if (head.pos[1] < 0) or (head.pos[1] >= WIDTH):
            return True
        return False

    @staticmethod
    def ate_tail(snake):
        head = snake.snaky[0]
        for Block in snake.snaky[1:]:
            if((Block.pos == head.pos).all()):
                return True
        return False

    #  for state observation
    @staticmethod
    def is_NextStepInDirection_dangerous(snake, move, apple, HEIGHT, WIDTH): # is there danger on the next block in this direction
        head_nextPos = snake.snaky[0].pos + move
        lenSnake = len(snake.snaky) + int(snake.ate_apple(snake, apple))
        New_Positions = [head_nextPos]
        for i in range(lenSnake - 1):
            New_Positions.append(snake.snaky[i].pos)
        # newSnake = snake[:lenSnake - 1]
        newSnake = snaky(New_Positions, snake.block_side)
        return (newSnake.ate_tail(newSnake) and newSnake.crashed_into_wall(newSnake, HEIGHT, WIDTH))


    @staticmethod
    def general_danger_in_relativeDirections(snake, direction, HEIGHT, WIDTH):
        head_pos = snake.snaky[0].pos

        # check which direction the snake is headed
        is_dir_R = (direction == DIRECTIONS[0]).all()
        is_dir_L = (direction == DIRECTIONS[1]).all()
        is_dir_D = (direction == DIRECTIONS[2]).all()
        is_dir_U = (direction == DIRECTIONS[3]).all()

        # check tail position
        tail_R, tail_L, tail_D, tail_U = False, False, False, False
        for Block in snake.snaky[2:]:  # check blocks from 4th block (snake cant eat itself before block 4)
            if Block.pos[0] < head_pos[0]:
                tail_L = True
            elif Block.pos[0] > head_pos[0]:
                tail_R = True
            elif Block.pos[1] < head_pos[1]:
                tail_U = True
            else:
                tail_D = True

        # check wall
        wall_danger_R = is_dir_R * ((WIDTH/2) < head_pos[0])
        wall_danger_L = is_dir_L * ((WIDTH/2) > head_pos[0])
        wall_danger_D = is_dir_D * ((HEIGHT/2) < head_pos[1])
        wall_danger_U = is_dir_U * ((HEIGHT/2) > head_pos[1])


        # find out if there is danger in relative directions
        if is_dir_R:
            danger_straight = tail_R or wall_danger_R
            danger_relativeRight = tail_D or wall_danger_D
            danger_relativeLeft = tail_U or wall_danger_U

        elif is_dir_L:
            danger_straight = tail_L or wall_danger_L
            danger_relativeRight = tail_U or wall_danger_U
            danger_relativeLeft = tail_D or wall_danger_D

        elif is_dir_D:
            danger_straight = tail_D or wall_danger_D
            danger_relativeRight = tail_L or wall_danger_L
            danger_relativeLeft = tail_R or wall_danger_R

        elif is_dir_U:
            danger_straight = tail_U or wall_danger_U
            danger_relativeRight = tail_R or wall_danger_R
            danger_relativeLeft = tail_L or wall_danger_L

        return danger_straight, danger_relativeRight, danger_relativeLeft


class gameENV(snaky, block):
    def __init__(self, HEIGHT, WIDTH, block_side):
        self.REWARDS = {}
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.block_side = block_side
        self.snakeIsAlive = True
        self.snakeDirection = None  # numpy vector of size (2,)
        self.snaky = None
        self.apple = None
        # gui
        self.displayScreen = None
        self.score = 0
        self.clock = None


    def generate_initial_direction(self): # generate an initial random direction for the snake
        return np.array(random.choice(DIRECTIONS))  # snake direction = np array


    def generate_snake(self): # generate a snake of length 4 in a random position on the board
        rand_row = random.randint(0, self.HEIGHT - 1 - 4)
        rand_col = random.randint(0, self.WIDTH - 1 - 4)
        head_pos = np.array([rand_row, rand_col])
        snakyPositions = [head_pos]
        direction_of_growth = np.multiply(-1, self.snakeDirection)
        for pos in range(3):  # 0,1,2
            last_pos = snakyPositions[pos]
            snakyPositions.append(last_pos + direction_of_growth)
        snakyPositions = np.array(snakyPositions)
        return snaky(snakyPositions, self.block_side)


    def generate_apple(self):  # this method updates self since it is used multiple times along the game unlike generating initial position and generating snake
        rand_row = random.randint(0, self.HEIGHT - 1)  # randomrandint generates a number between 0-k including k
        rand_col = random.randint(0, self.WIDTH - 1)
        rand_pos = np.array([rand_row, rand_col])
        for Block in self.snaky.snaky:
            if (rand_pos == Block.pos).all():
                self.generate_apple()
        self.apple = block(self.block_side, COLORS['RED'], rand_pos)


    def update_is_alive(self):
        ateTail = self.snaky.ate_tail(self.snaky)  # whether the snake ate its own tail
        crashed = self.snaky.crashed_into_wall(self.snaky, self.HEIGHT, self.WIDTH)  # whether the snake crashed in a wall
        self.snakeIsAlive = not ateTail and not crashed


    def dist_from_apple(self):  # manhattan distance
        apple_pos = self.apple.pos
        snake_head_pos = self.snaky.snaky[0].pos
        distance = np.abs(apple_pos[0] - snake_head_pos[0]) + np.abs(apple_pos[1] - snake_head_pos[1])
        return distance


    def dist_from_apple_euclidian(self):  # euclidian distance
        apple_pos = self.apple.pos
        snake_head_pos = self.snaky.snaky[0].pos
        distance = (apple_pos[0] - snake_head_pos[0]) ** 2 + (apple_pos[1] - snake_head_pos[1]) ** 2
        return distance


    def set_rewards(self, appleReward, deathReward, wonReward, closerReward, furtherReward):
        self.REWARDS['closer'] = closerReward
        self.REWARDS['further'] = furtherReward
        self.REWARDS['apple'] = appleReward
        self.REWARDS['death'] = deathReward
        self.REWARDS['won'] = wonReward

        return self.REWARDS


    def step(self, move):
        done = False
        initial_distance = self.dist_from_apple()
        self.snaky.move_snake(move, self.apple)  # assuming the action give is legal
        self.snakeDirection = move  # update overall dir
        self.update_is_alive()
        curr_distance = self.dist_from_apple()

        #  if the snake crashes into a wall or eats its own tail the game ends
        if not self.snakeIsAlive:
            done = True
            reward = self.REWARDS['death']

        #  If the snake ate an apple
        if self.snaky.ate_apple(self.snaky, self.apple):
            self.score += 1

            if len(self.snaky.snaky) >= (self.WIDTH * self.HEIGHT - 1): #  if the length of the snake is max and agent wins
                done = True
                reward = self.REWARDS['won']

            else:    #  if the length of the snake is still smaller than the board, the game continues, a new apple is generated and the agent gets rewarded
                reward = self.REWARDS['apple']
                self.generate_apple()

        else:   #  if the snake didn't die and didn't eat an apple
            if (curr_distance < initial_distance):
                reward = self.REWARDS['closer']

            elif (curr_distance > initial_distance):
                reward = self.REWARDS['further']

        # generate state observation for new state, if DONE the new state obs is a np.array of zeroes
        if done:
            # next_state_observation = None
            next_state_observation = np.zeros(shape=(11,))
        else:
            next_state_observation = self.generate_state_observation()

        return next_state_observation, reward, done


    def reset(self):
        self.score = 0
        self.snakeDirection = self.generate_initial_direction()
        self.snaky = self.generate_snake()
        self.generate_apple()
        state_observation = self.generate_state_observation()
        return state_observation


    def generate_state_observation(self):
        observation = []
        head_pos = self.snaky.snaky[0].pos

        # danger
        '''
        observation.append(1 * self.snaky.is_NextStepInDirection_dangerous(self.snaky, self.snakeDirection, self.apple, self.HEIGHT, self.WIDTH))  # danger straight

        rotate_right_matrix = np.array([[0, 1], [-1, 0]])
        rightDir_inRelationToSnake = self.snakeDirection @ rotate_right_matrix
        observation.append(1 * self.snaky.is_NextStepInDirection_dangerous(self.snaky, rightDir_inRelationToSnake, self.apple, self.HEIGHT, self.WIDTH))  # danger to the right of the moving direction

        rotate_left_matrix = np.array([[0, -1], [1, 0]])
        leftDir_inRelationToSnake = self.snakeDirection @ rotate_left_matrix
        observation.append(1 * self.snaky.is_NextStepInDirection_dangerous(self.snaky, leftDir_inRelationToSnake, self.apple, self.HEIGHT, self.WIDTH))  # danger to the left of the moving direction
        '''

        danger_S, danger_R_inRelationToSnake, danger_L_inRelationToSnake = self.general_danger_in_relativeDirections(self.snaky, self.snakeDirection, self.HEIGHT, self.WIDTH)
        observation.append(danger_S * 1)
        observation.append(danger_R_inRelationToSnake * 1)
        observation.append(danger_L_inRelationToSnake * 1)


        # snake direction
        observation.append(int((DIRECTIONS[0] == self.snakeDirection).all()))  # snake direction is right : 0 or 1
        observation.append(int((DIRECTIONS[1] == self.snakeDirection).all()))  # snake direction is left : 0 or 1
        observation.append(int((DIRECTIONS[2] == self.snakeDirection).all()))  # snake direction is up : 0 or 1
        observation.append(int((DIRECTIONS[3] == self.snakeDirection).all()))  # snake direction is down : 0 or 1

        # apple
        apple_pos = self.apple.pos
        observation.append(int(head_pos[0] < apple_pos[0]))  # snake head is to the left of apple : 0 or 1
        observation.append(int(head_pos[0] > apple_pos[0]))  # snake head to the right of apple : 0 or 1
        observation.append(int(head_pos[1] < apple_pos[1]))  # snake head above apple : 0 or 1
        observation.append(int(head_pos[1] > apple_pos[1]))  # snake head below apple : 0 or 1

        return np.array(observation)

    def test_step(self, move):
        done = False
        initial_distance = self.dist_from_apple()
        self.snaky.move_snake(move, self.apple)  # assuming the action give is legal
        self.snakeDirection = move  # update overall dir
        self.update_is_alive()
        curr_distance = self.dist_from_apple()

        #  if the snake crashes into a wall or eats its own tail the game ends
        if not self.snakeIsAlive:
            done = True

        #  If the snake ate an apple
        if self.snaky.ate_apple(self.snaky, self.apple):
            self.score += 1
            if len(self.snaky.snaky) >= (self.WIDTH * self.HEIGHT - 1): #  if the length of the snake is max and agent wins
                done = True
            else:    #  if the length of the snake is still smaller than the board, the game continues, a new apple is generated and the agent gets rewarded
                self.generate_apple()

        # generate state observation for new state, if DONE the new state obs is a np.array of zeroes
        if done:
            # next_state_observation = None
            next_state_observation = np.zeros(shape=(11,))
        else:
            next_state_observation = self.generate_state_observation()

        return next_state_observation, done


    # gui display functions _______________________________________
    def initialize_display(self):
        pg.init()
        self.clock = pg.time.Clock()
        self.displayScreen = pg.display.set_mode(size=(self.WIDTH * self.block_side, self.HEIGHT * self.block_side + self.block_side))
        pg.display.set_caption("Snaky")  # set window title

    def update_game_screen(self):
        self.displayScreen.fill(COLORS['BLACK'])
        if self.snakeIsAlive:
            self.apple.draw_block(self.displayScreen)
            self.snaky.draw_snake(self.displayScreen)

        else:  # is snake died, delay
            time.sleep(0.3)

        # display score
        self.show_score()

        pg.display.flip()
        self.clock.tick(350)

    def show_score(self): # displaying Score function
        score_font = pg.font.SysFont('times new roman', 30)  # creating font object my_font
        score_surface = score_font.render('Score : ' + str(self.score), True, COLORS['WHITE'])  # create the display surface object score_surface
        self.displayScreen.blit(score_surface, (0, self.HEIGHT * self.block_side))  # displaying text - blit will draw the text on screen
        pg.draw.rect(self.displayScreen, COLORS['WHITE'], (0, self.HEIGHT * self.block_side, self.WIDTH * self.block_side, 1))  # draw a thin line to seperate score from game



