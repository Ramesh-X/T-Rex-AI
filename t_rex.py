from model import QModel
import pygame
import random
from pygame import *
import numpy as np
import cv2
import os

__author__ = "Ramesh Rathnayake"
__copyright__ = "Copyright 2018, RSoft"
__credits__ = ["Ramesh Rathnayake", "Shivam Shekhar"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Ramesh Rathnayake"
__email__ = "rsoft.ramesh@gmail.com"
__status__ = "Development"


scr_size = (width, height) = (600, 150)
gravity = 0.6

model = QModel(state_shape=(6,), action_size=3, batch_size=16)
pygame.init()


black = (0, 0, 0)
white = (255, 255, 255)
background_col = (235, 235, 235)

screen = pygame.display.set_mode(scr_size)
pygame.display.set_caption("T-Rex Rush")

jump_sound = pygame.mixer.Sound('sprites/jump.wav')
die_sound = pygame.mixer.Sound('sprites/die.wav')
checkPoint_sound = pygame.mixer.Sound('sprites/checkPoint.wav')


def load_image(
        name,
        sizex=-1,
        sizey=-1,
        colorkey=None,
):
    fullname = os.path.join('sprites', name)
    image = pygame.image.load(fullname)
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)

    if sizex != -1 or sizey != -1:
        image = pygame.transform.scale(image, (sizex, sizey))

    return image, image.get_rect()


def load_sprite_sheet(
        sheetname,
        nx,
        ny,
        scalex=-1,
        scaley=-1,
        colorkey=None,
):
    fullname = os.path.join('sprites', sheetname)
    sheet = pygame.image.load(fullname)
    sheet = sheet.convert()

    sheet_rect = sheet.get_rect()

    sprites = []

    sizex = sheet_rect.width / nx
    sizey = sheet_rect.height / ny

    for i in range(0, ny):
        for j in range(0, nx):
            rect = pygame.Rect((j * sizex, i * sizey, sizex, sizey))
            image = pygame.Surface(rect.size)
            image = image.convert()
            image.blit(sheet, (0, 0), rect)

            if colorkey is not None:
                if colorkey is -1:
                    colorkey = image.get_at((0, 0))
                image.set_colorkey(colorkey, RLEACCEL)

            if scalex != -1 or scaley != -1:
                image = pygame.transform.scale(image, (scalex, scaley))

            sprites.append(image)

    sprite_rect = sprites[0].get_rect()

    return sprites, sprite_rect


def disp_gameOver_msg(retbutton_image, gameover_image):
    retbutton_rect = retbutton_image.get_rect()
    retbutton_rect.centerx = width / 2
    retbutton_rect.top = height * 0.52

    gameover_rect = gameover_image.get_rect()
    gameover_rect.centerx = width / 2
    gameover_rect.centery = height * 0.35

    screen.blit(retbutton_image, retbutton_rect)
    screen.blit(gameover_image, gameover_rect)


def extractDigits(number):
    if number > -1:
        digits = []
        while number / 10 != 0:
            digits.append(number % 10)
            number = int(number / 10)

        digits.append(number % 10)
        for i in range(len(digits), 5):
            digits.append(0)
        digits.reverse()
        return digits


class Dino:
    def __init__(self, sizex=-1, sizey=-1):
        self.images, self.rect = load_sprite_sheet('dino.png', 5, 1, sizex, sizey, -1)
        self.images1, self.rect1 = load_sprite_sheet('dino_ducking.png', 2, 1, 59, sizey, -1)
        self.rect.bottom = int(0.98 * height)
        self.rect.left = width / 15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0, 0]
        self.jumpSpeed = 11.5

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def draw(self):
        screen.blit(self.image, self.rect)

    def get_state(self):
        return self.rect.left, self.rect.bottom

    def checkbounds(self):
        if self.rect.bottom > int(0.98 * height):
            self.rect.bottom = int(0.98 * height)
            self.isJumping = False

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + gravity

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1) % 2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1) % 2

        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2 + 2

        if self.isDead:
            self.index = 4

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[self.index % 2]
            self.rect.width = self.duck_pos_width

        self.rect = self.rect.move(self.movement)
        self.checkbounds()

        if not self.isDead and self.counter % 7 == 6 and not self.isBlinking:
            self.score += 1
            if self.score % 100 == 0 and self.score != 0:
                if pygame.mixer.get_init() is not None:
                    checkPoint_sound.play()

        self.counter = (self.counter + 1)


class Cactus(pygame.sprite.Sprite):
    def __init__(self, speed=5, sizex=-1, sizey=-1):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet('cacti-small.png', 3, 1, sizex, sizey, -1)
        self.rect.bottom = int(0.98 * height)
        self.rect.left = width + self.rect.width
        self.image_index = random.randrange(0, 3)
        self.image = self.images[self.image_index]
        self.movement = [-1 * speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)

        if self.rect.right < 0:
            self.kill()

    def get_state(self):
        return [self.rect.left, self.image_index]


class Ptera(pygame.sprite.Sprite):
    def __init__(self, speed=5, sizex=-1, sizey=-1):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet('ptera.png', 2, 1, sizex, sizey, -1)
        self.ptera_height = [height * 0.82, height * 0.75, height * 0.60]
        self.rect.centery = self.ptera_height[random.randrange(0, 3)]
        self.rect.left = width + self.rect.width
        self.image = self.images[0]
        self.movement = [-1 * speed, 0]
        self.index = 0
        self.counter = 0

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index + 1) % 2
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter = (self.counter + 1)
        if self.rect.right < 0:
            self.kill()

    def get_state(self):
        return [self.rect.left, self.rect.centery]


class Ground:
    def __init__(self, speed=-5):
        self.image, self.rect = load_image('ground.png', -1, -1, -1)
        self.image1, self.rect1 = load_image('ground.png', -1, -1, -1)
        self.rect.bottom = height
        self.rect1.bottom = height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        screen.blit(self.image, self.rect)
        screen.blit(self.image1, self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right

        if self.rect1.right < 0:
            self.rect1.left = self.rect.right


class Cloud(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image('cloud.png', int(90 * 30 / 42), 30, -1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1 * self.speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


class Scoreboard:
    def __init__(self, x=-1, y=-1):
        self.score = 0
        self.tempimages, self.temprect = load_sprite_sheet('numbers.png', 12, 1, 11, int(11 * 6 / 5), -1)
        self.image = pygame.Surface((55, int(11 * 6 / 5)))
        self.rect = self.image.get_rect()
        if x == -1:
            self.rect.left = width * 0.89
        else:
            self.rect.left = x
        if y == -1:
            self.rect.top = height * 0.1
        else:
            self.rect.top = y

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self, score):
        score_digits = extractDigits(score)
        self.image.fill(background_col)
        for s in score_digits:
            self.image.blit(self.tempimages[s], self.temprect)
            self.temprect.left += self.temprect.width
        self.temprect.left = 0


class T_Rex:

    def __init__(self):
        self.high_score = 0
        self.__init()

    def __init(self):
        self.gamespeed = 4
        self.gameOver = False
        self.gameQuit = False
        self.playerDino = Dino(44, 47)
        self.new_ground = Ground(-1 * self.gamespeed)
        self.scb = Scoreboard()
        self.highsc = Scoreboard(int(width * 0.78))
        self.counter = 0

        self.cacti = pygame.sprite.Group()
        self.pteras = pygame.sprite.Group()
        self.clouds = pygame.sprite.Group()
        self.last_obstacle = pygame.sprite.Group()

        Cactus.containers = self.cacti
        Ptera.containers = self.pteras
        Cloud.containers = self.clouds

        self.retbutton_image, self.retbutton_rect = load_image('replay_button.png', 35, 31, -1)
        self.gameover_image, self.gameover_rect = load_image('game_over.png', 190, 11, -1)

        self.temp_images, self.temp_rect = load_sprite_sheet('numbers.png', 12, 1, 11, int(11 * 6 / 5), -1)
        self.HI_image = pygame.Surface((22, int(11 * 6 / 5)))
        self.HI_rect = self.HI_image.get_rect()
        self.HI_image.fill(background_col)
        self.HI_image.blit(self.temp_images[10], self.temp_rect)
        self.temp_rect.left += self.temp_rect.width
        self.HI_image.blit(self.temp_images[11], self.temp_rect)
        self.HI_rect.top = height * 0.1
        self.HI_rect.left = width * 0.73

    def step(self, action):
        # im = None
        if pygame.display.get_surface() is None:
            print("Couldn't load display surface")
            self.gameQuit = True
            self.gameOver = True
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.gameQuit = True
                    self.gameOver = True
            if action == model.KEY_UP:
                if self.playerDino.rect.bottom == int(0.98 * height):
                    self.playerDino.isJumping = True
                    if pygame.mixer.get_init() is not None:
                        jump_sound.play()
                    self.playerDino.movement[1] = -1 * self.playerDino.jumpSpeed

            if action == model.KEY_DOWN:
                if not (self.playerDino.isJumping and self.playerDino.isDead):
                    self.playerDino.isDucking = True

            if action == model.KEY_NONE:
                self.playerDino.isDucking = False

        for c in self.cacti:
            c.movement[0] = -1 * self.gamespeed
            if pygame.sprite.collide_mask(self.playerDino, c):
                self.playerDino.isDead = True
                if pygame.mixer.get_init() is not None:
                    die_sound.play()

        for p in self.pteras:
            p.movement[0] = -1 * self.gamespeed
            if pygame.sprite.collide_mask(self.playerDino, p):
                self.playerDino.isDead = True
                if pygame.mixer.get_init() is not None:
                    die_sound.play()

        if len(self.cacti) < 2:
            if len(self.cacti) == 0:
                self.last_obstacle.empty()
                self.last_obstacle.add(Cactus(self.gamespeed, 40, 40))
            else:
                for l in self.last_obstacle:
                    if l.rect.right < width * 0.7 and random.randrange(0, 50) == 10:
                        self.last_obstacle.empty()
                        self.last_obstacle.add(Cactus(self.gamespeed, 40, 40))

        if len(self.pteras) == 0 and random.randrange(0, 200) == 10 and self.counter > 500:
            for l in self.last_obstacle:
                if l.rect.right < width * 0.8:
                    self.last_obstacle.empty()
                    self.last_obstacle.add(Ptera(self.gamespeed, 46, 40))

        if len(self.clouds) < 5 and random.randrange(0, 300) == 10:
            Cloud(width, random.randrange(height / 5, height / 2))

        self.playerDino.update()
        self.cacti.update()
        self.pteras.update()
        self.clouds.update()
        self.new_ground.update()
        self.scb.update(self.playerDino.score)
        self.highsc.update(self.high_score)

        if pygame.display.get_surface() is not None:
            screen.fill(background_col)
            self.new_ground.draw()
            self.clouds.draw(screen)
            self.cacti.draw(screen)
            self.pteras.draw(screen)
            self.playerDino.draw()
            # im = pygame.surfarray.array3d(screen)
            # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            # im = np.expand_dims(im, axis=2)
            self.scb.draw()
            if self.high_score != 0:
                self.highsc.draw()
                screen.blit(self.HI_image, self.HI_rect)

            pygame.display.update()

        if self.playerDino.isDead:
            self.gameOver = True
            if self.playerDino.score > self.high_score:
                self.high_score = self.playerDino.score

        if self.counter % 700 == 699:
            self.new_ground.speed -= 1
            self.gamespeed += 1

        self.counter = self.counter + 1
        dino_state = self.playerDino.get_state()
        cacti_state = [-1, -1]
        _cacti_states = []
        sprites = self.cacti.sprites()
        for sprite in sprites:
            _cacti_states.append(sprite.get_state())
        if len(_cacti_states) > 0:
            _cacti_states = sorted(_cacti_states)
            cacti_state = _cacti_states[0]
            if cacti_state[0] <= dino_state[0] and len(_cacti_states) > 1:
                cacti_state = _cacti_states[1]
        pteras_state = [-1, -1]
        sprites = self.pteras.sprites()
        if len(sprites) > 0:
            pteras_state = sprites[-1].get_state()
        state = np.array(cacti_state + pteras_state + [dino_state[1], self.gamespeed])
        reward = 1
        if self.playerDino.rect.bottom != int(0.98 * height) and action != model.KEY_NONE:
            reward = 0
        return state, reward, self.gameOver

    def reset(self):
        if self.gameOver:
            if pygame.display.get_surface() is None:
                print("Couldn't load display surface")
                self.gameQuit = True
                self.gameOver = False
            else:
                self.gameOver = False
            self.highsc.update(self.high_score)
            if pygame.display.get_surface() is not None:
                disp_gameOver_msg(self.retbutton_image, self.gameover_image)
                if self.high_score != 0:
                    self.highsc.draw()
                    screen.blit(self.HI_image, self.HI_rect)
                pygame.display.update()
                self.__init()
        ret = self.step(model.KEY_NONE)
        while ret[0] is None:
            ret = self.step(model.KEY_NONE)
        return ret


NUM_EPISODES = 1000
MAX_T = 1000000


def main():
    t_rex = T_Rex()
    for episode in range(NUM_EPISODES):
        state, _, _ = t_rex.reset()
        state = np.expand_dims(state, axis=0)

        for t in range(MAX_T):
            action, max_q = model.get_action(state)
            next_state, reward, done = t_rex.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            reward = reward if not done else -1000

            mem_size = model.remember(state, action, reward, next_state, done)
            model.train()

            # Print data
            print("\nEpisode = %d" % episode)
            print("t = %d" % t)
            print("Action: %d" % action)
            print("State: %s" % str(state))
            print("Reward: %f" % reward)
            print("Best Q: %f" % max_q)
            print("Memory: %d" % mem_size)

            print("")

            if done:
                print("Episode %d finished after %f time steps" % (episode, t))
                break
            state = next_state

            # sleep(0.25)


main()
