import pygame
import sys
import random
from typing import Union
from enum import IntEnum
import os

pygame.init()
save_dir = os.path.dirname(os.path.realpath(__file__))

class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        self.index = 0
        self.counter = 0
        self.velocity = 0
        for num in range(1,4):
            img = pygame.image.load(f'{save_dir}\\img\\bird{num}.png')
            self.images.append(img)
        self.image = self.images[self.index]    
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.last_action = None

    def update(self, action, start_game, game_over):
        if start_game == True:
            if action == render_game.Actions.IDLE:
                self.velocity += 0.5
                if self.velocity > 8:
                    self.velocity = 8
                if self.rect.bottom < 763:
                    self.rect.y += int(self.velocity)
                self.last_action = action

        if game_over == False:            
            if action == render_game.Actions.FLAP:
                self.velocity = -10
                self.rect.y += int(self.velocity)
                self.last_action = action
            if action == render_game.Actions.CONST:
                self.velocity = 0
                self.rect.y += int(self.velocity)
                self.last_action = action

            self.counter += 1
            self.flap_cooldown = 7
            if self.counter > self.flap_cooldown:
                self.counter = 0
                self.index += 1
                if self.index == 2:
                    self.index = 0
                self.image = self.images[self.index]
            
            #rotate the bird
            self.image = pygame.transform.rotate(self.images[self.index], self.velocity * -2)
        else:
            self.image = pygame.transform.rotate(self.images[self.index], -90)
        

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(f'{save_dir}\\img\\pipe.png')
        self.rect = self.image.get_rect()
        
        #position 1 means from the top and -1 means from the down
        if position == 1:
            self.image = pygame.transform.flip(self.image,False,True)
            self.rect.bottomleft = [x, y-65]
        elif  position == -1:   
            self.rect.topleft = [x, y+65]
    
    def update(self,scroll_speed):
        self.rect.x -= scroll_speed
        self.pipe_x = self.rect.x
        if self.rect.right < 0:
            self.kill()


class render_game():

    def __init__(self, screen_width, screen_height):
        self.screen = None
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.surface = pygame.Surface((self.SCREEN_WIDTH,self.SCREEN_HEIGHT))
        self.SCORE_FONT_TYPE = pygame.font.SysFont('Bauhaus 93',50)
        self.SCORE_FONT_COLOR = (255,255,255)
        self.CLOCK = pygame.time.Clock()
        self.FBS = 120
        self.BACKGROUND = pygame.image.load(f'{save_dir}\\img\\bg.png')
        self.GROUND_IMAGE = pygame.image.load(f'{save_dir}\\img\\ground.png')
        self.start_game = False
        self.game_over = False
        self.alive = True
        self.pipe_frequency = 1500 #much of time for pipes to show every 1.5 seconds
        self.previous_time = pygame.time.get_ticks() - self.pipe_frequency
        self.score = 0
        self.passed_pipe = False
        self.birds_group = pygame.sprite.Group()
        self.pipe_group = pygame.sprite.Group()
        self.flappy = Bird(100, int(self.SCREEN_HEIGHT/2))
        self.birds_group.add(self.flappy)
        self.scroll_speed = 3
        self.ground_scroll = 0
        
    class Actions(IntEnum):
        """ Possible actions for the player to take. """
        IDLE, FLAP, CONST = 0, 1, 2

    def reset_game(self):
        self.pipe_group.empty()
        self.flappy.rect.x = 100
        self.flappy.rect.y = int(self.SCREEN_HEIGHT / 2)
        self.score = 0
        self.scroll_speed = 3
        self.pipe_frequency = 1500
        self.passed_pipe = False
        self.start_game == False 
        self.game_over == False
        self.alive = True
        return self.score, self.scroll_speed, self.pipe_frequency

    def draw_text(self,text, font, text_color):
        img = font.render(f'Reward: {text}',True,text_color)
        return img
    
    def make_display(self,set_mode,show_score):
        
        if set_mode:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH,self.SCREEN_HEIGHT))
            pygame.display.set_caption('Flappy bird')
        else:
            pass
        self.surface.blit(self.BACKGROUND,(0,0))
        
        self.pipe_group.draw(self.surface)
        self.surface.blit(self.GROUND_IMAGE,(0,768))
        
        self.birds_group.draw(self.surface)
        if show_score:
            score_text = self.draw_text(str(self.score), self.SCORE_FONT_TYPE, self.SCORE_FONT_COLOR)
            self.surface.blit(score_text, (int(self.SCREEN_WIDTH/1.4), 20))
        
    
    def update_display(self, draw):
        if draw:
            self.screen.blit(self.surface, [0, 0])
            pygame.display.update()
        else:
            return self.surface
        
    def update_state(self, action: Union[Actions, int]):
        
        if self.start_game == False and self.game_over == False:
            self.start_game = True
            self.alive = True
            return self.alive

        finish = False
        self.CLOCK.tick(self.FBS)
        #check the score
        if len(self.pipe_group) > 0:
            if self.birds_group.sprites()[0].rect.left > self.pipe_group.sprites()[0].rect.left and self.birds_group.sprites()[0].rect.right < self.pipe_group.sprites()[0].rect.right\
                and self.passed_pipe == False:
                self.passed_pipe = True
                
            if self.passed_pipe == True:
                if self.birds_group.sprites()[0].rect.left > self.pipe_group.sprites()[0].rect.right:
                    self.score+=1
                    self.passed_pipe = False

        if self.start_game == True and self.game_over == False:
            self.alive = True
            self.ground_scroll -= self.scroll_speed
            self.surface.blit(self.GROUND_IMAGE,(self.ground_scroll,768))
            #generate new pipes 
            time_now = pygame.time.get_ticks()
            if time_now - self.previous_time > self.pipe_frequency:
                pipe_gap = random.randint(-130,130)
                btm_green_pipe = Pipe(self.SCREEN_WIDTH, int(self.SCREEN_HEIGHT/2)+pipe_gap,-1)
                top_green_pipe = Pipe(self.SCREEN_WIDTH, int(self.SCREEN_HEIGHT/2)+pipe_gap,1)
                self.pipe_group.add(btm_green_pipe)
                self.pipe_group.add(top_green_pipe)
                self.previous_time = time_now
            if abs(self.ground_scroll) > 35:
                self.ground_scroll = 0
 
            self.pipe_group.update(self.scroll_speed)
                

        if self.game_over == True:
            self.game_over = False
            self.start_game = True
            
        #if bird hit the pipe or the bird hit the ground
        if pygame.sprite.groupcollide(self.birds_group,self.pipe_group, False, False) or self.flappy.rect.top < 0 or self.flappy.rect.bottom >= 763:
            self.game_over = True
            self.score, self.scroll_speed, self.pipe_frequency = self.reset_game()
            self.alive = False
            

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finish =  True
                pygame.quit()
                sys.exit(0)

            if finish:
                break
            
        self.birds_group.update(action,self.start_game,self.game_over)
        return self.alive

