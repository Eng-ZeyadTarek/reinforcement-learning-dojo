import pygame
import sys
import pyautogui
import random

pygame.init()

class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        self.index = 0
        self.counter = 0
        self.velocity = 0
        for num in range(1,4):
            img = pygame.image.load(f'C:\\Users\\Zeyad Tarek\\Documents\\Reinforcement learning\\Game\\img\\bird{num}.png')
            self.images.append(img)
        self.image = self.images[self.index]    
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.pressed = False

    def update(self,start_game, game_over):
        if start_game == True:
            self.velocity += 0.5
            if self.velocity > 8:
                self.velocity = 8
            if self.rect.bottom < 763:
                self.rect.y += int(self.velocity)

        if game_over == False:
            key = pygame.key.get_pressed()
            if key[pygame.K_SPACE] == 1 and self.pressed == False:
                self.pressed = True 
                self.velocity = -10
            if key[pygame.K_SPACE]==0:
                self.pressed = False 

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
        self.image = pygame.image.load(f'C:\\Users\\Zeyad Tarek\\Documents\\Reinforcement learning\\Game\\img\\pipe.png')
        self.rect = self.image.get_rect()
        
        #position 1 means from the top and -1 means from the down
        if position == 1:
            self.image = pygame.transform.flip(self.image,False,True)
            self.rect.bottomleft = [x, y-85]
        elif  position == -1:   
            self.rect.topleft = [x, y+85]
    
    def update(self,scroll_speed):
        self.rect.x -= scroll_speed
        self.pipe_x = self.rect.x
        if self.rect.right < 0:
            self.kill()
            
    def return_value(self):
        print(self.rect.x, self.rect.y)
        return self.rect.y

class Restart():
    def __init__(self,x,y,img):
        self.img = img
        self.rect = self.img.get_rect()
        self.rect.topleft = [x,y]
    
    def draw(self,screen):
        action = False
        #draw button
        screen.blit(self.img, (self.rect.x,self.rect.y))

        #get mouse position 
        mouse_position = pygame.mouse.get_pos()

        #check if the mouse is over button
        if self.rect.collidepoint(mouse_position):
            if pygame.mouse.get_pressed()[0] == 1:
                action = True
         
        return action

class render_game():

    def __init__(self, screen_width, screen_height):

        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.SCORE_FONT_TYPE = pygame.font.SysFont('Bauhaus 93',50)

        self.SCORE_FONT_COLOR = (255,255,255)
        self.CLOCK = pygame.time.Clock()
        self.FBS = 60
        self.BACKGROUND = pygame.image.load('C:\\Users\\Zeyad Tarek\\Documents\\Reinforcement learning\\Game\\img\\bg.png')
        self.GROUND_IMAGE = pygame.image.load('C:\\Users\\Zeyad Tarek\\Documents\\Reinforcement learning\\Game\\img\\ground.png')
        self.start_game = False
        self.game_over = False
        self.pipe_frequency = 1500 #much of time for pipes to show every 1.5 seconds
        self.previous_time = pygame.time.get_ticks() - self.pipe_frequency
        self.score = 0
        self.passed_pipe = False
        self.restart_image = pygame.image.load('C:\\Users\\Zeyad Tarek\\Documents\\Reinforcement learning\\Game\\img\\restart.png')
        self.birds_group = pygame.sprite.Group()
        self.pipe_group = pygame.sprite.Group()
        self.restart = Restart(self.SCREEN_WIDTH // 2 -50, self.SCREEN_HEIGHT // 2- 100, self.restart_image)
        self.flappy = Bird(100, int(self.SCREEN_HEIGHT/2))
        self.birds_group.add(self.flappy)
        self.scroll_speed = 3
        self.ground_scroll = 0
        

    def reset_game(self):
        self.pipe_group.empty()
        self.flappy.rect.x = 100
        self.flappy.rect.y = int(self.SCREEN_HEIGHT / 2)
        self.score = 0
        self.scroll_speed = 3
        self.pipe_frequency = 1500
        self.passed_pipe = False
        return self.score, self.scroll_speed, self.pipe_frequency

    def draw_text(self,text, font, text_color):
        img = font.render(f'Reward: {text}',True,text_color)
        return img
    
    def renderer(self):
        
        run = True
        finish = False
        screen = pygame.display.set_mode((self.SCREEN_WIDTH,self.SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy bird')
        while run:
            self.CLOCK.tick(self.FBS)
            screen.blit(self.BACKGROUND,(0,0))
            self.pipe_group.draw(screen)

            screen.blit(self.GROUND_IMAGE,(0,768))
            self.birds_group.draw(screen)

            score_text = self.draw_text(str(self.score), self.SCORE_FONT_TYPE, self.SCORE_FONT_COLOR)
            screen.blit(score_text, (int(self.SCREEN_WIDTH/1.4), 20))

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
                
                self.ground_scroll -= self.scroll_speed
                screen.blit(self.GROUND_IMAGE,(self.ground_scroll,768))
                
                #generate new pipes 
                time_now = pygame.time.get_ticks()
                if time_now - self.previous_time > self.pipe_frequency:
                    pipe_gap = random.randint(-130,130)
                    btm_green_pipe = Pipe(self.SCREEN_WIDTH, int(self.SCREEN_HEIGHT/2)+pipe_gap,-1)
                    top_green_pipe = Pipe(self.SCREEN_WIDTH, int(self.SCREEN_HEIGHT/2)+pipe_gap,1)
                    self.pipe_group.add(btm_green_pipe)
                    self.pipe_group.add(top_green_pipe)
                    self.previous_time = time_now

                    print(btm_green_pipe.return_value())
                    print(top_green_pipe.return_value())
                if abs(self.ground_scroll) > 35:
                    self.ground_scroll = 0
 
                self.pipe_group.update(self.scroll_speed)
                

            #if bird hit the pipe
            if pygame.sprite.groupcollide(self.birds_group,self.pipe_group, False, False) or self.flappy.rect.top < 0:
                self.game_over = True
        
            #check if the bird hit the ground
            if self.flappy.rect.bottom >=763:
                self.game_over = True
                self.start_game = False
                
        
            if self.game_over == True and self.start_game == False:
                if self.restart.draw(screen):
                    self.game_over = False
                    self.start_game = False
                    self.score, self.scroll_speed, self.pipe_frequency = self.reset_game()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    finish =  True
                    pygame.quit()
                    break

                if event.type == pygame.KEYDOWN and self.start_game == False and self.game_over == False:
                    self.start_game = True

            if finish:
                break
            
            self.birds_group.update(self.start_game,self.game_over)
            pygame.display.update()

game = render_game(864, 936-70)

game.renderer()