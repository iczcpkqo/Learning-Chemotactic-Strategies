#!/usr/bin/python3

import argparse
import pygame
import numpy as np
from pygame.locals import *

def main():
        ## Process command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("-debug", help="debug level (default: %(default)s)",                      type=int,   default=0)
        parser.add_argument("-eat",   help="agent's particle absorption rate (default: %(default)s)", type=float, default=0.25)
        parser.add_argument("-ax",    help="agent width (default: %(default)s)",                      type=float, default=100.0)
        parser.add_argument("-ay",    help="agent height (default: %(default)s)",                     type=float, default=20.0)
        parser.add_argument("-ax0",   help="agent initial position down (default: %(default)s)",      type=float, default=250.0)
        parser.add_argument("-ay0",   help="agent initial position right (default: %(default)s)",     type=float, default=250.0)
        parser.add_argument("-avel",  help="agent chase velocity (default: %(default)s)",             type=float, default=5.0)
        parser.add_argument("-sx",    help="source offset left from LR (default: %(default)s)",       type=float, default=100.0)
        parser.add_argument("-sy",    help="source offset up from LR (default: %(default)s)",         type=float, default=100.0)
        parser.add_argument("-winx",  help="arena width (default: %(default)s)",                      type=int,   default=1000)
        parser.add_argument("-winy",  help="arena height (default: %(default)s)",                     type=int,   default=600)
        parser.add_argument("-emit",  help="particle emission rate (default: %(default)s)",           type=float, default=5.0)
        parser.add_argument("-brown", help="particle diffusion variance (default: %(default)s)",      type=float, default=5.0)
        parser.add_argument("-dt",    help="simulation timestep (default: %(default)s)",              type=float, default=0.3)
        parser.add_argument("-frame", help="display every i-th frame (default: %(default)s)",         type=int,   default=1)
        parser.add_argument("-p", "--pygame", help="use pygame or not", action="store_true")
        args = parser.parse_args()

        # debug level
        debug = args.debug

        if debug > 0: print(args)

        # absorption rate: chance of a particle being eaten when on agent for one unit of time
        eatRate = args.eat

        # size of arena
        win_x = args.winx
        win_y = args.winy

        # size of agent
        agent_x = args.ax
        agent_y = args.ay

        # expected number of particles released per unit time
        releaseRate = args.emit

        # time step of simulation
        dt = args.dt

        # particle diffusion variance per unit time
        diffusionVariance = args.brown

        # initial position of agent (offset from upper left corner)
        u = (args.ax0, args.ay0)

        # chase velocity of agent
        vr = args.avel

        # source of particles (offset from lower right corner)
        source = (args.sx,args.sy)

        frame = args.frame

        ## Update derived quantities
        source = (win_x - source[0] , win_y - source[1])

        if debug > 0: print("source:", source)

        ## Initialize random number generator
        rng = np.random.default_rng()

        ## Initialize display (game) engine
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((win_x,win_y))
        pygame.display.set_caption("Smelly the Chemotactic Agent")
        font = pygame.font.SysFont('Cambria',40)

        white = (255,255,255)
        red = (255,0,0)
        green = (0,255,0)
        blue = (0,0,255)

        ## Initialize temportal loop variables
        particles = []
        t = 0.0
        framei = 0
        for _ in range(10000):
                if framei == 0:
                        ## Time to die?
                        for event in pygame.event.get():
                                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                                        pygame.quit()

                        ## Run, run, as fast as you can. You can't catch me, I'm the Stinky Cheese Man!
                        ## So comment this out:
                        # pygame.time.delay(100)

                        screen.fill(white)

                        ## Draw the agent
                        (u_x,u_y) = u
                        pygame.draw.rect(screen, green, (int(u_x), int(u_y), int(agent_x), int(agent_y)))
                        # with blue racing stripes
                        pygame.draw.rect(screen, blue, (int(u_x + agent_x/2)-1, int(u_y), 2, int(agent_y)))
                        pygame.draw.rect(screen, blue, (int(u_x), int(u_y + agent_y/2)-1, int(agent_x), 2))

                        ## Draw the odor source
                        pygame.draw.circle(screen, blue, intPair(source), 5)

                ## Maybe particles are born:
                particles += [source]* rng.poisson(releaseRate * dt)

                if debug > 1:
                        print("particles =", particles)

                if debug > 0:
                        print("#particles:", len(particles))

                ## Initialize particle loop variables
                quads = [0,0,0,0]
                newParticles = []

                for p in particles:

                        # Update particle position
                        (p_x,p_y) = p = updatePosition(p, wind(p, t), diffusionVariance, dt, rng)

                        if framei == 0:
                                pygame.draw.circle(screen, red, intPair(p), 2)

                        i = whatQuad(
                                p_x - (u_x + agent_x/2),
                                p_y - (u_y + agent_y/2),
                                agent_x/2, agent_y/2
                        )
                        if debug > 2:
                                print("agent", u, "particle", p, "quad", i)
                        if i >= 0:
                                quads[i] += 1

                        if p_y < 0 or p_y > win_y or p_x < 0 or p_x > win_x:
                                if debug > 2:
                                        print("Not adding: particle at", p, "off screen")
                        elif eatRate == 0 or i < 0 or rng.exponential(1/eatRate) >= dt:
                                newParticles.append(p)
                        else:
                                if debug > 2:
                                        print("Not adding: particle at", p, "quad", i, "eaten")

                particles = newParticles

                # Calculate the new coordinates of the agent
                (u_x,u_y) = updatePosition(u,calcv(quads, vr),0.0,dt,rng)

                if debug > 0:
                        if u_x < 0 or u_y < 0 or u_x + agent_x > win_x or u_y + agent_y > win_y:
                                print("Agent out of bounds")

                # Keep agent in bounds
                u_x = np.clip(0.0, u_x, win_x - agent_x)
                u_y = np.clip(0.0, u_y, win_y - agent_y)

                u = (u_x, u_y)

                if framei == 0:
                        # Update screen
                        screen.blit(font.render(' t = ' + str(round(t,3)), False, (0,0,0)), (0,0))
                        pygame.display.update()

                t += dt
                framei = np.mod(framei+1, frame)
        pygame.quit()

# a velocity based on differential odors (highly viscous medium)
def calcv(quads, vr):
        xdir = np.sign(- quads[0] + quads[1] - quads[2] + quads[3])
        ydir = np.sign(- quads[0] - quads[1] + quads[2] + quads[3])
        return scaleVect(vr/np.sqrt(2), (xdir, ydir))

def updatePosition(u,v,var,dt,rng):
        # integration position based on velocity, using a single Euler step of: d/dt position = velocity + noise
        return sumVect(u,
                       sumVect(scaleVect(dt, v),
                               scaleVect(np.sqrt(dt*var),
                                         (rng.standard_normal(), rng.standard_normal())
                                         )
                               )
                )

def whatQuad(x,y, maxx,maxy):
        if abs(x) > maxx or abs(y) > maxy or x == 0 or y == 0:
                return -1
        else:
                return int(x > 0) + 2*int(y > 0)

def wind(p,t):
        return -1.2, -0.3

def scaleVect(s, v):
        return (s*v[0], s*v[1])

def sumVect(u, v):
        return u[0]+v[0], u[1]+v[1]

def intPair(p):
        return int(p[0]),int(p[1])

if __name__ == "__main__":
    main()
