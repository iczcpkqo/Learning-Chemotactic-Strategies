import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-debug", help="debug level (default: %(default)s)",                      type=int,   default=0)
parser.add_argument("-eat",   help="agent's particle absorption rate (default: %(default)s)", type=float, default=0.0)
parser.add_argument("-ax",    help="agent width (default: %(default)s)",                      type=float, default=50.0)
parser.add_argument("-ay",    help="agent height (default: %(default)s)",                     type=float, default=50.0)
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
parser.add_argument("-pygame",help="use pygame or not", action="store_true")
parser.add_argument("-pattern",help="pattern name",                                           type=str,  default='')
parser.add_argument("-useRl", help="use RL or not", action="store_true")
parser.add_argument("-timeStep", help="Max timeSteps in each episode",                        type=int,   default=1000)
parser.add_argument("-test", help="测试时 source固定", action="store_true")

args = parser.parse_known_args()[0]

import pprint

pprint.pprint(args.__dict__)