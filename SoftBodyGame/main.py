from Libraries.game import *

if __name__ == "__main__":
    game = Game(screen_size=(1080, 720), scale=60, fps=60)
    game.run()