import sys
import main
import importlib

part1Cache = None
if __name__ == "__main__":
    dataset = None
    pre = None
    while True:
        if not dataset:
            dataset, pre = main.main_load()
        main.main_train(dataset, pre)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        importlib.reload(main)