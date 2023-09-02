import json
import sys

import service.ax_manager


def main():
    path = open(sys.argv[0])
    config = json.load(path)
    ax_manager = service.ax_manager.AxManager(config)
    ax_manager.run_experiment()


if __name__ == "__main__":
    main()
