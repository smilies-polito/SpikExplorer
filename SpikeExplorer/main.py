import os
import json
import service.ax_manager


def main(input_config):
    ax_manager = service.ax_manager.AxManager(input_config)
    ax_manager.run_experiment()


if __name__ == "__main__":
    for filename in os.listdir('./input/'):
        with open(os.path.join('./input/', filename), 'r') as config:
            data = json.load(config)
            print(f'current working config: {data}')
            main(data)
