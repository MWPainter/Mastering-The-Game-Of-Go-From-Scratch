from linear_schedule import LinearExploration, LinearSchedule
from basic_network import BasicNetwork
from composite_network import CompositeNetwork
import policy.configs as pc

def get_schedules_from_config(config):
        exp_schedule = LinearExploration(config.eps_begin, config.eps_end, config.eps_nsteps)
        lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)
        return exp_schedule, lr_schedule



if __name__ == "__main__":
    train_first_network = False
    configs = [pc.bntconfig, ..., ...]

    config = configs[0]
    if train_first_network:
        # train a basic network
        exp_schedule, lr_schedule = get_schedule_from_config(config)
        model = BasicNetwork(config)
        model.run(exp_schedule, lr_schedule)
    
    # frozen path
    network_blackbox_filename = config.frozen_model_output
        
    for config in configs[1:]:
        # train composite network
        exp_schedule, lr_schedule = get_schedule_from_config(config)
        model = CompositeNetwork(network_blackbox_filename, config)
        model.run(exp_schedule, lr_schedule)
        network_blackbox_filename = config.frozen_model_output

