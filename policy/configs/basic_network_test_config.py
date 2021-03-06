# Config for a *basic* network
class bntconfig():
    # env config
    render_train     = False
    render_test      = True
    overwrite_render = True

    # scoping
    scope = "default"            # variable scope for network

    # model and training config
    version           = 666 # useage internal to config
    prev_version      = 1
    board_size        = 5
    num_episodes_test = 5 # will test 5 white and 5 black
    grad_clip         = True
    clip_val          = 1
    checkpoint_freq   = 500
    gamma             = 0.99            # still want to discount, because prefer to win quick...
    log_freq          = 50
    eval_freq         = 5000

    # SHould we restore from checkpoint? If not, set to -1
    checkpoint        = 853999

    # Network parameters
    num_layers = 7
    num_filters = 64

    # output config
    model_name              = str(board_size)+'x'+str(board_size)+'.v'+str(version)
    output_path             = "results/" + model_name + '/'
    model_output            = output_path + "weights"
    frozen_model_output     = output_path + "final_graph"
    model_checkpoint_output = output_path + "checkpoints/" 
    log_path                = output_path + "log.txt"
    graph_name              = "graph_save.pb"
    graph_path              = output_path + graph_name
    opponent_dir            = output_path + "opponents/"
    load_checkpoint_file    = model_checkpoint_output + str(checkpoint)

    # hyper params
    #nsteps_train       = 40000000
    nsteps_train       = 1500000
    batch_size         = 32
    buffer_size        = 500 * 8 # 8 times how long before we start learning, because we add 8 examples per self play step, by symmetries
    lr_begin           = 0.010
    lr_end             = 0.001
    lr_nsteps          = nsteps_train/2
    eps_begin          = 0.05
    eps_end            = 0.001
    eps_nsteps         = nsteps_train/2
    learning_start     = 500
    
