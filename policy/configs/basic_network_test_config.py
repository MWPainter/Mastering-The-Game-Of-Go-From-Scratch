class bntconfig():
    # env config
    render_train     = False
    render_test      = True
    overwrite_render = True
    record           = False

    # scoping
    scope        = "default"            # variable scope for network
    target_scope = "default_target"    	# variable scope for target network 

    # model and training config
    version           = 1               # useage internal to config
    prev_version      = 0
    board_size        = 5
    num_episodes_test = 2
    grad_clip         = True
    clip_val          = 10
    checkpoint_freq   = 500
    gamma             = 0.99            # still want to discount, because prefer to win quick...
    log_freq          = 50
    eval_freq         = 2000
    soft_epsilon      = 0

    # output config
    model_name              = str(board_size)+'x'+str(board_size)+'.v'+str(version)
    output_path             = "results/" + model_name + '/'
    model_output            = output_path + "weights/"
    model_checkpoint_output = output_path + "checkpoints/" 
    log_path                = output_path + "log.txt"
    graph_name              = "graph_save.pb"
    graph_path              = output_path + graph_name
    opponent_dir            = output_path + "opponents/"

    # hyper params
    nsteps_train       = 1000000
    batch_size         = 32
    buffer_size        = 500
    target_update_freq = 50
    lr_begin           = 0.0025
    lr_end             = 0.001
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = nsteps_train/2
    learning_start     = 200
    
