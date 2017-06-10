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
    version           = 2               # useage internal to config
    prev_version      = 1
    board_size        = 5
    num_episodes_test = 5
    grad_clip         = True
    clip_val          = 1
    checkpoint_freq   = 500
    gamma             = 0.99            # still want to discount, because prefer to win quick...
    log_freq          = 50
    eval_freq         = 5000
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
    nsteps_train       = 100000
    batch_size         = 32
    buffer_size        = 500 * 8 # 8 times how long before we start learning, because we add 8 examples per self play step, by symmetries
    target_update_freq = 50
    lr_begin           = 0.001
    lr_end             = 0.0003
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.02
    eps_nsteps         = nsteps_train/2
    learning_start     = 500
    
