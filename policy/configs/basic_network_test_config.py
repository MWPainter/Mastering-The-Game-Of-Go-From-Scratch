class bntconfig():
    # env config
    render_train     = False
    render_test      = False
    overwrite_render = True
    record           = False

    # scoping
    scope        = "default"            # variable scope for network
    target_scope = "default_target"    	# variable scope for target network 

    # model and training config
    version           = 0               # useage internal to config
    prev_version      = 0
    board_size        = 3
    num_episodes_test = 10
    grad_clip         = True
    clip_val          = 1
    checkpoint_freq   = 5000
    gamma             = 0.99            # still want to discount, because prefer to win quick...
    log_freq          = 50
    eval_freq         = 2500
    soft_epsilon      = 0

    # output config
    output_path             = "results/"
    model_name              = str(board_size)+'x'+str(board_size)+'.v'+str(version)
    model_output            = output_path + model_name + ".weights/"
    model_checkpoint_output = output_path + "checkpoints/" + model_name + ".weights/"
    log_path                = output_path + "log.txt"

    # hyper params
    nsteps_train       = 30000
    batch_size         = 32
    buffer_size        = 500 * 8 # 8 times how long before we start learning, because we add 8 examples per self play step, by symmetries
    target_update_freq = 50
    lr_begin           = 0.001
    lr_end             = 0.0003
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = nsteps_train/2
    learning_start     = 500
    
