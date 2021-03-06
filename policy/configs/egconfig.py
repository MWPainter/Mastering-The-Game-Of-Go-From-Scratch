class config():
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
    board_size        = 9
    prev_board_size   = 5               # only if transfer learning is this needed
    prev_model_name   = 'results/'+str(prev_board_size)+'x' \
                        +str(prev_board_size)+'.v'+str(prev_version) \
                        +'.weights/'
    num_episodes_test = 10
    grad_clip         = True
    clip_val          = 10
    checkpoint_freq   = 5000
    gamma             = 0.99            # still want to discount, because prefer to win quick...
    log_freq          = 50
    eval_freq         = 5000
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
    buffer_size        = 500
    target_update_freq = 50
    lr_begin           = 0.00025
    lr_end             = 0.0001
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = nsteps_train/2
    learning_start     = 200
    
