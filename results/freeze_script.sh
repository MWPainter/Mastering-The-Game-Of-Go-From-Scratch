# argument to the script is the directory of the saved model
python -m tensorflow.python.tools.freeze_graph --input_graph=$1/graph_save.pb --input_checkpoint=$1/checkpoints/$2 \
--output_graph=$1/frozen.pb --output_node_names=out
