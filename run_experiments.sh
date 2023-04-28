# Command for the simplified experiment with the reduced number of target classes and maximum episodes.
# python main.py -model_name VGG16 -max_episodes 10000 -max_step 1 -alpha 0 -n_classes 1000 -z_dim 100 -n_target 100

# If you want to experiment with the settings reported in the paper, please run the commands below.

# VGG16
python main.py -model_name VGG16 -max_episodes 40000 -max_step 1 -alpha 0 -n_classes 1000 -z_dim 100 -n_target 1000

# ResNet-152
# python main.py -model_name ResNet-152 -max_episodes 40000 -max_step 1 -alpha 0 -n_classes 1000 -z_dim 100 -n_target 1000

# Face.evoLVe
# python main.py -model_name Face.evoLVe -max_episodes 40000 -max_step 1 -alpha 0 -n_classes 1000 -z_dim 100 -n_target 1000
