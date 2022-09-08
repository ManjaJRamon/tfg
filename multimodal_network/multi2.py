#############################################

#   AUTHOR: José Ramón Manjavacas Maceiras
#   LAST MODIFICATION DATE: 1/09/2022

#############################################

########## IMPORTS ##########
import os

import argparse
from array import *

import numpy as np

from time import perf_counter

import torch
from torch.utils.data import SubsetRandomSampler

from torchvision import transforms

from tqdm.auto import tqdm

from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.models import DiehlAndCook2015v2
from bindsnet.network.monitors import Monitor
from bindsnet.network.network import load
from bindsnet.utils import get_square_assignments, get_square_weights
from sklearn.model_selection import StratifiedKFold

#############################
import matplotlib.pyplot as plt
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)

########## ARGUMENTS PARSING ##########

# Define the argument parser.
parser = argparse.ArgumentParser()

# Add the desired arguments, defining their type (and default value for numeric arguments).
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=0)
parser.add_argument("--update_steps", type=int, default=256)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--n_folds", type=int, default=6)

# Set the default values for boolean arguments.
parser.set_defaults(gpu=False)

# Parse the arguments and store them in a variable.
args = parser.parse_args()

# Store all arguments in variables.
seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
update_steps = args.update_steps
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
gpu = args.gpu
n_folds = args.n_folds

#######################################

# Number of iterations to update the graphs.
update_interval = update_steps * batch_size
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
########## DEVICE SELECTION ##########

# Check if the 'gpu' argument is true and if the GPU is available to work with.
if gpu and torch.cuda.is_available():
    # Fix the GPU as the device to use.
    device = torch.device("cuda")
else:
    # Fix the CPU as the device to use.
    device = torch.device("cpu")
    if gpu:
        print("Could not use CUDA")
        gpu = False

######################################

# It exists more methods for establishing the seed, but this way the seed is fixed for all devices.
torch.manual_seed(seed)

# Print information about what device is being used.
print("Running on Device = %s\n" % (device))

########## LOAD MNIST TRAIN DATASET ##########

train_dataset = MNIST(
    # The encoding applied to the input data.
    PoissonEncoder(time=time, dt=dt),
    None,
    # Folder where the dataset files are stored.
    root="./data",
    # Wheter to download the dataset (the download page usually fails, so using a local copy is recommended).
    download=False,
    # Wheter to use train or test files.
    train=True,
    # The transformation applied to the encoded input data.
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

##############################################

# Stratified K-Fold declaration. By shuffling, data is not split in folds sequentially.
skfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)



# The first argument for the 'split' function is only the number of training instances (that is why it is just an array of zeros).
# The second argument for the split are the target values of each train instance. The 'narrow' function is used so that the number of train instances can be changed as desired.
for fold, (train_indices, val_indices) in enumerate(skfold.split(np.zeros(n_train), torch.narrow(train_dataset.targets, 0, 0, n_train))):
   
    # Print the current fold number.
    print("*********** FOLD %s **********" % (fold + 1))

    if fold==0:

      w_audio = torch.load("1F/X_Audio_post.pt", map_location="cpu")
      w_imag = torch.load("1F/X_Imag_post.pt", map_location="cpu")

    elif fold==1:

      w_audio = torch.load("2F/X_Audio_post.pt", map_location="cpu")
      w_imag = torch.load("2F/X_Imag_post.pt", map_location="cpu")


    ########## NETWORK CREATION ##########

    # Build network using the already created Dielh&Cook model.
    # More parameters can be set. For more information about them and their meaning, visit the model declaration.
      
    network = DiehlAndCook2015v2(
        w_imag, 
        w_audio,
        n_inpt=1568,  # Number of input neurons.
        n_neurons=n_neurons,  # Number of excitatory neurons.
        # Initial values of the excitatory-inhibitory connection weights.
        exc=exc,
        # Initial values of the inhibitory-excitatory connection weights.
        inh=inh,
        dt=dt,  # Step of the simulation.
        norm=78.4,  # Weight normalization factor.
        # Pre and post synaptic update learning rates, respectively.
        nu=(1e-4, 1e-2),
        # On-spike increment membrane threshold potential.
        theta_plus=theta_plus,
        # Shape of the input tensors (images of 28 x 28 pixels).
        inpt_shape=(1, 56, 28),
    )
    print(network.layers)


    
    # Choose where to store the initial weights

    dirName = "networks/"+ "arquitectura_3/"+ "matrices_pesos/" + "nuevo_"+ str(n_neurons) + "N_" + str(batch_size) + "BS_" + str(
            n_epochs) + "E_" + str(n_folds) + "F/" + str(fold+1) + "F/" + "iniciales/"
    os.makedirs(dirName)
    pesos = network.connections[("X","Ae_imag")].w
    torch.save(pesos, dirName+"X_Imag_pre.pt")
    pesos2 = network.connections[("X","Ae_audio")].w
    torch.save(pesos2, dirName+"X_Audio_pre.pt")

      ######################################

    # Store the network in the GPU.
    if gpu:
        network.to("cuda")

    ########## EXCITATORY NEURONS ASSIGMENTS ##########

    # This section comes from the original code example.
    n_classes = 10
    assignments1 = -torch.ones(n_neurons, device=device)
    proportions1 = torch.zeros((n_neurons, n_classes), device=device)
    rates1 = torch.zeros((n_neurons, n_classes), device=device)
    
    assignments2 = -torch.ones(n_neurons, device=device)
    proportions2 = torch.zeros((n_neurons, n_classes), device=device)
    rates2 = torch.zeros((n_neurons, n_classes), device=device)

    ###################################################

    ########## SPIKES MONITORING ##########

    # They must be recorded regardless it is used or not. This section comes from the original code example.
    spikes = {} 
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
        )
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    spike_record1 = torch.zeros(
        (update_interval, int(time / dt), n_neurons), device=device)

    spike_record2 = torch.zeros(
        (update_interval, int(time / dt), n_neurons), device=device)

    #######################################

    ########## TRAINING ##########

    print("\nBegin training.\n")

    # Create a sampler to sample the train data into a dataloader.
    # A 'SubsetRandomSampler' is used because it can load data according to a set of indices.
    train_sampler = SubsetRandomSampler(train_indices)

    # Dataloder for the training data.
    train_dataloader = DataLoader(
        train_dataset,  # Dataset to load.
        batch_size=batch_size,  # Batch size to use.
        num_workers=n_workers,  # CPU threads to use.
        pin_memory=gpu,  # True if using the GPU.
        sampler=train_sampler  # Sampler to use.
    )
    
    ########## EPOCH LOOP ##########

    for epoch in range(n_epochs):
        # If it is the first fold, record the start execution time.
        if fold == 0:
            start = perf_counter()

        ########## LEARNING LOGIC ##########

        # This section comes from the original code example.
        # 'update_steps' indicates in which iteration the accuracy is checked. This value highly affects the accuracy results.
        # 'tqdm' is used to show the progress of the loop.
        count=0
        labels = []
        for step, batch in enumerate(tqdm(train_dataloader, desc="Batches processed")):
            # Get next input sample.

            inputs = {"X": batch["encoded_image"]}

            if gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            if (step % update_steps == 0 and step > 0):
                # Convert the array of labels into a tensor
                label_tensor = torch.tensor(labels, device=device)
                # Assign labels to excitatory layer neurons
                assignments1, proportions1, rates1 = assign_labels(
                    spikes=spike_record1,
                    labels=label_tensor,
                    n_labels=n_classes,
                    rates=rates1,
                )

                assignments2, proportions2, rates2 = assign_labels(
                    spikes=spike_record2,
                    labels=label_tensor,
                    n_labels=n_classes,
                    rates=rates2,
                )

                labels = []

            labels.extend(batch["label"].tolist())

            # Run the network on the input.

            network.run(inputs=inputs, time=time, input_time_dim=1)

            # Add to spikes recording for both excitatory layers

            s1 = spikes["Ae_imag"].get("s").permute((1, 0, 2))

            s2 = spikes["Ae_audio"].get("s").permute((1, 0, 2))

            spike_record1[
                (step * batch_size)
                % update_interval: (step * batch_size % update_interval)
                + s1.size(0)
            ] = s1

            spike_record2[
                (step * batch_size)
                % update_interval: (step * batch_size % update_interval)
                + s2.size(0)
            ] = s2

            network.reset_state_variables()  # Reset state variables.

    ################################

    print("\nTraining complete.")

    # Choose where to store the final weights

    dirName = "networks/"+ "arquitectura_3/"+ "matrices_pesos/" +"nuevo_"+  str(n_neurons) + "N_" + str(batch_size) + "BS_" + str(
            n_epochs) + "E_" + str(n_folds) + "F/" + str(fold+1) + "F/" + "finales/"
    os.makedirs(dirName)
    pesos = network.connections[("X","Ae_imag")].w
    torch.save(pesos, dirName+"X_Imag_post.pt")
    pesos2 = network.connections[("X","Ae_audio")].w
    torch.save(pesos2, dirName+"X_Audio_post.pt")


    # Choose where to store the network files
    dirName = "networks/"+ "arquitectura_3/"+ "matrices_pesos/" + "nuevo_"+ str(n_neurons) + "N_" + str(batch_size) + "BS_" + str(
            n_epochs) + "E_" + str(n_folds) + "F/" + str(fold+1) + "F/" + "caract_red/"
    os.makedirs(dirName)
    # Save the network file.
    network.save(dirName + "network.pt")
    # Save the assigments file.
    torch.save(assignments1, dirName + "assignments_image.pt")
    torch.save(assignments2, dirName + "assignments_audio.pt")
    # Save the proportions file.
    torch.save(proportions1, dirName + "proportions_image.pt")
    torch.save(proportions2, dirName + "proportions_audio.pt")



    ########## EVALUATION TRAIN SET ##########

    # Train set accuracies
    accuracy1 = {"all": 0, "proportion": 0}

    accuracy2 = {"all": 0, "proportion": 0}

    print("\nBegin evaluation train set.\n")

    # Deactivate training (learning).
    network.train(mode=False)

    # Number of training samples.
    folds_train_samples = len(train_indices)

    ########## INFERENCE LOGIC ##########

    # This section comes from the original code example.
    # 'tqdm' is used to show the progress of the loop.

    suma = 0

    ##########################################

    print("\nEvaluation train set complete.\n")

    ##########################################

    print("Evaluation validation set complete.\n")

    ###############################################

    ########## EVALUATION TEST SET ##########

    # Load MNIST test data
    test_dataset = MNIST(
        # The encoding applied to the input data.
        PoissonEncoder(time=time, dt=dt),
        None,
        # Folder where the dataset files are stored.
        root="./data",
        # Wheter to download the dataset (the download page usually fails, so using a local copy is recommended).
        download=False,
        # Wheter to use train or test files.
        train=False,
        # The transformation applied to the encoded input data.
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )

    # Dataloder for the test data.
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,  # Batch size to use.
        num_workers=n_workers,  # CPU threads to use.
        pin_memory=gpu,  # True if using the GPU.
    )

    # Test set accuracies
    accuracy1 = {"all": 0}
    accuracy2 = {"all": 0}
    precisiones = {"all": 0}

    print("Begin evaluation test set.\n")

    # Deactivate training (learning).
    network.train(mode=False)

    ########## INFERENCE LOGIC ##########

    # This section comes from the original code example.
    # 'tqdm' is used to show the progress of the loop.

    cont = 0

    first = 0
    pred = {"labels": 0}
    all_pred1 = {"labels": 0}
    all_pred2 ={"labels": 0}
    labels = {"values":0}


    for step, batch in enumerate(tqdm(test_dataloader, desc="Batches processed")):
        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Add to spikes recording for both excitatory layers
        spike_record1 = spikes["Ae_imag"].get("s").permute((1, 0, 2))
        spike_record2 = spikes["Ae_audio"].get("s").permute((1, 0, 2))


        # Convert the array of labels into a tensor.
        label_tensor = torch.tensor(batch["label"], device=device)

        # Get network unimodal predictions for audio and image

        all_activity_pred1 = all_activity(
            spikes=spike_record1, assignments=assignments1, n_labels=n_classes
        )
        proportion_pred1 = proportion_weighting(
            spikes=spike_record1,
            assignments=assignments1,
            proportions=proportions1,
            n_labels=n_classes,
        )

        all_activity_pred2 = all_activity(
            spikes=spike_record2, assignments=assignments2, n_labels=n_classes
        )
        proportion_pred2 = proportion_weighting(
            spikes=spike_record2,
            assignments=assignments2,
            proportions=proportions2,
            n_labels=n_classes,
        )


        # Compute network accuracy according to available classification strategies.
        accuracy1["all"] += float(torch.sum(label_tensor.long()
                                 == all_activity_pred1).item())


        accuracy2["all"] += float(torch.sum(label_tensor.long()
                                 == all_activity_pred2).item())

        n_samples = spike_record1.size(0) # Number of samples in batch size

        spike1 = torch.zeros(
        (n_samples, n_neurons), device=device)
        spike2 = torch.zeros(
        (n_samples, n_neurons), device=device)

        spikes_record_total = torch.zeros(n_samples, n_neurons*2, device=device)


        # Sum through time dimension for both unimodal entities
        spike1 = spike_record1.sum(1)
        spike2 = spike_record2.sum(1)

        spikes_record_total_numpy= np.concatenate((spike1.cpu().numpy(), spike2.cpu().numpy()),axis=1)


        # Converting the matrix into tensor format
        spikes_record_total_x = torch.from_numpy(spikes_record_total_numpy)
        assignments_total_numpy = np.append(assignments1.cpu().numpy(), assignments2.cpu().numpy())
        assignments_total_x = torch.from_numpy(assignments_total_numpy)
        
        #Mapping the multimodal matrix to cuda device
        spikes_record_total = spikes_record_total_x.cuda()
        assignments_total = assignments_total_x.cuda()


        # Variable for getting global predictions of the networks
        rates = torch.zeros((n_samples, 10), device=spikes_record_total.device)

        for i in range(10): 
              # For each label count the number of neurons with this label assignment.
           n_assigns = torch.sum(assignments_total == i).float()
           if n_assigns > 0:
                  # Get indices of samples with this label.
                  indices = torch.nonzero(assignments_total == i).view(-1)

                  # Compute layer-wise firing rate for this label.
                  rates[:, i] = torch.sum(spikes_record_total[:, indices], 1) / n_assigns

        # Predictions are arg-max of layer-wise firing rates.
        predicciones = torch.sort(rates, dim=1, descending=True)[1][:, 0]
        
        if first==0:

          pred["labels"] = predicciones
          all_pred1["labels"] = all_activity_pred1
          all_pred2["labels"] = all_activity_pred2
          labels["values"] = label_tensor
          first+=1


        elif first==1:

          pred["labels"] = np.append(pred["labels"].cpu(), predicciones.cpu().numpy())
          all_pred1["labels"] = np.append(all_pred1["labels"].cpu(), all_activity_pred1.cpu().numpy())
          all_pred2["labels"] = np.append(all_pred2["labels"].cpu(), all_activity_pred2.cpu().numpy())
          labels["values"] = np.append(labels["values"].cpu(), label_tensor.cpu().numpy())
          first+=2

        else : 

          pred["labels"] = np.append(pred["labels"], predicciones.cpu().numpy())
          all_pred1["labels"] = np.append(all_pred1["labels"], all_activity_pred1.cpu().numpy())
          all_pred2["labels"] = np.append(all_pred2["labels"], all_activity_pred2.cpu().numpy())
          labels["values"] = np.append(labels["values"], label_tensor.cpu().numpy())

        network.reset_state_variables()  # Reset state variables.
        # Compute network accuracy according to available classification strategies.
      
    #####################################
  
        precisiones["all"]+=float(torch.sum(label_tensor.long()== predicciones).item())

    all_mean_accuracy1 = round((accuracy1["all"] / n_test) * 100, 2)

        # Print accuracies
    print("\nAll accuracy test set imagen: %.2f" % (all_mean_accuracy1))

    all_mean_accuracy2 = round((accuracy2["all"] / n_test) * 100, 2)
    
    # Print accuracies
    print("\nAll accuracy test set audio: %.2f" % (all_mean_accuracy2)) 
    
    precision = round((precisiones["all"]/n_test)*100,2)
    print("\nAll accuracy test set global: %.2f" % (precision))
    

    dirName = "networks/" + "arquitectura_3/"+ "matrices_pesos/" + "nuevo_"+ str(n_neurons) + "N_" + str(batch_size) + "BS_" + str(
          n_epochs) + "E_" + str(n_folds) + "F/" + str(fold+1) + "F/" + "predictions/"

    os.makedirs(dirName)

    # Store the labels predicted for audio, image and multimodal prediction. Also store the actual labels expected for the samples 
    torch.save(torch.tensor(pred["labels"]), dirName+"predicciones.pt")
    torch.save(torch.tensor(all_pred1["labels"]), dirName+"imagen_pred.pt")
    torch.save(torch.tensor(all_pred2["labels"]), dirName+"audio_pred.pt")
    torch.save(torch.tensor(labels["values"]), dirName+"etiquetas_muestras.pt")

    ##########################################

    print("Evaluation test set complete.\n")

    #########################################