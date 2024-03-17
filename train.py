import torch
from torch.optim.lr_scheduler import StepLR
from CNN_Model import ASLClassifier
from dataset_creation import device_check, create_dataset, split_dataset
from torch.nn import CrossEntropyLoss
import time
import os
import tqdm as tqdm
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
# Set the environment variable for PyTorch CUDA memory allocation configuration
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, checkpoint_path):
    torch.cuda.empty_cache()
    try:
        # Define the profiler configuration
        profiler_config = {
            "use_cuda": True,
            "profile_memory": True,  # Optionally profile memory usage
            "record_shapes": True,  # Optionally record the shapes of tensors
        }
        # Create a context manager to profile the training process
        with profile(**profiler_config) as prof:
            model.to("cuda")
            best_valid_loss = float('inf')
            valid_losses = []
            train_losses = []
            train_accuracies = []
            valid_accuracies = []
            total_start_time = time.time()

            for epoch in range(num_epochs):
                print("Inside epoch {}".format(epoch))
                model.train(True)
                running_loss = 0.0
                train_corrects = 0.0
                total_predictions = 0.0
                start_time = time.time()
                for i, (images, labels) in enumerate(train_loader):
                    images = images.cuda()
                    labels = labels.cuda()
                    # print(f"labels of the {i} batch: {labels}")
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    print(f"Loss: {loss.item()} for batch {i}")
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += labels.size(0)
                    train_corrects += (predicted == labels).sum().item()

                epoch_train_loss = running_loss / len(train_loader.dataset)
                train_losses.append(epoch_train_loss)
                train_accuracy = train_corrects / len(train_loader.dataset)
                train_accuracies.append(train_accuracy)
                # scheduler.step()

                model.eval()
                with torch.no_grad():
                    print("Now in validation mode...")
                    valid_loss = 0.0
                    correct_val_predictions = 0.0
                    total_val_predictions = 0.0
                    for i, (images, labels) in enumerate(valid_loader):
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        valid_loss += criterion(outputs, labels).item() * images.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        total_val_predictions += labels.size(0)
                        correct_val_predictions += (predicted == labels).sum().item()

                # Printing training statistics
                epoch_val_loss = valid_loss / len(valid_loader.dataset)
                valid_losses.append(epoch_val_loss)
                epoch_accuracy = correct_val_predictions / len(valid_loader.dataset)
                valid_accuracies.append(epoch_accuracy)
                end_time = time.time()
                print(f'Epoch {epoch + 1}/{num_epochs}, Time taken:{(end_time - start_time) / 60.0:.4f} minutes,'
                      f' Train Loss: {epoch_train_loss:.4f},'
                      f' Val Loss: {epoch_val_loss:.4f}, Validation Accuracy:{epoch_accuracy: .3f}')

                if epoch_val_loss < best_valid_loss:
                    best_valid_loss = epoch_val_loss
                    save_checkpoint(model, optimizer, epoch, best_valid_loss, device, checkpoint_path)

            total_end_time = time.time()
            total_time_taken = total_end_time - total_start_time
            print(f'Total Time Taken to train is {total_time_taken / 60:.2f} minutes')

            # For Plotting losses in each epoch
            plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
            plt.plot(range(1, num_epochs + 1), valid_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Train Loss vs Validation Loss')
            plt.legend()
            plt.show()

            # For Plotting Accuracies in each epoch
            plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
            plt.plot(range(1, num_epochs + 1), valid_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Train Accuracy vs Validation Accuracy')
            plt.legend()

            plt.show()

        # Print profiling results
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

        # Optionally, save the profiler output to a file
        prof.export_chrome_trace("profile_results.json")

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print('WARNING: Out of memory error detected. Trying to continue...')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise e

    except KeyboardInterrupt:
        print('WARNING: Interrupt Detected! Saving before closing...')
        save_checkpoint(model, optimizer, epoch, best_valid_loss, device, checkpoint_path)
        # print(device, checkpoint_path, epoch)
        print('Checkpoint saved. Quitting...')


def save_checkpoint(model, optimizer, epoch, best_valid_loss, device, checkpoint_path):
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_valid_loss': best_valid_loss,
        'device': device
    }, checkpoint_path)


if __name__ == '__main__':
    device_name = device_check()
    train, classnames = create_dataset()
    dictionary = split_dataset(train, device_name)
    train_loader = dictionary['train']
    valid_loader = dictionary['valid']

    model = ASLClassifier(num_classes=27, freeze_pretrained=True)  # Loading pretrained ResNet and freeze layers
    # model = ASLClassifier(num_classes=27)
    # model.to("cuda")  # Moving model to GPU if available
    criteron = CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.classifier_layers.parameters(),  # Only optimizing parameters of custom classifier layers
        # model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.001
    )
    # optimizer = torch.optim.Adam(model.classifier_layers.parameters(), lr=0.005)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    checkpoint_path = r'D:\Mediapipe ASL\MediaPipe ASL\model\checkpoints\checkpoint.pt'

    num_epochs = 10
    train_prompt = input("Do you want to train the model? yes/no: ")
    if train_prompt == "y" or train_prompt == "yes":
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            best_valid_loss = checkpoint['best_valid_loss']
            device = checkpoint['device']
            # model.to("cuda")
            print(f"Checkpoint loaded. Resuming training from epoch {epoch}.")
            remaining_epochs = num_epochs - epoch
            print(f"Remaining epochs: {remaining_epochs}")
            if remaining_epochs > 0:
                train_model(model, train_loader, valid_loader, criteron, optimizer, remaining_epochs, device_name,
                            checkpoint_path)
                # Saving final model
                final_model_path = r'D:\Mediapipe ASL\MediaPipe ASL\model\final_model.pt'
                torch.save(model.state_dict(), final_model_path)
            else:
                print(f"Training already completed for the {num_epochs} number  of epochs.")
        else:
            epoch = 0
            best_valid_loss = float('inf')
            print("Checkpoint not found. Starting training from scratch.")
            train_model(model, train_loader, valid_loader, criteron, optimizer, num_epochs, device_name, checkpoint_path)
            # Saving final model after training from scratch
            final_model_path = r'D:\Mediapipe ASL\MediaPipe ASL\model\final_model.pt'
            torch.save(model.state_dict(), final_model_path)

