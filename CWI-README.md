# Data Structure

## Small CNN Zoo
Was processed to fit the following data structure by Simon:

```
unthi_zoo/
├── unthi_cifar10/
│   ├── NN_unthi_cifar10_11169340_{model_id}/
│   │   ├── checkpoint_00000{0-8}/
│   │   │   └── checkpoints (pickle file)
│   │   ├── params.json
│   │   └── result.json
├── unthi_mnist/
│   ├── NN_unthi_mnist_11169340_{model_id}/
│   │   ├── checkpoint_00000{0-8}/
│   │   │   └── checkpoints (pickle file)
│   │   ├── params.json
│   │   └── result.json
├── unthi_fmnist/
│   ├── NN_unthi_fmnist_11169340_{model_id}/
│   │   ├── checkpoint_00000{0-8}/
│   │   │   └── checkpoints (pickle file)
│   │   ├── params.json
│   │   └── result.json
├── unthi_svhn/
│   ├── NN_unthi_svhn_11169340_{model_id}/
│   │   ├── checkpoint_00000{0-8}/
│   │   │   └── checkpoints (pickle file)
│   │   ├── params.json
│   │   └── result.json
```

There are 30,000 models per dataset, and each model has 9 checkpoints (epochs 0-8).

These are compressed and saved as zips in `/projects/prjs2156/shared/wsl/unthi_zoo/`, but are too big to all be stored there (too many files). The uncompressed versions are stored in `/scratch-shared/mmiddelkoop/unthi_zoo/`. However, these will be deleted if they are untouched for more than two weeks.

### TokenDatasets

## ResNet18-CIFAR100