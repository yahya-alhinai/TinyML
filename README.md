# TinyML
 Ultra-efficient 1D neural network on ARM-STM32 MCU, achieving 97.5% accuracy with just 6ms latency in detecting life-threatening ventricular arrhythmias


# Requirements
No additional requirements added from the original repo.
- Python 3.6
- PyTorch 1.4.0
- torchtext 0.5.0
- numpy 1.18.1
- matplotlib 3.1.3
- tqdm 4.42.1
- scikit-learn 0.22.1


# How to run
In the model_training_repo folder, the follwing command are used to train, test and validate the model.

- `python3 training_save_deep_models.py` to train the model
- `python3 testing_performances.py` to test the model
- `python3 validation.py` to validate the model deployed on the MCU board
