# Solving XOR Problem with Neural Networks 🤖

A simple Python neural network project demonstrating solutions to the XOR problem using NumPy .

## Project Description 📝

This project contains two types of neural network models:

* **OneLayerNN**: Single-layer neural network (logistic regression) . Cannot perfectly solve XOR.
* **TwoLayerNN**: Two-layer neural network (2 → 2 → 1) . Capable of learning the XOR function .

The goal is to demonstrate how a minimal neural network can learn non-linear relationships, such as XOR, which cannot be solved by simple linear models. This illustrates the advantage of multi-layer networks in capturing non-linear patterns compared to traditional linear regression or single-layer models .

<img width="1536" height="1024" alt="ChatGPT Image 19 Kas 2025 20_15_38" src="https://github.com/user-attachments/assets/97da3dda-8a27-4908-ac78-1422b440ae09" />


## Project Structure 📂

* ├─ src/
* │ ├─ main.py # Train and test models
* │ └─ NeuralNetworks/
* │   ├─ OneLayerNN.py # Single-layer neural network implementation
* │   └─ TwoLayerNN.py # Two-layer neural network implementation
* ├─ README.md 
* ├─ pyproject.toml 

---

## Installation 🛠️

Use Poetry to install dependencies:

```bash
poetry install
```

## Usage ▶️

```bash
python src/main.py
```

## Sample Output
- Expected Output = [0, 1, 1, 0]
- <img width="382" height="43" alt="image" src="https://github.com/user-attachments/assets/888e6dee-9d7f-4c36-8a95-16fcd0c769c8" />


python src/main.py
```

