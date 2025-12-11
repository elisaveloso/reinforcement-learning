import torch
import numpy as np
from collections import OrderedDict
import copy
import json
import matplotlib.pyplot as plt

from dataset import get_client_loader
from client import FlowerClient
from rl_agent import DQNAgent
from model import SimpleCNN

# Configurações
NUM_CLIENTS = 10
NUM_ROUNDS = 100
BATCH_SIZE = 16

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def aggregate_parameters(results):
    # results: list of (parameters, num_examples)
    total_examples = sum([num_examples for _, num_examples in results])
    
    # Initialize aggregated parameters with the first result
    first_params, _ = results[0]
    aggregated_params = [np.zeros_like(p) for p in first_params]
    
    for params, num_examples in results:
        weight = num_examples / total_examples
        for i, p in enumerate(params):
            aggregated_params[i] += p * weight
            
    return aggregated_params

def main():
    print("Iniciando simulação Manual de FL com RL...")
    
    # Inicializar Modelo Global
    global_model = SimpleCNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    
    # Inicializar Agente RL
    # Estado: [Round, Last Accuracy]
    # Ação: Escolher entre 1 a NUM_CLIENTS
    rl_agent = DQNAgent(state_dim=2, action_dim=NUM_CLIENTS)
    
    last_accuracy = 0.0
    
    # Histórico para plotagem
    history = {
        "rounds": [],
        "accuracy": [],
        "loss": [],
        "num_clients": [],
        "rewards": []
    }
    
    # Carregar loaders para todos os clientes (ou carregar sob demanda para economizar memória)
    # Vamos carregar sob demanda dentro do loop
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_num} ---")
        
        # 1. RL Agent escolhe ação
        state = [round_num, last_accuracy]
        action_idx = rl_agent.act(state)
        num_clients_to_sample = action_idx + 1
        
        print(f"[RL Agent] State={state}, Action={action_idx} (Sampling {num_clients_to_sample} clients)")
        
        # Selecionar clientes aleatoriamente
        client_indices = np.random.choice(range(NUM_CLIENTS), num_clients_to_sample, replace=False)
        print(f"Clientes selecionados: {client_indices}")
        
        # 2. Treinamento nos clientes
        results = []
        accuracies = []
        losses = []
        
        global_params = get_parameters(global_model)
        
        for cid in client_indices:
            # Carregar dados do cliente
            trainloader, valloader, _ = get_client_loader(int(cid), NUM_CLIENTS, BATCH_SIZE)
            
            # Instanciar cliente
            client = FlowerClient(trainloader, valloader)
            
            # Treinar
            # fit retorna: parameters, num_examples, metrics
            updated_params, num_examples, _ = client.fit(global_params, config={})
            
            # Avaliar
            # evaluate retorna: loss, num_examples, metrics
            loss, _, metrics = client.evaluate(updated_params, config={})
            accuracy = metrics["accuracy"]
            
            results.append((updated_params, num_examples))
            accuracies.append(accuracy)
            losses.append(loss)
            
            print(f"Client {cid}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
            
        # 3. Agregar Parâmetros
        new_global_params = aggregate_parameters(results)
        set_parameters(global_model, new_global_params)
        
        # 4. Calcular Acurácia Média (Global)
        # Em FL real, avaliaríamos o modelo global em um conjunto de teste separado ou agregando validações
        # Aqui usamos a média das acurácias dos clientes participantes (aproximação)
        current_accuracy = sum(accuracies) / len(accuracies)
        current_loss = sum(losses) / len(losses)
        print(f"Round {round_num} Aggregated Accuracy: {current_accuracy:.4f}")
        
        # 5. Atualizar Agente RL
        # Recompensa: Acurácia - Custo
        reward = current_accuracy - (0.01 * num_clients_to_sample)
        
        # Salvar histórico
        history["rounds"].append(round_num)
        history["accuracy"].append(current_accuracy)
        history["loss"].append(current_loss)
        history["num_clients"].append(int(num_clients_to_sample))
        history["rewards"].append(reward)
        
        next_state = [round_num + 1, current_accuracy]
        done = round_num == NUM_ROUNDS
        
        rl_agent.remember(state, action_idx, reward, next_state, done)
        rl_agent.replay(batch_size=32)
        
        last_accuracy = current_accuracy
        
    print("\nTreinamento Finalizado.")
    
    # Salvar resultados em JSON
    with open('simple_fl_rl/results.json', 'w') as f:
        json.dump(history, f)
    print("Resultados salvos em simple_fl_rl/results.json")
    
    # Gerar Plots
    plot_results(history)

def plot_results(history):
    rounds = history["rounds"]
    
    plt.figure(figsize=(15, 5))
    
    # Plot Acurácia
    plt.subplot(1, 3, 1)
    plt.plot(rounds, history["accuracy"], marker='o', label='Accuracy')
    plt.title('Accuracy per Round')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot Número de Clientes (Ações)
    plt.subplot(1, 3, 2)
    plt.plot(rounds, history["num_clients"], marker='s', color='orange', label='Num Clients')
    plt.title('RL Agent Actions (Num Clients)')
    plt.xlabel('Round')
    plt.ylabel('Number of Clients')
    plt.yticks(range(1, NUM_CLIENTS + 1))
    plt.grid(True)
    
    # Plot Recompensa
    plt.subplot(1, 3, 3)
    plt.plot(rounds, history["rewards"], marker='^', color='green', label='Reward')
    plt.title('RL Agent Reward')
    plt.xlabel('Round')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_fl_rl/results_plot.png')
    print("Gráfico salvo em simple_fl_rl/results_plot.png")

if __name__ == "__main__":
    main()
