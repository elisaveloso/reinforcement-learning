# Projeto Básico de Aprendizado Federado com Reinforcement Learning (DQN)

Este projeto implementa um sistema de Aprendizado Federado (Federated Learning) onde um servidor utiliza um agente de Aprendizado por Reforço (DQN) para selecionar dinamicamente o número de clientes participantes em cada rodada de treinamento.

## Estrutura

- `dataset.py`: Carrega e particiona o dataset de imagens (daninha vs nao_daninha).
- `model.py`: Define a arquitetura da Rede Neural Convolucional (CNN).
- `client.py`: Define o Cliente Flower que treina o modelo localmente.
- `rl_agent.py`: Implementa o agente DQN (Deep Q-Network).
- `main.py`: Configura a simulação, a estratégia customizada e inicia o treinamento.

## Como Executar

1. Certifique-se de estar no ambiente virtual correto:
   ```bash
   source ../venv/bin/activate
   ```

2. Instale as dependências (se ainda não estiverem instaladas):
   ```bash
   pip install flwr torch torchvision numpy
   ```

3. Execute a simulação:
   ```bash
   python main.py
   ```

## Funcionamento

- O projeto utiliza um loop customizado de Aprendizado Federado (sem usar o `flwr.simulation` para evitar overhead de memória em ambientes restritos).
- O **Agente RL** observa o estado atual (número da rodada e acurácia da rodada anterior).
- Ele escolhe uma **Ação**: quantos clientes (de 1 a 5) devem participar do treinamento nesta rodada.
- O **Loop Principal** seleciona os clientes aleatoriamente e coordena o treinamento:
    - Cada cliente treina localmente em sua partição de dados.
    - Os parâmetros são agregados pelo servidor (FedAvg).
    - A acurácia global é estimada pela média das acurácias dos clientes.
- O agente recebe uma **Recompensa** baseada na acurácia obtida menos um custo por cliente utilizado.
