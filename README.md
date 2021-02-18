# Contact Tracing Simulator

E' un'implementazione Python di un Simulatore SEIR stocastico a cui vengono integrati diversi tipi tracciamento digitale su un grafo dinamico che rappresenta una rete di prossimità in un contesto urbano.

## Istallazione

1. Installa un venv Python (in alternativa installa Python 3+)
   - Controlla se è presente pip

    ```bash  
    pip -h  
    ```
   - Installa virtualenv e crea un venv
    
    ```bash   
    pip install virtualenv
    virtualenv path_directory
    ```
   - Attiva il venv
   ```bash   
    source path_directory/bin/activate
    ```

   Quando si ha terminato l'installazione e l'esecuzione si può disattivare
   ```bash   
    deactivate
    ```
   
2. Installa Networkx e Matplotlib
```bash

pip install networkx
pip install matplotlib
```

## Esecuzione
1. Configurare opportunamente i valori nei file config_files.

2. Eseguire epidemic_sim.py:
   
    - Per lanciare una simulazione seir, senza creare strutture inutili:
        ```bash
        python3.6 epidemic_sim.py "seir"  n_s abs
        ```
        Esempio:
        ```bash
        python3.6 epidemic_sim.py "seir"  10 True
        ```

    - Per lanciare una simulazione con tracciamento:
        ```bash
        python3.6 epidemic_sim.py "tracing" n_s abs
        ```
        Esempio:
        ```bash
        python3.6 python epidemic_sim.py "tracing"  10 True
        ```
   
    - Per stampare i grafici del tracciamento dai file contenenti le medie:
        ```bash
        python3.6 epidemic_sim.py "result_from_avg"
        ```

    - Per stampare i risultati dai dati e calcolare le medie:
        ```bash
        python3.6 epidemic_sim.py "result"
        ```
## License
