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
   
2. Installa Networkx, Matplotlib e Yaml
```bash

pip install networkx
pip install matplotlib
pip install PyYAML
```

## Esecuzione
1. Configurare opportunamente i valori nei file config_files.

2. Eseguire epidemic_sim.py:
   
    - Per lanciare una simulazione seir sfruttando n core in parallelo:
        ```bash
        python3.x epidemic_sim.py "multiprocess" "seir" n_s abs
        ```
        - n_s è il numero di simulazione per core
        - abs: True se si vuole che i risultati plottati siano percentuali, False altrimenti 
    
        Esempio:
        ```bash
        python3.8 epidemic_sim.py "multiprocess" "tracing"  10 True
        ```
    - Per lanciare una simulazione con tracciamento sfruttando n core in parallelo:
        ```bash
        python3.x epidemic_sim.py "multiprocess" "tracing" n_s abs
        ```
        Esempio:
        ```bash
        python3.8 python epidemic_sim.py "multiprocess""tracing"  10 True
        ```
   
    - Per stampare i grafici del tracciamento dai file contenenti le medie:
        ```bash
        python3.8 epidemic_sim.py "result_from_avg"
        ```

    - Per stampare i risultati dai dati e calcolare le medie:
        ```bash
        python3.8 epidemic_sim.py "result"
        ```
## License
