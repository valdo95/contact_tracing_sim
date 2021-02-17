# Contact Tracing Simulator

E' un'implementazione Python di un Simulatore SEIR stocastico a cui vengono integrati diversi tipi tracciamento digitale su un grafo dinamico che rappresenta una rete di prossimità in un contesto urbano.

## Istallazione

1. Install Python 3.6
2. Install EoN (non indispensabile)
```bash
pip install EoN
python epidemic_sim.py
```

## Esecuzione
1. Configurare opportunamente i valori nei file config_files.

2. Eseguire epidemic_sim.py:
   
    - Per lanciare una simulazione seir, senza creare strutture inutili:
        ```bash
        python python epidemic_sim.py "seir"  n_s abs
        ```
        Esempio:
        ```bash
        python python epidemic_sim.py "seir"  10 True
        ```

    - Per lanciare una simulazione con tracciamento:
        ```bash
        python epidemic_sim.py "tracing" n_s abs
        ```
        Esempio:
        ```bash
        python python epidemic_sim.py "tracing"  10 True
        ```
   
    - Per stampare i grafici del tracciamento dai file contenenti le medie:
        ```bash
        python python epidemic_sim.py "result_from_avg"

    - Per stampare i risultati dai dati e calcolare le medie:
        ```bash
        python python epidemic_sim.py "result"
        ```
## License
