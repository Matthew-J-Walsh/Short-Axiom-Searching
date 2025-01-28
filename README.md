# Short-Axiom-Searching
 
Made by Matthew Walsh with the help of Branden Fitelson, Northeastern University.

Used to prove that Meredith's 21-symbol single axiom of classical sentential logic is the shortest possible, in addition to finding other axioms of length 21.

Used to prove that Meredith's 19-symbol single axioms of classical sentential logic with the fulsum operator are the shortest possible, in addition to finding other axioms of length 19.

Original preprint: http://fitelson.org/walsh.pdf

## Installation

```git clone https://github.com/Matthew-J-Walsh/Short-Axiom-Searching```

```cd Short-Axiom-Searching```

```python setup.py```

Use "python setup.py --include_libraries" if you have a fresh python install without numpy and scipy

Testing with:

```python test.py```

## Relevent maximum relevent runs (known lowest axiom lengths):

```python run.py CN 21 runs/cn_definition.json```

```python run.py CT 19 runs/ct_definition.json```

```python run.py BACN 17 runs/bacn_definition.json```

```python run.py BAON 20 runs/baon_definition.json```

```python run.py BAAN 20 runs/baan_definition.json```

```python run.py BACF 17 runs/bacf_definition.json```

