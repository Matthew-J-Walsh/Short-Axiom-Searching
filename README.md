# Short-Axiom-Searching
 
Made by Matthew Walsh with the help of Branden Fitelson, Northeastern University.

Used to prove that Meredith's 21-symbol single axiom of classical sentential logic is the shortest possible, in addition to finding other axioms of length 21.

Used to prove that Meredith's 19-symbol single axioms of classical sentential logic with the fulsum operator are the shortest possible, in addition to finding other axioms of length 19.

Original preprint: http://fitelson.org/walsh.pdf

Setup with: python setup.py

Test with: (these should take at most 60s each)
python3 run.py BAAN 11 runs/baan_definition.json
python3 run.py BAC 11 runs/bac_definition.json
python3 run.py BACF 11 runs/bacf_definition.json
python3 run.py BACN 11 runs/bacn_definition.json
python3 run.py BAON 11 runs/baon_definition.json
python3 run.py C0 11 runs/c0_definition.json
python3 run.py C1 11 runs/c1_definition.json
python3 run.py CN 11 runs/cn_definition.json
python3 run.py HADN 11 runs/hadn_definition.json
python3 run.py LUK3VI 11 runs/luk3vi_definition.json
python3 run.py RAN 11 runs/ran_definition.json

Relevent maximums (known lowest axiom lengths):
python3 run.py CN 21 runs/cn_definition.json
python3 run.py C0 19 runs/c0_definition.json
#TODO: BAs