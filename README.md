# F1 Solver

The python script `f1_solver.py` provides functionality to repoduce the experimental results from the paper “How Hard Is the Manipulative Design of Scoring Systems?” [[BH19]](#BH19).
They examine real-world race results from the Formula 1 from 1961 to 2008 and try to find out, if other drivers would have won the season with a different scoring system.
Therefore, they use the "PrefLib ED-00010: F1 and Skiing" [[Bre09]](#Bre09) dataset.

## Implementation

`f1_solver.py` provides a command line interface and works with SOI-files from [[Bre09]](#Bre09).
Results are printed to the command line, but can be saved into a file using pipes.
The implementation builds the ILP from Section 5 of [[BH19]](#BH19) and uses PuLP, a linear programming toolkit for Python, to solve it.

The following example calls `f1_solver.py` with data from 2008, adds restriction (I.) from the paper, activates more verbose output and redirects the printed results into `result.txt`.

```
python -m f1_solver './Data/ED-00010-00000048.soi' -r1 -v > result.txt
```

`f1_solver_test.py` contains some unit and integration tests.

## References

<a id="BH19">[BH19]</a>
Dorothea Baumeister and Tobias Hogrebe.
“How Hard Is the Manipulative Design of Scoring Systems?”
In: IJCAI. 2019, pp. 74–80.

<a id="Bre09">[Bre09]</a>
Robert Bredereck.
PrefLib ED-00010: F1 and Skiing. 2009.
URL: https://www.preflib.org/data/election/f1/ (visited on 05/11/2021).
