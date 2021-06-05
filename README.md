# F1 Solver

The python script `f1_solver.py` provides functionality to repoduce the experimental results from the paper “How Hard Is the Manipulative Design of Scoring Systems?” [[BH19]](#BH19).
They examine real-world race results from the Formula 1 from 1961 to 2008 and try to find out, if other drivers would have won the season with a different scoring system.
Therefore, they use the "PrefLib ED-00010: F1 and Skiing" [[Bre09]](#Bre09) dataset.

## Implementation

`f1_solver.py` offers a command line interface and works with SOI-files from [[Bre09]](#Bre09).
Results are printed to the command line, but can be saved into a file using pipes.
The implementation builds the ILP from Section 5 of [[BH19]](#BH19) and uses [PuLP](http://coin-or.github.io/pulp/), a linear programming toolkit for Python, to solve it.

The following example calls `f1_solver.py` with data from 2008, adds restriction (I.) from the paper, activates more verbose output and redirects the printed results into `result.txt`.

```
python -m f1_solver './Data/ED-00010-00000048.soi' -r1 -v > result.txt
```

`f1_solver_test.py` contains some unit and integration tests.

## Interpretation of Results

The following code block shows the results for the example call from above without verbose output.
Verbose output shows more information about the dataset and contains a line for each candidate `c` who cannot win: `Candidate c cannot win.`

```
Candidate  5 wins with distance  0.
Candidate  8 wins with distance 17.
Candidate 10 wins with distance 23.
Candidate 17 wins with distance  1.
```

The distances are the minimum Manhattan distances to the original scoring system.
More precisely, each distance is an optimal solution of the underlying ILP for one specific candidate.
There should always be exactly one candidate who wins with distance 0.
This is the winner from the original scoring system.
Further listed winners are unique winners for a different scoring vector.
On the contrary, an infeasible solution means that there is no alternative scoring system which makes candidate `c` a unique winner.

## References

<a id="BH19">[BH19]</a>
Dorothea Baumeister and Tobias Hogrebe.
“How Hard Is the Manipulative Design of Scoring Systems?”
In: IJCAI. 2019, pp. 74–80.
URL: https://www.ijcai.org/Proceedings/2019/11 (visited on 2021-06-05).

<a id="Bre09">[Bre09]</a>
Robert Bredereck.
PrefLib ED-00010: F1 and Skiing. 2009.
URL: https://www.preflib.org/data/election/f1/ (visited on 2021-05-11).
