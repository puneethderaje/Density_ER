Column descriptions:

Block: One of two temporal blocks (4a or 4b).
PopID: A number given to each population within a treatment and block
PopIDUniq: A unique alphanumeric identifier for a population
media: Percentage of corn flour used in the media the population was raised on: either 99.8 or 98.2
richpoor: Either R (for rich, 98.2% corn media) or P (for poor, 99.8% corn media)
size: Founding population size: either L (for large, 40 individuals) or S (for small, 10 individuals)
N0-N10: Population size at the given generation (0-10), with N0 representing the founding population size.  A value of 0 indicates a population became extinct at that generation, while an empty cell indicates the population was not surveyed (due to previous extinction or experimenter error, see notes)
treat: The treatment applied, either evo (for evolution) or noevo (for the constrained evolution controls).
evomixgen9: The mixing treatment the applied on the 8th generation, has values mixed, notmixed, and noevo for populations set up in the 8th generation, and is blank for populations not set up
extinct: 1 if a population became extinct at any point in the experiment, 0 otherwise
psuedo: 1 if there were 2 or fewer individuals in generation 9, 0 otherwise
genexinc: The generation of extinction
notes: various problems (mostly with the mix-no mix experiment, for example "not set up" means for some reason the 10th gen wasn't set up)
flag: data recording problems