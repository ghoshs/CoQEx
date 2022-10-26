# CoQEx
1. [Introduction](#introduction)
2. [Data](#data)
3. [Code](#code)
4. [Citation](#citation)
5. [License](#license)

## Introduction <a name="introduction"></a>

We introduce CoQEx as a method to answering queries on entity counts by consolidating evidences across multiple text snippets.
The CoQEx methodology uses a span-based QA model to separately extract candidate count contexts and instances from the top-50 search-engine snippets. The user is shown the following components.

<li>An <strong>answer inference</strong> is predicted by a distribution-aware inference over count contexts. </li>
<li>
    The count contexts are further classified into semantic groups with respect to the inferred answer to form the <strong>explanation by contexts</strong>. They are grouped based on whether the contexts are quite similar to the inferred answer or if they represent a subset of the inferred answer or if they are incomparable. 
</li>
<li>
    The instances are ranked by their compatibility with the answer type. They form the <strong>explanation by instances</strong> since they likely ground the counts into their constituting entities. CoQEx extracts the answer type from the query.                  
</li>
<li>
    The snippets are annotated with the count context and instance candidates to form the <strong>explanation by provenance</strong>.
</li>
                

Unlike previous systems, our method infers final answers from multiple observations, supports semantic qualifiers for the counts, and provides evidence by enumerating representative instances.

Here we provide the data for our paper:
- <i>Answering Count Queries with Explanatory Evidence</i> Ghosh et al. SIGIR 2022 (<https://dl.acm.org/doi/pdf/10.1145/3477495.3531870>).
- <i>Answering Count Queries with Structured Answers from Text</i> Ghosh et al. submitted to JoWS (<https://arxiv.org/pdf/2209.07250.pdf>)

And the code for setting up an interactive user demonstration.


## Data <a name="data"></a>

The [data](https://github.com/ghoshs/CoQEx/tree/main/data) for the SIGIR'22 and JoWS experiments can be found in the respective folders. The dayaset comprises:

- `CoQuAD` dataset - training data for inference, groundtruth annoatation for infernce, contextualization and explanations, annotated dataset characteristics. Version 2 contains more evaluation data and query characteristics.

- `stresstest` - a set of hand-curated challenging count queries with groundtruth inferences.

- `lcquad_v2` - subset of count queries from LC-QuAD v2 dataset with groundtruth inferences.

- `natural_questions` - a subset of count queries from Natural Questions dataset with groundtruth inferences.

                
Run `./download.sh` to download all data in the dataset folder or download them individually.

## Code <a name="code"></a>

The code for setting up a Flask app of CoQEx is included in the folder `coqex/`.
 
A pipeline for executing CoQEx coming up soon.

## Citation <a name="citation"></a> 

If you use our work please cite us:

```bibtex
@inproceedings{ghosh2022answering,
    title = "Answering Count Queries with Explanatory Evidence",
    author = "Shrestha Ghosh and Simon Razniewski and Gerhard Weikum",
    booktitle = "To appear at SIGIR 2022",
    month = jul,
    year = "2022",
}
```


## License <a name="license"></a>

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
