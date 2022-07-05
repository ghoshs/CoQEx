# CoQEx
1. [Introduction](#introduction)
2. [Data](#data)
3. [Code](#code)
4. [Citation](#citation)
5. [License](#license)

## Introduction <a name="introduction"></a>


merely answer these with a single, and sometimes puzzling number or return a ranked list of text snippets with different numbers.
<emph>CoQEx</emph> is a methodology for answering count queries with inference, contextualization and explanatory evidence. Unlike
previous systems, our method infers final answers from multiple observations, supports semantic qualifiers for the counts, and provides
evidence by enumerating representative instances.

Here we provide the data for our paper:
- <i>Answering Count Queries with Explanatory Evidence</i> Ghosh et al. SIGIR 2022 (<https://arxiv.org/pdf/2204.05039.pdf>).


## Data <a name="data"></a>

The data comprises:
- `CoQuAD_v1` dataset - training data for inference, groundtruth annoatation for infernce, contextualization and explanations, annotated dataset characteristics,

- `lcquad_v2` - subset of count queries from LC-QuAD v2 dataset with with groundtruth inference.

Run `./download.sh` to download all data in the dataset folder.

## Code <a name="code"></a>

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

More information on the methodology is available on our paper (<https://arxiv.org/pdf/2204.05039.pdf>).


## License <a name="license"></a>

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
