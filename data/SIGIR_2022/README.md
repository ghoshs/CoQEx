This is the data folder of the experiments published in SIGIR 2022.

The data comprises:

1. The CoQuAD_v1 dataset
	- direct download: <https://nextcloud.mpi-klsb.mpg.de/index.php/s/kiWC5wEgR9nxws5>
	- `CoQuAD_v1/inference` - train/test split for training count span prediction model
	- `CoQuAD_v1/characteristics` - annoations of the evaluation data for dataset charcateristics (Section 4)
	- `CoQuAD_v1/contextualization` - category annotations for count contexts (synonyms, subgroups, incomparables)
	- `CoQuAD_v1/explanation` - instance annotations for count explanation by instances.

2. Count queries from LCQuAD_v2
	- direct download: <https://nextcloud.mpi-klsb.mpg.de/index.php/s/9eDTnArAJmyLwkd>
	- `count_lcquad_v2/count_lcquadv2.csv` - the LCQuAD count queries with their count answers
	- `count_lcquad_v2/count_lcquadv2_bing_v1.json` - the snippets used by CoQEx.

3. Manually curated stresstest
	- direct download: <https://nextcloud.mpi-klsb.mpg.de/index.php/s/G5NfyMo4DARJGAF>
	- `stresstest_v1/stresstest.csv` - the manually curated stresstest queries with count answers.
	- `stresstest_v1/stresstest_bing_v1.json` - the snippets used by CoQEx.

Run `./download.sh` to download all data in your dataset folder. Alternately you can individually download the above datasets from their direct links. 