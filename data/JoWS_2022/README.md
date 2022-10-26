This is the data folder of the experiments submitted to JoWS 2022.

The data comprises:

1. The CoQuAD_v2 dataset
	- direct download: <https://nextcloud.mpi-klsb.mpg.de/index.php/s/Qp92Nc9irdKDZre/download>
	- `CoQuAD_v2/inference` - train/test split for training count span prediction model
	- `CoQuAD_v2/characteristics` - annotations of the evaluation data for dataset charcateristics
	- `CoQuAD_v2/contextualization` - category annotations for count contexts (synonyms, subgroups, incomparables)
	- `CoQuAD_v2/explanation` - instance annotations for count explanation by instances and their aliases.

2. Count queries from LCQuAD_v2
	- direct download: <https://nextcloud.mpi-klsb.mpg.de/index.php/s/9eDTnArAJmyLwkd>
	- `count_lcquad_v2/count_lcquadv2.csv` - the LCQuAD count queries with their count answers
	- `count_lcquad_v2/count_lcquadv2_bing_v1.json` - the snippets used by CoQEx.

3. Manually curated stresstest
	- direct download: <https://nextcloud.mpi-klsb.mpg.de/index.php/s/G5NfyMo4DARJGAF>
	- `stresstest_v1/stresstest.csv` - the manually curated stresstest queries with count answers.
	- `stresstest_v1/stresstest_bing_v1.json` - the snippets used by CoQEx.

4. Count queries from Natural Questions
	- direct download: <https://nextcloud.mpi-klsb.mpg.de/index.php/s/HRd6L9XRyYxZ8rW/download>
	- `natural_questions/inference` - count queries from the Natural Questions dataset with count answers.
	- `natural_questions/explanation` - instance annotations for count explanation by instances and their aliases.
 
Run `./download.sh` to download all data in your dataset folder. Alternately you can individually download the above datasets from their direct links. 