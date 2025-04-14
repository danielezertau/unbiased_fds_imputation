# Unbiased Functional-Dependencies-Based Imputation
In recent years, more data has become available for use by users. The sources of this data are usually unknown, and so it might be hard to reason about it. Moreover, the data might be incomplete, i.e. include missing values.

One option for using incomplete information is simply ignoring tuples with missing values. However, this approach might loose information, and can significantly reduce the amount of data.

In this project, we explore completing missing information by leveraging approximate functional dependencies (FDs) discovered by the TANE FD mining algorithm. Since mined FDs are prone to be biased, we leverage an LLM for trying to identify these FDs, and either disregard them or use them with care.

# Usage Guide
## Installation
Make sure you have python installed with version `>=3.11`. Using conda:
```shell
conda create -n fd-imp python=3.11
conda activate fd-imp
```
Install the project's dependencies:
```shell
pip install -r requirements.txt
```

After installing the dependencies, make sure you have an OpenAI API key, and create a file called `config/.env` with the following:
```dotenv
OPENAI_API_KEY=<YOUR_API_KEY>
```

## Imputing Data
The fd-imp-cli can be used in order to run the project.

### Arguments
- `--min_num_partitions`: The minimal number of partitions allowed for a functional dependency to be considered for imputation. Default is `2`.
- `--max_lhs_size`: The maximal number of attributes in the LHS of a functional dependency. Default is `3`. 
- `--error_threshold`: The allowed error rate for a functional dependency. Default is `0.06`.
- `--data_dir`: The directory in which we can find the input data in csv format. Default is `./data`.
- `--data_filename`: The input data filename (excluding the .csv suffix). Default is `adult-rand-1000`. 

### Output
Since mining for functional dependencies and querying an LLM on their bias is time intensive, a cache with the resulting FDs is created in the`${data_dir}/cache` directory. 

After running the imputation process, the imputed data will be written to the `${data_dir}/out` directory.

### Examples
Imputing data, considering FDs with maximal LHS size of `4` and minimal number of partitions of `5`:
```shell
python fd-imp-cli.py --max_lhs_size 4 --min_num_partitions 5
```

Imputing data with a large error threshold of `0.5`:
```shell
python fd-imp-cli.py --error_threshold 0.5
```

Imputing data with only exact FDs:
```shell
python fd-imp-cli.py --error_threshold 0
```

Imputing data without biased FDs, with an error threshold of 0.01:
```shell
python fd-imp-cli.py --error_threshold 0.01 --use_biased_fds False
```

Imputing data without biased FDs, using SimpleImputer with mean strategy as a final resort:
```shell
python fd-imp-cli.py --use_biased_fds False --use_simple_imputer True --simple_imputer_strategy mean
```

Imputing data while aggressively balancing biased FDs:
```shell
python fd-imp-cli.py --balancing_power 0.1
```

Imputing data slightly balancing biased FDs, ignoring most of the bias:
```shell
python fd-imp-cli.py --balancing_power 0.9
```
