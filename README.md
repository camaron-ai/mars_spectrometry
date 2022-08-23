# Mars Spectrometry: Detect Evidence for Past Habitability

repository for [Mars Spectrometry: Detect Evidence for Past Habitability](https://www.drivendata.org/competitions/93/nasa-mars-spectrometry/) competition hosted by DrivenData.

to run the code:

- download the competition data from the website
- add the root of the repository to the python path

```shell
export PYTHONPATH=${PYTHONPATH}:${HOME}/path-to-repo
```

- build the enviroment by running

```shell
 make setup-env
 ```

- write cross validation indices by running

```shell
 make write-cv-index
 ```

- run the notebooks in final-nbs in order.
