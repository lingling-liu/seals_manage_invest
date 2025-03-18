# Project Setup Instructions

Follow these steps to set up your environment and install the necessary dependencies for this project.

---

## 1. Create a New Environment

Create a new environment with Python 3.11.6:

```bash
mamba create -y -n wbt python=3.11.6
```

## 2. Activate the Environment

Activate the environment using:

```bash
mamba activate wbt
```

## 3. Install Necessary Packages

Install required packages from `conda-forge` using `mamba`:

```bash
mamba install -y -c conda-forge natcap.invest
mamba install -y -c conda-forge geopandas rasterstats netCDF4 cartopy xlrd markdown \
qtpy qtawesome plotly descartes pygeoprocessing taskgraph cython rioxarray dask \
google-cloud-datastore google-cloud-storage aenum anytree statsmodels openpyxl seaborn \
twine pyqt ipykernel imageio pandoc conda numba intake more-itertools \
google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2 gdown \
tqdm sympy gekko python-pptx
```

To ensure compatibility, uninstall and reinstall key geospatial libraries:

```bash
pip uninstall -y geopandas shapely fiona pygeos
pip install --upgrade --no-cache-dir geopandas shapely fiona
```

## 4. Install Local Packages

Navigate to the directories `hazelbean_dev` and `seals_dev`, then install the packages in development mode:

```bash
pip install -e hazelbean_dev
pip install -e seals_dev
```

## 5. Final Setup

After completing the steps above, your environment should be fully configured and ready to use.

---

## Additional Information

For troubleshooting or further setup details, refer to the project documentation or contact the team.

