
# Project Setup Instructions

Follow these steps to set up your environment and install the necessary dependencies for this project.

## 1. Create a New Environment

To create a new environment with Python 3.11.6, run the following command:

```bash
mamba create -y -n wbt python=3.11.6
```

## 2. Activate the Environment

Activate the environment using the command:

```bash
mamba activate wbt
```

## 3. Install Necessary Packages

Install the required packages from `conda-forge` using `mamba`:

```bash
mamba install -y -c conda-forge natcap.invest
mamba install -y -c conda-forge geopandas rasterstats netCDF4 cartopy xlrd markdown qtpy qtawesome plotly descartes pygeoprocessing taskgraph cython rioxarray dask google-cloud-datastore google-cloud-storage aenum anytree statsmodels openpyxl seaborn twine pyqt ipykernel imageio pandoc conda numba intake more-itertools google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2 gdown tqdm sympy gekko python-pptx
```

## 4. Install Local Packages for `hazelbean` and `seals`

Once the environment is set up, navigate to the specific directories (`hazelbean_dev` and `seals_dev`), and install the packages with the following `pip` commands for development:

```bash
pip install -e hazelbean_dev
pip install -e seals_dev
```

## 5. Final Setup

After completing the steps above, your environment should be ready to use with all the necessary dependencies installed for your project.

---

### Additional Information

For any further setup or troubleshooting, please refer to the project documentation or reach out to the team.
