# package: `vacoda.comtrade`
This module provides functionality to download trade data from the Comtrade Bulk API. It supports both single-query downloads and bulk downloads of multiple jobs defined in the configuration.

## Usage
To use the Comtrade downloader, you can configure it in your `comtrade.yaml` file. Here's an example configuration for bulk downloading:

```yaml
bulk:
  enabled: true
  endpoint: "https://comtradeapi.un.org/bulk/v1/get/"
  jobs:
    - name: "wood_panel_trade"
      params:
        typeCode: "C"
        freqCode: "A"
        clCode: "HS"
        partnerCode:
        cmdCode: "WOOD_PANEL"
        flowCode: "FLOW_ALL"
        period: "2007,2008"
      output_file: "wood_panel_trade.csv"
```
In this example, we define a bulk job named "wood_panel_trade" that will download trade data for wood panels (using the `cmdCode` filter) for the years 2007 and 2008. The downloaded data will be saved to `wood_panel_trade.csv` in the specified output directory.

## Authentication
The downloader supports authentication via API tokens. You can specify the authentication configuration in your `comtrade.yaml` file under the `auth` section. For example:

```yaml
auth:
  type: "api_token"
    token: "your_api_token_here"
  query_param_name: "token" # The name of the query parameter to use for the token (default is "token")
  env_file: ".env" # Optional path to a .env file containing the token
```

## Output
The downloaded files will be saved to the output directory specified in the configuration. A manifest file (`manifest.csv`) will also be created in the output directory, summarizing the status of each bulk job, including any errors encountered during the download process.

## Error Handling
If an error occurs during the download of a bulk job, the error message will be logged, and the manifest will indicate that the job encountered an error along with the error details. This allows you to easily identify and troubleshoot any issues with specific jobs in your bulk download configuration.

## Note
- The bulk endpoint does not support commodity filters like `cmdCode` or `cmsCode`. If you need to filter by commodity, you may need to perform multiple bulk downloads with different parameters or use the single-query endpoint instead.
- Ensure that your API token has the necessary permissions to access the Comtrade Bulk API and that you are adhering to any rate limits or usage policies set by the API provider.

## For more details on the Comtrade Bulk API and its parameters, please refer to the official documentation: https://comtradeapi.un.org/bulk/v1/

## Package information
This module has following structure:
```
vacoda/
├── comtrade/
│   ├── __init__.py
│   ├── README.md
│   ├── download.py
│   ├── utils.py
│   ├── api.py
│   ├── api_pipeline.py
│   ├── comtrade_pipeline.py
│   ├── utils.py
│   └── reference/
│       └── hs_sets.py
│       └── flow_sets.py
```

### download.py
This file contains the main logic for downloading data from the Comtrade Bulk API. It defines functions to handle both single-query and bulk downloads, manage authentication, and save the downloaded data to the specified output directory. The bulk download function iterates through the defined jobs in the configuration, attempts to download each job, and records the results in a manifest file for easy tracking of successful and failed downloads.

### utils.py
This file contains utility functions that support the download process, such as path resolution, authentication handling,
and any other helper functions needed for the download operations.

### api.py
This file contains functions that directly interact with the Comtrade API, including constructing API requests, handling responses, and managing any API-specific logic.

### api_pipeline.py
This file defines a data pipeline for processing API requests and responses. It may include functions for batching requests, handling retries, and processing the data before saving it.

### comtrade_pipeline.py
This file defines a data pipeline specific to the Comtrade data, including any transformations or processing steps needed to prepare the data for analysis or storage.

### reference/hs_sets.py
This file contains predefined sets of HS codes that can be used as filters in the API requests.

## Package structure overview
`api.py`
    Low-level normal API calls:
    - build URL
    - send request
    - handle response
    - pagination / count query if needed

`bulk.py`
    Bulk API handling:
    - download bulk files
    - inspect .bin/.gz files
    - decompress/read TSV files
    - filter cmdCode / flowCode locally

`download.py`
    High-level orchestration:
    - read config
    - decide normal API vs bulk
    - iterate jobs
    - call `api.py` or `bulk.py`
    - save outputs

`manifest.py`
    Save download status:
    - downloaded
    - failed
    - no_data
    - skipped_existing
    - too_large

`config.py`
    Load and validate comtrade.yaml:
    - resolve WOOD_PANEL
    - resolve FLOW_ALL
    - check required fields

`utils.py`
    Small shared helpers only:
    - ensure_dir
    - timestamp
    - safe_filename