# `vacoda.comtrade`

`vacoda.comtrade` provides a configuration-driven download pipeline for UN Comtrade data.

It supports two acquisition modes:

1. **Single API downloads**  
   Precise, query-based downloads from the normal Comtrade Data API.  
   Useful for specific commodity codes, selected years, and controlled experiments.

2. **Bulk API downloads**  
   Broad yearly downloads from the Comtrade Bulk API.  
   Useful for large-scale acquisition, especially when many commodity codes are needed.  
   Commodity and flow filters are applied locally after download, because the Bulk API does not support commodity-level filtering in the request itself.

The package includes:

- authentication preflight
- endpoint resolution
- retry handling
- pagination for single API downloads
- parallel downloads via `max_workers`
- gzip detection for bulk files
- timestamped logs
- user-facing configuration summary
- confirmation prompt before large downloads
- JSON manifests
- automatic manifest audit
- reference-token resolution such as `WOOD_PANEL` and `FLOW_ALL`

---

## Package structure

```bash
src/vacoda/comtrade/
├── __init__.py
├── README.md
├── api.py
├── config.py
├── download.py
├── manifest.py
├── pipeline.py
└── utils.py
```
### Related reference files:
```bash
src/vacoda/reference/
├── hs_sets.py
├── flow_sets.py
└── hs_version_set.py
```

### Related tests/ files:
```bash
tests/comtrade/
├── test_manifest.py
└── test_pipeline_expansion.py
```

## Main Entry point
The main script is:
```bash
python scripts/run_comtrade_download.py
```
The pipeline performs the following steps:
1. loads and summarizes the configuration
2. asks for user confirmation
3. validates authentication
4. runs bulk downloads if enabled
5. runs single API downloads if enabled
6. writes and audits manifest files
7. logs a final summary

## Configuration file
The main config file is
```bash
configs/comtrade.yaml
```
The configuration file defines the parameters for both single API and bulk downloads. It includes sections for authentication, output settings, and specific parameters for each download mode. The configuration allows you to specify the years, commodity codes, flow codes, and other filters for your downloads, as well as settings for parallelism and error handling.


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



## How to use

1. Update `configs/comtrade.yaml` with the desired download settings.

   Check the following before running the pipeline:

   - _Authentication_: make sure your Comtrade API token is available, usually through `.env` as `COMTRADE_API_KEY`.
   - _Mode selection_: choose whether to run `single`, `bulk`, or both. For large downloads, it is usually safer to enable only one mode at a time.
   - _Years_: configure year expansion with the `years` block.
   - _Commodity and flow filters_: use reference tokens such as `WOOD_PANEL` and `FLOW_ALL` where appropriate.
   - _Bulk local filters_: remember that the Bulk API does not apply commodity filters at the API level. Use `local_filter` to record filters that should be applied after download.
   - _Output settings_: verify the output directory and manifest filename.
   - _Runtime settings_: review `timeout_seconds`, `overwrite_existing`, and `max_workers`.
   - _Logging_: confirm the configured log level and log file location.

2. Run the main download script:

  ```bash
  python scripts/run_comtrade_download.py
  ```

3. Review the configuration summary.
  Before downloads start, the pipeline prints a summary of the selected configuration and asks for confirmation:
  ```text
  Proceed with this Comtrade download configuration? [y/N]:
  ```
  Enter `y` to proceed or `N` to abort and make changes to the configuration file.

4. Monitor progress in the console and log file.
  Each run creates a timestamped log file under:
  ```
  output/reports/comtrade_download_YYYYMMDD_HHMMSS.log
  ```

5. Check the manifest and audit summary.
  The downloaded files are saved to the configured output directory. The pipeline also writes a manifest file summarizing each job.

  Example manifest locations:
  ```
  data/raw/api/single/download_manifest_single.json
  data/raw/api/bulk/download_manifest_bulk.json
  ```
  At the end of the run, the pipeline audits the manifest and reports whether the download looks complete.

6.Troubleshoot errors if needed.
  If a download fails, check:
  * console output for error messages
  * the timestamped log file in `output/reports/`
  * the manifest file for any jobs marked with errors, which will include error details for further investigation


