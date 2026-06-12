"""
scripts.run_vacoda_pipeline
Thin wrapper around vacoda.pipeline.cli.main() to run the VACODA master pipeline.
"""


from vacoda.pipeline.cli import main

if __name__ == "__main__":
    main()