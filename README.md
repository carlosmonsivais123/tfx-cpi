# tfx-cpi

# run_it.py
# This will take care of the following:
    1. Reading in data from BLS (Inlcudes Data Cleaning and Reformatting)
    2. Pushing Data to BQ
    3. Splitting Data into GCS Buckets --> Training: Every item has a bucket --> Testing: Every item has a bucket

# run_it2.py
# This will take care of the following:
    1. TFX Pipeline
    2. Start out wiht simple Linear Regression Model
    3. Every item will have it's own pipleline (Need to count how many Items)
    4. Run Pipelines in Vertex-AI

# run_it3.py
# This will take care of the following:
    1. Take care of looking at the metadata for every pipeline
    2. Make sure no pipelines fail
    3. Return back a log as a csv inlcuding the following --> * Pipelines Succeded/Failed
                                                              * Model Accuracy per pipeline (Training Data)
                                                              * Forecast 1 year
                                                              * Data Aggregation to see which item is the most important to the index