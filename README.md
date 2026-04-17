# VectorMerge 🚀
![logo](./images/logo.png)

## Vector Embedding Dataset (about 1TB)

Precomputed vector embeddings are hosted at:

- https://huggingface.co/datasets/DB-Edinburgh/VectorBenchmark/

This dataset is access-controlled. Please fill in the required access information on Hugging Face and wait for approval before downloading.

If approval is delayed for a long time, please email `suchunsv@gmail.com` to remind us to approve your request.

# Generate Embeddings
## vectormerge generate-embedding

# Create Reference
## vectormerge create-reference
vectormerge create-reference -d [dataset_name] Optional(-m1 [model_name] -m2 [model_name] --m1--m2 [replace default m1 and m2]) --reference-strategy [reference_strategy] --reference-path [reference_path]

```
if reference strategy is random, we do not need to specify m1 and m2, because we can do it randomly.
if reference strategy is ours, we need to specify m1 and m2, because we need to do some clustering
the reference will output to the reference_path, 
```

# Cluster Reference

vectormerge cluster-reference 
-d [dataset_name] 
--reference-key -rk [reference_key]
--reference-path [reference_path] 
--cluster-method [cluster_method] 
--cluster-path [cluster_path]
--force [force retrain, default is use existing cluster]


# Map Embedding [Core]
## vectormerge map-embedding
vectormerge map-embedding -m1 [model_name] -m2 [model_name] --m1--m2 [replace default m1 and m2] -d [dataset_name] --load-reference-path [reference_path] --mapping-method [mapping_method] --mapping-config [mapping_config] --output-path [output_path]

-mapping-method [mapping_method]
-m1 [model_name]
-m2 [model_name]
--m1-m2 [replace default m1 and m2]
-d [dataset_name]
--reference-key [reference_key]
--load-reference-path [reference_path]
--mapping-config [mapping_config]
--output-path [output_path]
--force [force retrain, default is use existing mapping]


# Evaluate
## vectormerge evaluate

vectormerge evaluate 
-d [dataset_name]
-m1 [model_name]
-m2 [model_name]
--reference-key [reference_key]
--mapping-method [mapping_method]
--m1-m2 [replace default m1 and m2]
--reference-path [reference_path]
--mapping-config [mapping_config]
--output-path [output_path]
--force [force retrain, default is use existing mapping]
**kwargs**
